"""Chain-Hybrid (EAGLE3 + Suffix) per-depth online construction patch.

At each EAGLE3 chain draft step (speculative_eagle_topk=1), the suffix
decoding cache proposes one candidate token alongside eagle3's top-1.
The candidate with the higher per-token probability wins:

    choose suffix iff S(suffix) > eagle_p   (and the tokens differ)

where eagle_p is the draft model's full-vocab softmax prob (conditional on
the chain so far) and S(suffix) is the suffix draft's first-token score.
By default S(suffix) = suffix_p, the trie edge count ratio (the original
select-1). When SGLANG_CHAIN_HYBRID_CALIB names a frozen isotonic map,
S(suffix) is the calibrated P(accept) the map assigns to that edge prob
(optionally Jeffreys count-shrunk first) — putting it on the same scale as
eagle_p instead of comparing a raw count ratio against a softmax prob. The
chosen token is written into the draft loop's ``topk_index`` / ``topk_p``
before ``select_top_k_tokens`` consumes them, so the next draft forward
continues the chain from the winner and the verify input is assembled with
zero downstream changes.

Trie semantics follow ArcticInference's official usage: tokens committed by
verify are fed to ``add_active_response`` immediately every step
(incremental), not batched at request finish.

Activation (all required, validated at patch time):
  * SGLANG_CHAIN_HYBRID=1  (read by oracle_patch.patch_eagle_worker_full)
  * SGLANG_LATENCY_ONLY=1  (real speculative decoding; vanilla force-accept
    would nullify the experiment)
  * --speculative-eagle-topk 1  (chain mode)
  * --disable-cuda-graph  (the per-step GPU->CPU sync requires the eager
    draft path; with graphs enabled draft_forward is bypassed entirely and
    the patch would be a silent no-op)

Decision log (JSONL, SGLANG_CHAIN_HYBRID_LOG, tp_rank 0 only):
  {"type": "decision", "rid": ..., "decode_step": N, "depth": i,
   "eagle_token": ..., "eagle_p": ..., "eagle_p_cal": p|null,
   "suffix_token": ..., "suffix_p": ..., "suffix_count": c|null,
   "suffix_total": n|null, "suffix_p_cal": p|null, "match_len": ...,
   "suffix_score": ..., "chosen": "eagle3"|"suffix", "agreement": bool|null}
  {"type": "step", "rid": ..., "decode_step": N, "accept_len": a}

suffix_count (c) and suffix_total (n) are the raw trie counts for the first
suffix edge (suffix_p = c/n); fit_chain_hybrid_calib.py uses them to build
both the raw and Jeffreys-shrunk calibration maps. suffix_p_cal is the
calibrated score actually compared against eagle_p (null when uncalibrated).

Suffix tail append (SGLANG_CHAIN_HYBRID_TAIL=<T_max>, 0/unset = off):
after the S-step head chain is fully drafted, the suffix cache is queried
once more with (context + full chain) and the returned continuation run
(up to T_max tokens, bounded by SGLANG_CHAIN_HYBRID_TAIL_FACTOR x match_len
and SGLANG_CHAIN_HYBRID_TAIL_MIN_PROB) is appended to the linear chain in
the EagleVerifyInput — verify scores S+1+t tokens in ONE target forward,
with zero extra eagle forwards. Requires bs=1 (--max-running-requests 1)
and the triton target attention backend. Two sglang-internal constants are
handled per step: the triton backend's num_draft_tokens (overridden around
verify) and the draft-extend kernel's next_power_of_2(num_steps+1)
constexpr (speculative_num_steps temporarily bumped by t during
forward_draft_extend_after_decode; silent position truncation otherwise).
Tail decision records carry "tail": true and depth = S+k; their suffix_p is
the CUMULATIVE path probability (not a single-edge count ratio), so the
calibration fitter must exclude them. The join rule is unchanged: tail
token at depth d accepted iff accept_len >= d+1 (accept_len may now exceed
S). Only validated for greedy sampling (temperature 0).
SGLANG_CHAIN_HYBRID_TAIL_CHECK=1 additionally verifies, every step, that
the hand reconstruction of positions/retrive_*/custom_mask at the original
size bit-matches the build kernel's output before extending (debug).

Join rule for offline analysis: the decision at depth d (0-based) was
accepted by verify iff the same (rid, decode_step)'s accept_len >= d + 1.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.eagle_worker import EAGLEWorker

logger = logging.getLogger(__name__)

# Stale-request GC: every GC_INTERVAL decode batches, stop_request() any rid
# not seen for GC_MAX_IDLE batches (covers aborted/retracted requests that
# never reach req.finished()).
GC_INTERVAL = 512
GC_MAX_IDLE = 2048


class _ChainHybridState:
    """Per-process state shared by the hooks (one EAGLEWorker/process)."""

    def __init__(self, eagle_worker, suffix_cache, log_path: str,
                 tail_max: int = 0, tail_factor: float = 4.0,
                 tail_min_prob: float = 0.1, tail_check: bool = False,
                 mode: str = "select1", score_threshold: float = 5.0,
                 fb_factor: float = 1.0, fb_min_prob: float = 0.1):
        self.worker = eagle_worker
        self.cache = suffix_cache
        self.log_path = log_path
        # Decision mode: "select1" = per-depth competition (default);
        # "score_fallback" = Arctic-style per-STEP hybrid — one suffix run is
        # drafted from the committed context; if its score >= threshold the
        # run occupies the chain (eagle fills depths past the run end), else
        # the step is pure eagle3/MTP. Mirrors the simulator's hybrid_e3:t
        # (paper-faithful defaults F=1.0, T=0.1; score = sum of path probs).
        # "record" = pure pass-through baseline that dumps each finished
        # request's (input_ids, output_ids) to SGLANG_CHAIN_HYBRID_GT_OUT —
        # the greedy ground-truth source for the oracle arm.
        # "oracle" = per-depth selection ORACLE: ground-truth token gt[L+d]
        # (from a record-arm dump, request matched by exact input_ids) decides
        # the chain — eagle if it matches, else suffix (injected) if it
        # matches, else the chain is dead at d. The arm's accept_mean IS the
        # selection-policy ceiling. Requires the agent to --replay the record
        # arm's conversation so prompts match byte-exactly.
        self.mode = mode
        self.score_threshold = score_threshold
        self.fb_factor = fb_factor
        self.fb_min_prob = fb_min_prob
        self.fallback_runs: list | None = None    # per-row token runs (or None)
        self.fallback_meta: list | None = None    # per-row (score, probs, match_len)
        # record/oracle mode state
        self.gt_out_path: str | None = None       # record: dump path
        self.gt_map: dict | None = None           # oracle/pin: input_ids -> output_ids
        # PIN mode: a select1/score_fallback arm whose committed tokens are
        # forced onto a standalone trajectory (gt_map) so every arm follows the
        # SAME path (no FP-tie divergence), while still building its own draft
        # chain and recomputing its own eagle features. Reuses the oracle
        # GT-forcing machinery (verify override + post-verify rewrite) WITHOUT
        # the oracle selection logic (the chain is built by the calib/raw rule).
        self.pin_active = False
        self.gt: dict = {}                        # rid -> gt output list (or None)
        self.gt_offtrack: dict = {}               # rid -> bool (FP divergence)
        self.gt_pos: list | None = None           # per-row L at stash time
        # Per-row GT continuation [num_steps+1] used to OVERRIDE the verify
        # target_predict (argmax) in oracle mode -> commit follows GT exactly
        # (no FP-tie divergence/runaway); see _install_verify_greedy_oracle.
        self.gt_predict_override: list | None = None
        self.gt_stats = {"matched": 0, "unmatched": 0, "offtrack": 0}
        # Suffix tail-append config (0 = disabled = original behavior).
        self.tail_max = tail_max
        self.tail_factor = tail_factor
        self.tail_min_prob = tail_min_prob
        self.tail_check = tail_check
        self.last_tail_len = 0   # tokens appended this step; consumed by the
                                 # draft-extend wrapper, reset at each draft
        self.active: set = set()        # rids with start_request() done
        self.last_out_len: dict = {}    # rid -> len(output_ids) already fed to trie
        self.decode_step: dict = {}     # rid -> decode step counter (1-based)
        self.last_seen: dict = {}       # rid -> batch_counter at last sighting
        self.batch_counter = 0
        # Per-draft-call transient state (set by the draft wrapper, consumed
        # by the select_top_k_tokens wrapper, cleared when draft returns).
        self.stash: list | None = None      # [(rid, context_tail)] in batch.reqs order
        self.chains: list | None = None     # per-row chain tokens chosen so far
        self.pending: list = []             # records awaiting flush
        self._warned: set = set()
        # Online-calibration state (set in patch_chain_hybrid when active).
        self.online: "_OnlineWindowCalibrator | None" = None
        self.online_pairs_path: str | None = None
        self.online_pending: list = []      # online_pairs rows awaiting flush
        # Transient per-step decision store for the post-verify q_target join.
        # flush() empties self.pending before chain_forward runs, so the join
        # reads this instead. rid -> {depth: (eagle_tok, eagle_p, suffix_tok,
        # suffix_p, eagle_cmp, suffix_cmp, chosen)}.
        self.last_decisions: dict = {}
        self.last_num_draft: int | None = None  # spec_info.draft_token_num (q-join offset)

    def warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(f"chain-hybrid [{key}]: {msg}")

    def flush(self) -> None:
        if not self.pending:
            return
        records, self.pending = self.pending, []
        if getattr(self.worker, "tp_rank", 0) != 0:
            return
        try:
            with open(self.log_path, "a") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
        except OSError as e:
            self.warn_once("log-write", str(e))

    def flush_online(self) -> None:
        if not self.online_pending or not self.online_pairs_path:
            self.online_pending = []
            return
        records, self.online_pending = self.online_pending, []
        if getattr(self.worker, "tp_rank", 0) != 0:
            return
        try:
            with open(self.online_pairs_path, "a") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
        except OSError as e:
            self.warn_once("online-pairs-write", str(e))


_STATE: _ChainHybridState | None = None


# ---------------------------------------------------------------------------
# Suffix-probability calibration (optional)
# ---------------------------------------------------------------------------
#
# A faithful serving-side port of run_tree_oracle_sim._FrozenIsoCalibrator:
# loads the same JSON map ({"meta": {"shrink": ...}, "groups": {"suffix":
# {"x": [...], "y": [...]}}}) produced by fit_chain_hybrid_calib.py /
# fit_iso_calibration.py and does the identical step-function lookup
# raw_edge_prob -> P(accept). When the map was fitted with Jeffreys
# count-shrink (meta.shrink truthy), the caller must shrink the suffix edge
# prob the same way BEFORE lookup — mirrored here via ``wants_shrunk``.

class _ServingIsoCalibrator:
    """Frozen step-function calibration map p -> P(accept), loaded once at patch
    time. Two layouts, auto-detected from meta.per_position:
      global        groups[grp] = {"x":[...],"y":[...]}            (one curve/grp)
      per-position  groups[grp] = {"<depth>": {"x":[...],"y":[...]}, ...}
    Any calibrator family (histogram/isotonic/logistic/beta) works — the map is
    just a dense sampled (x ascending -> y) lookup; predict() never assumes a
    fit family. predict(group, p, fallback, depth) ignores depth for global maps
    and picks the depth's curve (nearest fitted depth <= depth, else deepest) for
    per-position maps."""

    def __init__(self, blob: dict):
        import numpy as np
        self.per_position = bool(blob.get("meta", {}).get("per_position"))
        if self.per_position:
            self._maps = {
                grp: {int(d): (np.asarray(m["x"], dtype=np.float64),
                               np.asarray(m["y"], dtype=np.float64))
                      for d, m in dd.items()}
                for grp, dd in blob["groups"].items()
            }
        else:
            self._maps = {
                grp: (np.asarray(m["x"], dtype=np.float64),
                      np.asarray(m["y"], dtype=np.float64))
                for grp, m in blob["groups"].items()
            }
        self.wants_shrunk = bool(blob.get("meta", {}).get("shrink"))

    @classmethod
    def load(cls, path: str) -> "_ServingIsoCalibrator":
        with open(path) as f:
            return cls(json.load(f))

    @staticmethod
    def _lookup(xs, ys, p: float) -> float:
        import numpy as np
        i = int(np.searchsorted(xs, p, side="right")) - 1
        if i < 0:
            i = 0
        v = float(ys[i])
        return v if v > 1e-6 else 1e-6

    def predict(self, group: str, p: float, fallback: float,
                depth: int = 0, **_) -> float:
        m = self._maps.get(group)
        if m is None:
            return fallback
        if self.per_position:
            if depth in m:
                xs, ys = m[depth]
            else:  # fallback: nearest fitted depth <= depth, else deepest
                le = [d for d in m if d <= depth]
                xs, ys = m[max(le)] if le else m[max(m)]
            return self._lookup(xs, ys, p)
        xs, ys = m
        return self._lookup(xs, ys, p)


_CALIB: _ServingIsoCalibrator | None = None


# Histogram bin grid for the online calibrator — fixed-width, matches
# plot_calib_reliability.BIN_W (0.05 -> 20 bins) so the served histogram
# calibrator and the offline per-depth graphs use identical bins.
_ONLINE_NB = 20
_ONLINE_BIN_W = 1.0 / _ONLINE_NB


class _OnlineWindowCalibrator:
    """Causal sliding-window calibrator to the target_p objective.

    Same predict(group, p, fallback, depth) interface as _ServingIsoCalibrator
    (so the select hook is unchanged), but the per-(group, depth) map is learned
    ONLINE from a rolling window of (raw_prob, q_target) pairs ingested AFTER each
    verify. A decision at step T only ever reflects ingests from steps < T (the
    select hook runs during draft, before chain_forward ingests this step) — so it
    is causal. The window is measured in CLOCK units (the global batch_counter for
    continuous scope; per-rid decode_step for per_request scope); samples whose
    clock falls outside [now-window, now] have their contribution removed exactly.
    One algorithm per instance:
      histogram  -> incremental per-bin (count, ysum); predict = bin mean. O(1).
      isotonic   -> plot_calib_methods.fit_isotonic (handles continuous y).
      logistic   -> fit_chain_hybrid_calib_perpos._fit_continuous (LinearRegression on p).
      beta       -> _fit_continuous (LinearRegression on [ln p, ln(1-p)]).
    Cold start: a (group,depth) cell with < min_samples (or, for histogram, an
    empty bin) returns the raw fallback -> behaves exactly like raw select1 until
    the window fills.
    """

    def __init__(self, method: str, window: int, min_samples: int = 50,
                 refit_k: int = 0, scope: str = "continuous",
                 label: str = "target_p", conditional: bool = False):
        import collections
        if method not in ("histogram", "isotonic", "logistic", "beta"):
            raise RuntimeError(f"unknown online calib method {method!r}")
        if label not in ("target_p", "accept_rate"):
            raise RuntimeError(f"unknown online label {label!r}")
        self.method = method
        self.window = int(window)
        self.min_samples = int(min_samples)
        # label: "target_p" regresses raw_prob onto q_target (continuous, the
        # target's softmax prob of the drafted token); "accept_rate" onto the
        # binary accept event (drafted token == target's argmax at that row).
        # conditional: ingest a depth only if the realized chain prefix was
        # accepted through it (accept_len >= depth) -> per-step CONDITIONAL accept.
        self.label = label
        self.conditional = bool(conditional)
        # histogram/isotonic are cheap -> refit every step; logistic/beta refit
        # every refit_k steps to bound wall-clock (MAT is unaffected by calibrator
        # latency, only wall-clock is).
        self.refit_k = (int(refit_k) if refit_k and int(refit_k) > 0
                        else (1 if method in ("histogram", "isotonic") else 8))
        self.scope = scope
        self.wants_shrunk = False  # target_p maps are never Jeffreys-shrunk
        self._buf = collections.defaultdict(collections.deque)  # key->deque[(p,q,clock)]
        self._fitted: dict = {}    # key->predict callable (iso/logistic/beta)
        self._dirty: set = set()
        self._bins: dict = {}      # key->(count[NB], ysum[NB])  (histogram)
        self._fitfn = None         # lazily-imported fit function
        self.ingested = 0          # diagnostics

    @staticmethod
    def _bin_idx(p: float) -> int:
        i = int(float(p) / _ONLINE_BIN_W)
        if i < 0:
            return 0
        return _ONLINE_NB - 1 if i >= _ONLINE_NB else i

    def predict(self, group: str, p: float, fallback: float, depth: int = 0,
                **_) -> float:
        key = (group, int(depth))
        buf = self._buf.get(key)
        if buf is None or len(buf) < self.min_samples:
            return fallback
        if self.method == "histogram":
            cb = self._bins.get(key)
            if cb is None:
                return fallback
            count, ysum = cb
            b = self._bin_idx(p)
            if count[b] <= 0:
                return fallback
            v = float(ysum[b] / count[b])
            return v if v > 1e-6 else 1e-6
        fn = self._fitted.get(key)
        if fn is None:
            return fallback
        try:
            # fitted predictors (isotonic/_fit_continuous) expect a >=1-d input
            # (sklearn isotonic rejects scalars); call with a 1-element list.
            out = fn([float(p)])
            v = float(out[0]) if hasattr(out, "__len__") else float(out)
        except Exception:
            return fallback
        v = min(max(v, 0.0), 1.0)
        return v if v > 1e-6 else 1e-6

    def ingest(self, group: str, depth: int, raw_prob: float, q: float,
               clock: int) -> None:
        import numpy as np
        key = (group, int(depth))
        buf = self._buf[key]
        buf.append((float(raw_prob), float(q), int(clock)))
        self.ingested += 1
        if self.method == "histogram":
            cb = self._bins.get(key)
            if cb is None:
                cb = (np.zeros(_ONLINE_NB), np.zeros(_ONLINE_NB))
                self._bins[key] = cb
            count, ysum = cb
            b = self._bin_idx(raw_prob)
            count[b] += 1.0
            ysum[b] += float(q)
        cutoff = int(clock) - self.window
        while buf and buf[0][2] <= cutoff:
            op, oq, _oc = buf.popleft()
            if self.method == "histogram":
                count, ysum = self._bins[key]
                ob = self._bin_idx(op)
                count[ob] -= 1.0
                ysum[ob] -= oq
                if count[ob] < 0:
                    count[ob] = 0.0  # fp guard
        self._dirty.add(key)

    def _get_fit_fn(self):
        if self._fitfn is not None:
            return self._fitfn
        import sys
        from pathlib import Path
        sd = str(Path(__file__).resolve().parents[1] / "scripts")
        if sd not in sys.path:
            sys.path.insert(0, sd)
        if self.method == "isotonic":
            # isotonic handles both continuous q_target and 0/1 accept natively
            from plot_calib_methods import fit_isotonic
            self._fitfn = lambda p, y: fit_isotonic(p, y)[3]
        elif self.label == "accept_rate":
            # binary 0/1 label -> the REAL logistic/beta calibrators
            from plot_calib_methods import fit_logistic, fit_beta
            fn = fit_logistic if self.method == "logistic" else fit_beta
            self._fitfn = lambda p, y, _fn=fn: _fn(p, y)[3]
        else:  # target_p continuous -> LinearRegression analog (NOT real logistic)
            from fit_chain_hybrid_calib_perpos import _fit_continuous
            m = self.method
            self._fitfn = lambda p, y, _m=m: _fit_continuous(_m, p, y)
        return self._fitfn

    def maybe_refit(self, clock: int) -> None:
        if self.method == "histogram":
            return  # incremental — nothing to batch-refit
        if int(clock) % self.refit_k != 0 or not self._dirty:
            return
        import numpy as np
        try:
            fitfn = self._get_fit_fn()
        except Exception as e:
            logger.warning(f"online-calib fit import failed ({self.method}): {e}")
            self._dirty.clear()
            return
        for key in list(self._dirty):
            buf = self._buf.get(key)
            if not buf or len(buf) < self.min_samples:
                continue
            n = len(buf)
            p = np.fromiter((s[0] for s in buf), dtype=np.float64, count=n)
            y = np.fromiter((s[1] for s in buf), dtype=np.float64, count=n)
            try:
                self._fitted[key] = fitfn(p, y)
            except Exception:
                pass  # keep prior fit; treat as cold only if never fit
        self._dirty.clear()

    def reset_request(self) -> None:
        """per_request scope: drop all windowed state at a new request."""
        self._buf.clear()
        self._fitted.clear()
        self._dirty.clear()
        self._bins.clear()


class _OnlineMultiFeatCalibrator:
    """COMBO: online sliding-window MULTI-FEATURE per-proposer calibrator with the
    accept-conditioned (alive-prefix) ingest. Like _OnlineWindowCalibrator but each
    group keeps a window of (feature-vector, y) and refits a standardized logistic
    (accept_rate) / linear (target_p) model instead of a 1-D map:
      eagle  features = [eagle_p, depth]
      suffix features = [suffix_p, log1p(count), log1p(total), match_len, depth]
    No training/frozen map — learned at serving time from the rolling window.
    predict() falls back to raw until the window holds >= min_samples."""
    FEATS = {"eagle": ("eagle_p", "depth"),
             "suffix": ("suffix_p", "log1p_count", "log1p_total", "match_len", "depth")}

    def __init__(self, window: int, min_samples: int = 50, refit_k: int = 0,
                 scope: str = "continuous", label: str = "accept_rate",
                 conditional: bool = True):
        import collections
        if label not in ("target_p", "accept_rate"):
            raise RuntimeError(f"unknown online label {label!r}")
        self.method = "multifeat"
        self.multifeat = True
        self.window = int(window)
        self.min_samples = int(min_samples)
        self.refit_k = int(refit_k) if refit_k and int(refit_k) > 0 else 8
        self.scope = scope
        self.label = label
        self.conditional = bool(conditional)
        self.wants_shrunk = False
        self._buf = collections.defaultdict(collections.deque)  # group->deque[(fv,y,clk)]
        self._model = {}   # group->(mean,std,coef,intercept,kind)
        self._dirty: set = set()
        self.ingested = 0

    def featvec(self, group, p, depth, count, total, match_len):
        import numpy as np
        out = []
        for f in self.FEATS[group]:
            if f in ("eagle_p", "suffix_p"):
                out.append(float(p))
            elif f == "depth":
                out.append(float(depth))
            elif f == "match_len":
                if match_len is None:
                    return None
                out.append(float(match_len))
            elif f == "log1p_count":
                if count is None:
                    return None
                out.append(float(np.log1p(count)))
            elif f == "log1p_total":
                if total is None:
                    return None
                out.append(float(np.log1p(total)))
        return np.asarray(out, dtype=np.float64)

    def ingest(self, group, featvec, y, clock):
        if featvec is None:
            return
        buf = self._buf[group]
        buf.append((featvec, float(y), int(clock)))
        self.ingested += 1
        cutoff = int(clock) - self.window
        while buf and buf[0][2] <= cutoff:
            buf.popleft()
        self._dirty.add(group)

    def maybe_refit(self, clock: int) -> None:
        if int(clock) % self.refit_k != 0 or not self._dirty:
            return
        import numpy as np
        for g in list(self._dirty):
            buf = self._buf.get(g)
            if not buf or len(buf) < self.min_samples:
                continue
            X = np.array([b[0] for b in buf]); y = np.array([b[1] for b in buf])
            mean = X.mean(0); std = X.std(0); std[std == 0] = 1.0
            Xs = (X - mean) / std
            try:
                if self.label == "accept_rate":
                    from sklearn.linear_model import LogisticRegression
                    yb = (y >= 0.5).astype(int)
                    if np.unique(yb).size < 2:
                        continue  # single class in window -> keep prior fit
                    m = LogisticRegression(max_iter=500).fit(Xs, yb)
                    self._model[g] = (mean, std, m.coef_[0], float(m.intercept_[0]), "logistic")
                else:
                    from sklearn.linear_model import LinearRegression
                    m = LinearRegression().fit(Xs, y)
                    self._model[g] = (mean, std, m.coef_, float(m.intercept_), "linear")
            except Exception:
                pass
        self._dirty.clear()

    def predict(self, group, p, fallback, depth=0, count=None, total=None,
                match_len=None, **_):
        import numpy as np
        mdl = self._model.get(group)
        buf = self._buf.get(group)
        if mdl is None or buf is None or len(buf) < self.min_samples:
            return fallback
        x = self.featvec(group, p, depth, count, total, match_len)
        if x is None:
            return fallback
        mean, std, coef, intc, kind = mdl
        z = intc + float(np.dot(coef, (x - mean) / std))
        v = 1.0/(1.0+np.exp(-z)) if kind == "logistic" else min(max(z, 0.0), 1.0)
        return float(v) if v > 1e-6 else 1e-6

    def reset_request(self) -> None:
        self._buf.clear(); self._model.clear(); self._dirty.clear()


_ONLINE = None


class _ServingMultiFeatCalibrator:
    """Direction-2 MULTI-FEATURE per-proposer calibrator. Loads a frozen
    fit_chain_hybrid_calib_multifeat map and evaluates the standardized
    linear/logistic form at serving (no sklearn). Per-proposer (NOT fusion):
      eagle  features = [eagle_p, depth]
      suffix features = [suffix_p, log1p(count), log1p(total), match_len, depth]
    predict() returns the raw fallback if a required feature is missing."""

    def __init__(self, blob: dict):
        import numpy as np
        self.kind = blob.get("meta", {}).get("kind", "logistic")
        self.wants_shrunk = False  # multi-feat uses raw suffix_p (no Jeffreys)
        self._g = {}
        for g, m in blob["groups"].items():
            self._g[g] = {
                "features": list(m["features"]),
                "mean": np.asarray(m["mean"], float),
                "std": np.asarray(m["std"], float),
                "coef": np.asarray(m["coef"], float),
                "intercept": float(m["intercept"]),
                "kind": m["kind"]}

    @classmethod
    def load(cls, path: str) -> "_ServingMultiFeatCalibrator":
        with open(path) as f:
            return cls(json.load(f))

    def _vec(self, feats, p, depth, count, total, match_len):
        import numpy as np
        out = []
        for fn in feats:
            if fn in ("eagle_p", "suffix_p"):
                out.append(float(p))
            elif fn == "depth":
                out.append(float(depth))
            elif fn == "match_len":
                if match_len is None:
                    return None
                out.append(float(match_len))
            elif fn == "log1p_count":
                if count is None:
                    return None
                out.append(float(np.log1p(count)))
            elif fn == "log1p_total":
                if total is None:
                    return None
                out.append(float(np.log1p(total)))
            else:
                return None
        return np.asarray(out, float)

    def predict(self, group: str, p: float, fallback: float, depth: int = 0,
                count=None, total=None, match_len=None, **_) -> float:
        import numpy as np
        m = self._g.get(group)
        if m is None:
            return fallback
        x = self._vec(m["features"], p, depth, count, total, match_len)
        if x is None:
            return fallback
        xs = (x - m["mean"]) / m["std"]
        z = m["intercept"] + float(np.dot(m["coef"], xs))
        if m["kind"] == "logistic":
            return float(1.0 / (1.0 + np.exp(-z)))
        return float(min(max(z, 0.0), 1.0))


class _ServingDiscriminator:
    """Joint logistic/beta discriminator: predicts P(pick suffix) from BOTH
    proposers' features at once, replacing the per-proposer suffix_cmp>eagle_cmp
    test. Per-proposer calibration can only see its own score; selection is a
    joint function of both scores plus the trie evidence (count/total/match_len),
    so we fit ONE model on the comparative label. Serving needs no sklearn — just
    standardize, dot, sigmoid.

    JSON blob (written by fit_chain_hybrid_discriminator.py):
      {"kind":"logistic"|"beta", "feature_names":[...],
       "mean":[...], "std":[...], "coef":[...], "intercept":float}
    The feature VECTOR is built by features() below, which MUST match the fitter
    exactly (the fitter imports this same staticmethod)."""

    def __init__(self, blob: dict):
        import numpy as np
        self.kind = blob["kind"]
        self.feature_names = list(blob.get("feature_names", []))
        # depth is an OPTIONAL trailing feature (our-Bayes spec); a map fit without
        # it has no "depth" name -> serving must not append it (keeps the feature
        # vector byte-identical to the fit). Old maps therefore still load.
        self.with_depth = "depth" in self.feature_names
        self.accept_conditioned = bool(blob.get("accept_conditioned", False))
        # GBM boundary arms (Panel-B ceiling / Bayes): a pickled sklearn
        # HistGradientBoostingClassifier on (suffix_p, eagle_p, [depth]); apply
        # predict_proba directly (no standardize/dot/sigmoid).
        self.is_gbm = self.kind.startswith("gbm")
        if self.is_gbm:
            import base64
            import pickle
            self.model = pickle.loads(base64.b64decode(blob["model_b64"]))
            return
        self.mean = np.asarray(blob["mean"], dtype=np.float64)
        self.std = np.asarray(blob["std"], dtype=np.float64)
        self.coef = np.asarray(blob["coef"], dtype=np.float64)
        self.intercept = float(blob["intercept"])

    @classmethod
    def load(cls, path: str) -> "_ServingDiscriminator":
        with open(path) as f:
            return cls(json.load(f))

    @staticmethod
    def features(kind: str, suffix_p, eagle_p, match_len,
                 suffix_count, suffix_total, depth=None) -> list:
        """Build the feature vector from the chosen 5-signal set. logistic uses
        the two probs raw; beta encodes each prob as [ln p, ln(1-p)] (= beta
        calibration generalised to multiple inputs). Non-prob signals
        (match_len, count, total) enter both the same way. depth, when provided
        (our-Bayes spec), is appended LAST so the with/without-depth vectors share
        a prefix and old maps stay loadable."""
        ml = float(match_len or 0.0)
        c = float(suffix_count or 0.0)
        n = float(suffix_total or 0.0)
        sp = float(suffix_p if suffix_p is not None else 0.0)
        ep = float(eagle_p if eagle_p is not None else 0.0)
        if kind == "beta":
            def lo(p):
                p = min(max(p, 1e-4), 1.0 - 1e-4)
                return [math.log(p), math.log(1.0 - p)]
            feats = lo(sp) + lo(ep) + [ml, c, n]
        else:
            feats = [sp, ep, ml, c, n]
        if depth is not None:
            feats = feats + [float(depth)]
        return feats

    def predict(self, suffix_p, eagle_p, match_len,
                suffix_count, suffix_total, depth=None) -> float:
        import numpy as np
        if self.is_gbm:
            sp = float(suffix_p if suffix_p is not None else 0.0)
            ep = float(eagle_p if eagle_p is not None else 0.0)
            x = [sp, ep] + ([float(depth if depth is not None else 0)] if self.with_depth else [])
            return float(self.model.predict_proba([x])[0, 1])
        d = depth if self.with_depth else None
        x = np.asarray(self.features(self.kind, suffix_p, eagle_p, match_len,
                                     suffix_count, suffix_total, depth=d),
                       dtype=np.float64)
        z = (x - self.mean) / self.std
        logit = float(self.coef @ z) + self.intercept
        if logit >= 0:  # numerically stable sigmoid
            return 1.0 / (1.0 + math.exp(-logit))
        e = math.exp(logit)
        return e / (1.0 + e)


_DISC: _ServingDiscriminator | None = None


# ---------------------------------------------------------------------------
# Per-depth decision
# ---------------------------------------------------------------------------

def _decide_and_inject(st: _ChainHybridState, depth: int, topk_p, topk_index):
    """Run the suffix-vs-eagle3 decision for every batch row at this depth.

    Returns (topk_p, topk_index) — clones with injected tokens when at least
    one row chose suffix, the originals otherwise. Cloning matters at depth 0
    where the tensors are spec_info.topk_p / a fresh gather of topk_index;
    in-place writes could leak into state SGLang still owns.
    """
    p_cpu = topk_p.detach().cpu()
    idx_cpu = topk_index.detach().cpu()
    new_p = None
    new_idx = None

    for r, (rid, ctx_tail) in enumerate(st.stash):
        eagle_tok = int(idx_cpu[r, 0])
        eagle_p = float(p_cpu[r, 0])
        chain = st.chains[r]
        rec = {
            "type": "decision",
            "rid": rid,
            "decode_step": st.decode_step.get(rid, 0),
            "depth": depth,
            "eagle_token": eagle_tok,
            "eagle_p": round(eagle_p, 6),
            "eagle_p_cal": None,
            "suffix_token": None,
            "suffix_p": None,
            "suffix_count": None,
            "suffix_total": None,
            "suffix_p_cal": None,
            "match_len": None,
            "suffix_score": None,
            "chosen": "eagle3",
            "agreement": None,
        }

        suffix_tok = None
        suffix_p = None
        suffix_count = None
        suffix_total = None
        if rid in st.active:
            try:
                ctx = (list(ctx_tail) + chain)[-st.cache.max_tree_depth:]
                if chain:
                    # The trie learned only the committed context; insert the
                    # speculative chain so matching can extend through it,
                    # then roll back bit-exactly (requires enable_undo=True).
                    with st.cache.temporary_extension(rid, chain):
                        draft = st.cache.speculate(
                            rid, ctx, max_spec_tokens=1, use_tree_spec=False)
                else:
                    draft = st.cache.speculate(
                        rid, ctx, max_spec_tokens=1, use_tree_spec=False)
                if not draft.is_empty:
                    suffix_tok = int(draft.token_ids[0])
                    suffix_p = float(draft.probs[0]) if draft.probs else 0.0
                    rec["suffix_token"] = suffix_tok
                    rec["suffix_p"] = round(suffix_p, 6)
                    rec["match_len"] = int(draft.match_len)
                    rec["suffix_score"] = round(float(draft.score), 4)
                    # Raw trie counts for the first edge: c = matched count,
                    # n = c / (c/n) recovered from the count-ratio prob. These
                    # let the fitter compute both the raw (c/n) and the
                    # Jeffreys-shrunk ((c+.5)/(n+1)) edge prob offline, and let
                    # the Jeffreys calibration variant shrink at eval here.
                    if draft.counts and suffix_p > 0.0:
                        suffix_count = int(draft.counts[0])
                        suffix_total = int(round(suffix_count / suffix_p))
                        rec["suffix_count"] = suffix_count
                        rec["suffix_total"] = suffix_total
            except Exception as e:
                st.warn_once("speculate", str(e))

        # Eagle distributional confidence (capture-time, env-gated): entropy +
        # top-2 margin of eagle's full next-token distribution, plus the draft's
        # endorsement of the suffix candidate p_eagle(suffix_token). Stashed by
        # the fast_topk wrapper for THIS step (it runs just before this hook).
        if getattr(st, "log_eagle_dist", False):
            ed = getattr(st, "eagle_dist", None)
            if ed is not None and r < ed["entropy"].shape[0]:
                rec["eagle_entropy"] = round(float(ed["entropy"][r]), 6)
                rec["eagle_top2_margin"] = round(float(ed["margin"][r]), 6)
                pr = ed["probs"]
                t2d = getattr(st, "t2d", None)

                def _peagle(tok, _pr=pr, _r=r, _t2d=t2d):
                    di = (int(tok) if _t2d is None
                          else _t2d.get(int(tok)))  # target id -> draft index
                    if di is None or _r >= _pr.shape[0] or di >= _pr.shape[-1]:
                        return 0.0  # token outside the draft's hot vocab
                    return float(_pr[_r, di])

                rec["p_eagle_eagle"] = round(_peagle(eagle_tok), 6)
                if suffix_tok is not None:
                    rec["p_eagle_suffix"] = round(_peagle(suffix_tok), 6)

        # ORACLE mode: the ground-truth token gt[L+depth] decides — eagle if
        # it matches, else suffix (injected) if it matches, else the chain is
        # dead here (proposal-limited selection ceiling).
        if st.mode == "oracle":
            gt_list = st.gt.get(rid)
            L = (st.gt_pos[r] if st.gt_pos is not None
                 and r < len(st.gt_pos) else None)
            gt_tok = None
            if (gt_list is not None and L is not None
                    and L + depth < len(gt_list)):
                gt_tok = int(gt_list[L + depth])
            rec["gt_token"] = gt_tok
            rec["agreement"] = (suffix_tok == eagle_tok
                                if suffix_tok is not None else None)
            chosen_tok = eagle_tok
            hit = "nogt" if gt_tok is None else "none"
            if gt_tok is not None:
                e_hit = eagle_tok == gt_tok
                s_hit = suffix_tok == gt_tok if suffix_tok is not None else False
                if e_hit:
                    hit = "both" if s_hit else "eagle"
                elif s_hit:
                    hit = "suffix"
                    if new_p is None:
                        new_p = topk_p.clone()
                        new_idx = topk_index.clone()
                    new_idx[r, 0] = suffix_tok
                    new_p[r, 0] = 1.0
                    chosen_tok = suffix_tok
                    rec["chosen"] = "suffix"
            rec["oracle_hit"] = hit
            chain.append(chosen_tok)
            st.pending.append(rec)
            continue

        # The two values the decision compares. Raw select-1 compares the raw
        # suffix count-ratio against eagle's raw softmax prob. The calibrated
        # variants map BOTH through their frozen isotonic curves so the
        # comparison is accept-prob vs accept-prob (calibrating only suffix
        # would pit a calibrated accept-prob against an inflated softmax and
        # over-suppress suffix). suffix may be Jeffreys-shrunk first; eagle_p
        # has no counts, so its group is never shrunk.
        suffix_cmp = suffix_p
        eagle_cmp = eagle_p
        if _CALIB is not None:
            eagle_cmp = _CALIB.predict("eagle", eagle_p, eagle_p, depth=depth)
            rec["eagle_p_cal"] = round(eagle_cmp, 6)
            if suffix_p is not None:
                p_in = suffix_p
                if _CALIB.wants_shrunk:
                    if suffix_count is not None and suffix_total:
                        p_in = (suffix_count + 0.5) / (suffix_total + 1)
                    else:
                        st.warn_once(
                            "no-counts",
                            "calib map wants Jeffreys-shrunk probs but the "
                            "suffix draft exposed no counts; using raw suffix_p")
                suffix_cmp = _CALIB.predict("suffix", p_in, suffix_p, depth=depth,
                                            count=suffix_count, total=suffix_total,
                                            match_len=rec.get("match_len"))
                rec["suffix_p_cal"] = round(suffix_cmp, 6)

        chosen_tok = eagle_tok
        if suffix_tok is not None:
            rec["agreement"] = suffix_tok == eagle_tok
            if suffix_tok != eagle_tok:
                if _DISC is not None and suffix_p is not None:
                    # Joint discriminator: P(pick suffix) from BOTH proposers'
                    # features. The chain score stays the raw suffix_p (the
                    # token's own prob, as in the raw arm) — the discriminator
                    # only decides the pick, not the verifier's chain score.
                    pick_p = _DISC.predict(suffix_p, eagle_p,
                                           rec.get("match_len"),
                                           suffix_count, suffix_total,
                                           depth=depth)
                    rec["disc_p"] = round(float(pick_p), 6)
                    take = pick_p > 0.5
                    score_val = suffix_p
                else:
                    take = (suffix_cmp is not None and suffix_cmp > eagle_cmp)
                    score_val = suffix_cmp
                if take:
                    if new_p is None:
                        new_p = topk_p.clone()
                        new_idx = topk_index.clone()
                    new_idx[r, 0] = suffix_tok
                    # Clamp keeps cumulative chain scores monotone non-increasing
                    # (probs are <= 1 by construction; defensive).
                    new_p[r, 0] = min(max(score_val, 0.0), 1.0)
                    chosen_tok = suffix_tok
                    rec["chosen"] = "suffix"

        # PIN: log the GT comparison (oracle_hit/gt_token) for analysis WITHOUT
        # changing the served pick -- gt is precomputed when pin is active, so any
        # arm (raw/calib/disc) gets per-decision selection-accuracy labels.
        if st.pin_active and "oracle_hit" not in rec:
            gt_list = st.gt.get(rid)
            Lp = (st.gt_pos[r] if st.gt_pos is not None and r < len(st.gt_pos)
                  else None)
            gt_tok = (int(gt_list[Lp + depth]) if (gt_list is not None and Lp is not None
                      and Lp + depth < len(gt_list)) else None)
            rec["gt_token"] = gt_tok
            if gt_tok is None:
                rec["oracle_hit"] = "nogt"
            else:
                e_hit = eagle_tok == gt_tok
                s_hit = (suffix_tok == gt_tok) if suffix_tok is not None else False
                rec["oracle_hit"] = ("both" if (e_hit and s_hit) else "eagle" if e_hit
                                     else "suffix" if s_hit else "none")
        chain.append(chosen_tok)
        if st.online is not None:
            # Transient store for the post-verify q_target join (flush() empties
            # st.pending before chain_forward runs).
            st.last_decisions.setdefault(rid, {})[depth] = rec
        st.pending.append(rec)

    return (new_p if new_p is not None else topk_p,
            new_idx if new_idx is not None else topk_index)


def _inject_fallback_run(st: _ChainHybridState, depth: int, topk_p, topk_index):
    """score_fallback mode: occupy depth with the step's suffix run token.

    The run was drafted ONCE per step (in the draft wrapper) from the
    committed context; rows whose run is None (score < threshold, or no
    match) keep eagle's candidate untouched — and depths past the run end
    are eagle continuation tokens drafted conditioned on the suffix prefix.
    """
    runs = st.fallback_runs
    new_p = None
    new_idx = None
    idx_cpu = None

    for r, (rid, _ctx) in enumerate(st.stash):
        run = runs[r] if runs and r < len(runs) else None
        score, probs, match_len = (
            st.fallback_meta[r] if st.fallback_meta and r < len(st.fallback_meta)
            else (None, [], None))
        rec = {
            "type": "decision",
            "policy": "score_fallback",
            "rid": rid,
            "decode_step": st.decode_step.get(rid, 0),
            "depth": depth,
            "eagle_token": None,
            "eagle_p": None,
            "eagle_p_cal": None,
            "suffix_token": None,
            "suffix_p": None,
            "suffix_count": None,
            "suffix_total": None,
            "suffix_p_cal": None,
            "match_len": match_len,
            "suffix_score": round(float(score), 4) if score is not None else None,
            "chosen": "eagle3",
            "agreement": None,
        }
        if run is not None and depth < len(run):
            if new_p is None:
                new_p = topk_p.clone()
                new_idx = topk_index.clone()
            tok = int(run[depth])
            p = float(probs[depth]) if depth < len(probs) else 0.0
            new_idx[r, 0] = tok
            new_p[r, 0] = min(max(p, 0.0), 1.0)
            rec["suffix_token"] = tok
            rec["suffix_p"] = round(p, 6)
            rec["chosen"] = "suffix"
            st.chains[r].append(tok)
        else:
            # eagle's candidate stays; record its token for the log.
            if idx_cpu is None:
                idx_cpu = topk_index.detach().cpu()
            eagle_tok = int(idx_cpu[r, 0])
            rec["eagle_token"] = eagle_tok
            st.chains[r].append(eagle_tok)
        st.pending.append(rec)

    return (new_p if new_p is not None else topk_p,
            new_idx if new_idx is not None else topk_index)


def _install_verify_greedy_oracle() -> None:
    """Oracle teacher-forcing for sglang >=0.5.12: override the verify's
    target_predict (= argmax of the target logits, eagle_info.verify) with the
    recorded GT so the committed token (accepted chain prefix + bonus) follows
    GT EXACTLY. Without this, greedy diverges from the recorded GT on FP
    near-ties; the old force-to-GT (applied after forward_batch_generation) is
    too late to fix the next-draft conditioning (built from verify_output inside
    forward_batch_generation) -> divergence cascades into runaway generation
    (past_gt_end ~1e6) and the draft is conditioned off-GT (eagle==gt collapses).
    Forcing target_predict=GT makes verify NATURALLY commit GT and rebuild the
    extend input from GT, with zero divergence. Linear chain (topk=1):
    target_predict[r, j] = GT[gt_pos[r] + j] (root..step S); -1 past GT -> keep
    the real argmax there (request is at/after EOS and finishes)."""
    import sglang.srt.speculative.eagle_info as eagle_info
    if getattr(eagle_info, "_chain_hybrid_verify_oracle_patched", False):
        return
    original = eagle_info.verify_tree_greedy_func

    def patched(*args, **kwargs):
        st = _STATE
        ov = getattr(st, "gt_predict_override", None) if st is not None else None
        tp = kwargs.get("target_predict")
        if ov is not None and tp is not None and getattr(tp, "dim", None) \
                and tp.dim() == 2:
            try:
                import torch
                ov_t = torch.tensor(ov, dtype=tp.dtype, device=tp.device)
                rows = min(tp.shape[0], ov_t.shape[0])
                cols = min(tp.shape[1], ov_t.shape[1])
                sub = ov_t[:rows, :cols]
                mask = sub >= 0
                tp[:rows, :cols] = torch.where(mask, sub, tp[:rows, :cols])
            except Exception as e:
                if st is not None:
                    st.warn_once("verify-oracle-override", str(e))
        return original(*args, **kwargs)

    eagle_info.verify_tree_greedy_func = patched
    eagle_info._chain_hybrid_verify_oracle_patched = True
    logger.info(
        "chain-hybrid: verify_tree_greedy_func oracle GT-override installed")


def _install_eagle_dist_logging() -> None:
    """Capture-time instrument (env SGLANG_CHAIN_HYBRID_LOG_EAGLE_DIST=1): wrap
    fast_topk so eagle's FULL next-token distribution (the arg to fast_topk,
    before the top-1 reduction) is summarised into entropy + top-2 margin and
    stashed per step. The decision hook then records those + p_eagle(suffix_tok).
    Adds a full-vocab reduction per draft step -> only for the small feature
    capture, never the default path."""
    import importlib
    import torch
    for modname in ("sglang.srt.speculative.eagle_worker",
                    "sglang.srt.speculative.multi_layer_eagle_worker"):
        try:
            m = importlib.import_module(modname)
        except Exception:
            continue
        if not hasattr(m, "fast_topk") \
                or getattr(m, "_chain_hybrid_dist_patched", False):
            continue
        orig = m.fast_topk

        def wrapped(probs, topk, dim=-1, _orig=orig):
            st = _STATE
            try:
                if st is not None and probs is not None and probs.dim() == 2:
                    p = probs.float()
                    ent = -(p * torch.log(p + 1e-12)).sum(dim=-1)
                    t2 = torch.topk(p, 2, dim=-1).values
                    st.eagle_dist = {
                        "entropy": ent.detach().cpu(),
                        "margin": (t2[:, 0] - t2[:, 1]).detach().cpu(),
                        "probs": p.detach()}
            except Exception as e:
                if st is not None:
                    st.warn_once("eagle-dist", str(e))
            return _orig(probs, topk, dim=dim)

        m.fast_topk = wrapped
        m._chain_hybrid_dist_patched = True
    # Build target_id -> draft(hot-vocab) index map so p_eagle(token) can index
    # the draft `probs` (which live in the reduced hot-token space; topk_index is
    # remapped to target ids via hot_token_id at eagle_worker.py:873/929). None
    # hot_token_id => draft uses full target vocab => direct indexing.
    try:
        hid = getattr(_STATE.worker, "hot_token_id", None)
        _STATE.t2d = ({int(t): i for i, t in enumerate(hid.tolist())}
                      if hid is not None else None)
        logger.info("chain-hybrid: eagle-dist t2d map "
                    f"({'None(full-vocab)' if _STATE.t2d is None else len(_STATE.t2d)})")
    except Exception as e:
        _STATE.t2d = None
        _STATE.warn_once("t2d", str(e))
    logger.info("chain-hybrid: eagle-distribution logging installed (fast_topk)")


def _install_select_wrapper() -> None:
    """Rebind select_top_k_tokens with the decision hook in EVERY module that
    binds it by-name.

    draft_forward resolves select_top_k_tokens as a module global at call
    time and invokes it at the top of every draft step with exactly the
    state we need (depth i, topk_p, topk_index) — and crucially AFTER both
    hot_token_id remap sites, so injected suffix tokens (full-vocab ids)
    are never remapped. Same rebind pattern as oracle_patch's
    organize_draft_results tracer.

    sglang imports select_top_k_tokens by-name into BOTH eagle_worker and
    multi_layer_eagle_worker (the MTP path, used on Qwen3.5-27B under
    --disable-overlap-schedule); patching only eagle_worker would make the hook
    a silent no-op on MTP. We rebind all available bindings (eagle_worker,
    multi_layer_eagle_worker, spec_utils) to the same wrapper.
    """
    import importlib

    mods = []
    for modname in ("sglang.srt.speculative.eagle_worker",
                    "sglang.srt.speculative.multi_layer_eagle_worker",
                    "sglang.srt.speculative.spec_utils"):
        try:
            m = importlib.import_module(modname)
        except Exception:
            continue
        if hasattr(m, "select_top_k_tokens"):
            mods.append(m)
    if not mods:
        logger.warning("chain-hybrid: no select_top_k_tokens binding found")
        return
    # Capture the TRUE original from any module not yet patched (all modules
    # import the same underlying function by-name).
    original = None
    for m in mods:
        if not getattr(m, "_chain_hybrid_select_patched", False):
            original = m.select_top_k_tokens
            break
    if original is None:
        return  # every binding already patched

    def chain_hybrid_select(i, topk_p, topk_index, hidden_states, scores, topk):
        st = _STATE
        if (st is None or st.stash is None or topk != 1
                or topk_p is None or topk_p.dim() != 2):
            return original(i, topk_p, topk_index, hidden_states, scores, topk)
        if topk_p.shape[0] != len(st.stash):
            st.warn_once(
                "shape-mismatch",
                f"topk_p rows={topk_p.shape[0]} != stash={len(st.stash)}; "
                f"passing through (decisions skipped this step)")
            return original(i, topk_p, topk_index, hidden_states, scores, topk)
        try:
            if st.mode == "record":
                pass  # pure pass-through baseline (GT capture only)
            elif st.mode == "score_fallback":
                topk_p, topk_index = _inject_fallback_run(
                    st, i, topk_p, topk_index)
            else:  # select1 (raw/calibrated) and oracle
                topk_p, topk_index = _decide_and_inject(
                    st, i, topk_p, topk_index)
        except Exception as e:
            st.warn_once("decide", str(e))
        return original(i, topk_p, topk_index, hidden_states, scores, topk)

    installed = []
    for m in mods:
        if getattr(m, "_chain_hybrid_select_patched", False):
            continue
        m.select_top_k_tokens = chain_hybrid_select
        m._chain_hybrid_select_patched = True
        installed.append(m.__name__.rsplit(".", 1)[-1])
    logger.info(
        f"chain-hybrid: select_top_k_tokens decision hook installed on {installed}")


# ---------------------------------------------------------------------------
# Suffix tail append (route b): extend the chain EagleVerifyInput in place
# ---------------------------------------------------------------------------

def _rebuild_chain_tensors(ndt: int, seq_len: int, device, dtypes: dict):
    """Hand-build the linear-chain verify tensors at size ndt (bs=1).

    Mirrors build_tree_kernel_efficient's FULL_MASK output for topk=1:
      positions  = seq_len + arange(ndt)
      retrive_index = arange(ndt)                row vector (1, ndt)
      retrive_next_token = [1..ndt-1, -1]
      retrive_next_sibling = all -1
      custom_mask rows: row i = [ones(seq_len), causal_{j<=i}(ndt)], flattened
    """
    import torch
    positions = torch.arange(
        seq_len, seq_len + ndt, dtype=dtypes["positions"], device=device)
    retrive_index = torch.arange(
        ndt, dtype=dtypes["retrive"], device=device).view(1, -1)
    nxt = torch.full((1, ndt), -1, dtype=dtypes["retrive"], device=device)
    nxt[0, : ndt - 1] = torch.arange(
        1, ndt, dtype=dtypes["retrive"], device=device)
    sib = torch.full((1, ndt), -1, dtype=dtypes["retrive"], device=device)
    mask = torch.cat(
        [
            torch.ones(ndt, seq_len, dtype=dtypes["mask"], device=device),
            torch.tril(torch.ones(ndt, ndt, dtype=dtypes["mask"],
                                  device=device)),
        ],
        dim=1,
    ).flatten()
    return positions, retrive_index, nxt, sib, mask


def _tail_append(st: _ChainHybridState, spec_info) -> None:
    """Append a suffix continuation run to the chain EagleVerifyInput.

    Runs after original_draft returns and before verify consumes spec_info.
    Mutates spec_info in place; on any guard failure it simply returns,
    leaving the original (correct) verify input untouched.
    """
    import torch

    if st.tail_max <= 0 or st.stash is None:
        return
    if len(st.stash) != 1:
        st.warn_once("tail-bs>1", f"batch size {len(st.stash)} > 1: tail "
                                  f"append only supports bs=1; skipping")
        return
    if getattr(spec_info, "topk", None) != 1:
        return
    rid, ctx_tail = st.stash[0]
    if rid not in st.active:
        return

    ndt_old = int(spec_info.draft_token_num)
    if spec_info.draft_token.numel() != ndt_old:
        st.warn_once("tail-shape", "draft_token numel != draft_token_num; "
                                   "skipping tail")
        return
    # Derive prefix length from the mask size (bs=1: ndt*(seq_len+ndt)).
    mask_numel = spec_info.custom_mask.numel()
    if mask_numel % ndt_old != 0:
        st.warn_once("tail-mask", "unexpected custom_mask size; skipping")
        return
    seq_len = mask_numel // ndt_old - ndt_old
    if seq_len <= 0 or seq_len != int(spec_info.seq_lens_sum):
        st.warn_once(
            "tail-seqlen",
            f"mask-derived seq_len {seq_len} != seq_lens_sum "
            f"{int(spec_info.seq_lens_sum)}; skipping")
        return

    # KV headroom: verify allocates ndt_new slots but the scheduler only
    # reserved server_ndt; skip the tail when the pool is nearly full.
    alloc = getattr(st.worker, "token_to_kv_pool_allocator", None)
    if alloc is not None:
        try:
            if alloc.available_size() < ndt_old + st.tail_max + 64:
                st.warn_once("tail-headroom", "KV pool nearly full; "
                                              "skipping tail this step")
                return
        except Exception:
            pass

    dev = spec_info.draft_token.device
    # sglang 0.5.9 spelled these "retrive_*"; >=0.5.11 fixed to "retrieve_*".
    _ri = "retrieve_index" if hasattr(spec_info, "retrieve_index") else "retrive_index"
    _nx = ("retrieve_next_token" if hasattr(spec_info, "retrieve_next_token")
           else "retrive_next_token")
    _sb = ("retrieve_next_sibling" if hasattr(spec_info, "retrieve_next_sibling")
           else "retrive_next_sibling")
    dtypes = {
        "positions": spec_info.positions.dtype,
        "retrive": getattr(spec_info, _ri).dtype,
        "mask": spec_info.custom_mask.dtype,
    }

    if st.tail_check:
        # Debug gate: our reconstruction at the ORIGINAL size must bit-match
        # the build kernel's output, else the layout assumption is wrong and
        # extending would corrupt verify — skip and warn.
        pos0, ri0, nx0, sb0, mk0 = _rebuild_chain_tensors(
            ndt_old, seq_len, dev, dtypes)
        checks = [
            ("positions", torch.equal(pos0, spec_info.positions)),
            ("retrive_index", torch.equal(ri0, getattr(spec_info, _ri))),
            ("retrive_next_token",
             torch.equal(nx0, getattr(spec_info, _nx))),
            ("retrive_next_sibling",
             torch.equal(sb0, getattr(spec_info, _sb))),
            ("custom_mask", torch.equal(mk0, spec_info.custom_mask)),
        ]
        bad = [name for name, ok in checks if not ok]
        if bad:
            st.warn_once("tail-check",
                         f"chain reconstruction mismatch in {bad}; "
                         f"tail disabled this step")
            return

    # The chain verify will walk — taken from the verify input itself
    # (bit-exact, robust to any dropped per-depth decision).
    chain = spec_info.draft_token[1:].tolist()
    if len(chain) != ndt_old - 1:
        return

    # Suffix continuation run from (context + chain).
    try:
        ctx = (list(ctx_tail) + chain)[-st.cache.max_tree_depth:]
        with st.cache.temporary_extension(rid, chain):
            draft = st.cache.speculate(
                rid, ctx,
                max_spec_tokens=st.tail_max,
                max_spec_factor=st.tail_factor,
                min_token_prob=st.tail_min_prob,
                use_tree_spec=False,
            )
    except Exception as e:
        st.warn_once("tail-speculate", str(e))
        return
    if draft.is_empty:
        return

    tail_tokens = [int(x) for x in draft.token_ids[: st.tail_max]]
    t = len(tail_tokens)
    ndt_new = ndt_old + t

    # Tensor surgery: extend the linear chain to ndt_new.
    positions, ri, nx, sb, mask = _rebuild_chain_tensors(
        ndt_new, seq_len, dev, dtypes)
    spec_info.draft_token = torch.cat(
        [spec_info.draft_token,
         torch.tensor(tail_tokens, dtype=spec_info.draft_token.dtype,
                      device=dev)])
    spec_info.positions = positions
    setattr(spec_info, _ri, ri)
    setattr(spec_info, _nx, nx)
    setattr(spec_info, _sb, sb)
    spec_info.custom_mask = mask
    spec_info.spec_steps = int(spec_info.spec_steps) + t  # sizes accept_index
    spec_info.draft_token_num = ndt_new
    st.last_tail_len = t

    # Decision records for the tail (join rule unchanged).
    decode_step = st.decode_step.get(rid, 0)
    probs = list(draft.probs or [])
    counts = list(draft.counts or [])
    for k, tok in enumerate(tail_tokens):
        st.pending.append({
            "type": "decision",
            "tail": True,
            "rid": rid,
            "decode_step": decode_step,
            "depth": (ndt_old - 1) + k,
            "eagle_token": None,
            "eagle_p": None,
            "eagle_p_cal": None,
            "suffix_token": tok,
            "suffix_p": round(float(probs[k]), 6) if k < len(probs) else None,
            "suffix_count": int(counts[k]) if k < len(counts) else None,
            "suffix_total": None,
            "suffix_p_cal": None,
            "match_len": int(draft.match_len),
            "suffix_score": round(float(draft.score), 4),
            "chosen": "suffix",
            "agreement": None,
        })


# ---------------------------------------------------------------------------
# Draft wrapper: per-request context capture + lazy lifecycle
# ---------------------------------------------------------------------------

def _patch_draft(eagle_worker) -> None:
    original_draft = eagle_worker.draft

    def chain_draft(batch: "ScheduleBatch"):
        st = _STATE
        st.last_tail_len = 0
        try:
            if batch.forward_mode.is_idle():
                st.stash = None
                st.chains = None
                st.fallback_runs = None
                st.fallback_meta = None
            else:
                st.batch_counter += 1
                if st.online is not None:
                    st.last_decisions = {}  # fresh per step (consumed post-verify)
                stash = []
                chains = []
                fb_runs = []
                fb_meta = []
                gt_pos = []
                for req in batch.reqs:
                    rid = req.rid
                    if st.mode == "record":
                        # pass-through baseline: no trie, no decisions —
                        # the forward hook dumps GT at request finish.
                        stash.append((rid, ()))
                        chains.append([])
                        continue
                    if rid not in st.active:
                        try:
                            st.cache.start_request(
                                rid, list(req.origin_input_ids))
                            st.active.add(rid)
                            st.last_out_len[rid] = 0
                            st.decode_step[rid] = 0
                        except Exception as e:
                            st.warn_once("start_request", str(e))
                        if st.mode == "oracle" or st.pin_active:
                            g = (st.gt_map or {}).get(
                                tuple(req.origin_input_ids))
                            st.gt[rid] = list(g) if g is not None else None
                            st.gt_offtrack[rid] = False
                            st.gt_stats[
                                "matched" if g is not None else "unmatched"
                            ] += 1
                            # Emit this rid's prompt ONCE so offline
                            # target-prob capture can bridge rid -> input_ids
                            # -> GT trajectory exactly. The decision log keys
                            # rows by an opaque rid only, and greedy outputs can
                            # collide across DISTINCT prompts (same continuation,
                            # different input_ids), so a content-only bridge is
                            # ambiguous; input_ids is the exact key (matches the
                            # gt_map lookup above and gt_tokens.jsonl).
                            st.pending.append({
                                "type": "req", "rid": rid,
                                "input_ids": [int(x)
                                              for x in req.origin_input_ids],
                            })
                            if g is None:
                                st.warn_once(
                                    "gt-unmatched",
                                    "request prompt not found in GT dump; "
                                    "running pure eagle for unmatched "
                                    "requests (count in gt_stats)")
                    st.decode_step[rid] = st.decode_step.get(rid, 0) + 1
                    st.last_seen[rid] = st.batch_counter
                    ctx = (list(req.origin_input_ids)
                           + list(req.output_ids))[-st.cache.max_tree_depth:]
                    stash.append((rid, ctx))
                    chains.append([])
                    gt_pos.append(len(req.output_ids or []))
                    # score_fallback: ONE suffix run per step from the
                    # committed context (sim hybrid_e3 semantics: suffix iff
                    # draft.score >= threshold, else pure eagle this step).
                    if st.mode == "score_fallback":
                        run = None
                        meta = (None, [], None)
                        if rid in st.active:
                            try:
                                d = st.cache.speculate(
                                    rid, ctx,
                                    max_spec_tokens=(
                                        st.worker.speculative_num_steps),
                                    max_spec_factor=st.fb_factor,
                                    min_token_prob=st.fb_min_prob,
                                    use_tree_spec=False)
                                if not d.is_empty:
                                    meta = (float(d.score),
                                            [float(p) for p in d.probs],
                                            int(d.match_len))
                                    if d.score >= st.score_threshold:
                                        run = [int(t) for t in d.token_ids]
                            except Exception as e:
                                st.warn_once("fallback-speculate", str(e))
                        fb_runs.append(run)
                        fb_meta.append(meta)
                if len(stash) > 1:
                    st.warn_once(
                        "bs>1",
                        f"batch size {len(stash)} > 1: decision path is "
                        f"written batch-safe but only validated at bs=1")
                st.stash = stash
                st.chains = chains
                st.fallback_runs = fb_runs if st.mode == "score_fallback" else None
                st.fallback_meta = fb_meta if st.mode == "score_fallback" else None
                st.gt_pos = (gt_pos if (st.mode == "oracle" or st.pin_active)
                             else None)
                # Oracle/pin: precompute each row's GT continuation so the verify
                # hook can override target_predict (= argmax) with GT, forcing
                # the committed token (incl. the bonus) onto the GT trajectory.
                # S+1 entries (root..last step); -1 past GT end (chain ends).
                if st.mode == "oracle" or st.pin_active:
                    # head chain (S+1) + optional suffix tail (tail_max): the
                    # verify is one linear sequence [root,c1..cS,t1..tT], so
                    # verify token k -> output position gt_pos[r]+k for the whole
                    # length; override target_predict over head AND tail.
                    s1 = int(eagle_worker.speculative_num_steps) + 1 \
                        + int(st.tail_max)
                    ov = []
                    for r, (rid, _ctx) in enumerate(stash):
                        g = st.gt.get(rid)
                        L = gt_pos[r] if r < len(gt_pos) else None
                        if g is not None and L is not None:
                            ov.append([int(g[L + j]) if (L + j) < len(g) else -1
                                       for j in range(s1)])
                        else:
                            ov.append([-1] * s1)
                    st.gt_predict_override = ov
                else:
                    st.gt_predict_override = None
        except Exception as e:
            st.stash = None
            st.chains = None
            st.fallback_runs = None
            st.fallback_meta = None
            st.warn_once("draft-stash", str(e))
        try:
            result = original_draft(batch)
            if st.tail_max > 0 and st.stash is not None \
                    and st.mode != "record":
                try:
                    _tail_append(st, result)
                except Exception as e:
                    st.warn_once("tail", str(e))
            if st.online is not None:
                # draft_token_num gives the per-request stride into the verify
                # logits (req_offset = i*num_draft) for the post-verify q join.
                st.last_num_draft = getattr(result, "draft_token_num", None)
            return result
        finally:
            st.stash = None
            st.chains = None
            st.fallback_runs = None
            st.fallback_meta = None
            st.flush()

    eagle_worker.draft = chain_draft


def _load_gt_map(path: str) -> dict:
    """Load a gt_tokens.jsonl ({input_ids, output_ids} per line) into a
    {tuple(input_ids): output_ids} map. Shared by oracle mode (selection
    ceiling) and pin mode (force every arm's committed tokens onto this
    standalone trajectory)."""
    gt_map: dict = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            gt_map[tuple(rec["input_ids"])] = rec["output_ids"]
    return gt_map


def _dump_gt(st: _ChainHybridState, req) -> None:
    """record mode: append the finished request's (input_ids, output_ids) to
    the GT dump — the oracle arm matches requests by exact input_ids."""
    if getattr(st.worker, "tp_rank", 0) != 0 or not st.gt_out_path:
        return
    try:
        with open(st.gt_out_path, "a") as f:
            f.write(json.dumps({
                "input_ids": [int(x) for x in req.origin_input_ids],
                "output_ids": [int(x) for x in (req.output_ids or [])],
            }) + "\n")
    except OSError as e:
        st.warn_once("gt-dump", str(e))


# ---------------------------------------------------------------------------
# Forward wrapper: incremental trie updates + per-step accept records
# ---------------------------------------------------------------------------

def _online_ingest(st: "_ChainHybridState", eagle_worker, i: int, rid: str,
                   accept_lens: list) -> None:
    """Post-verify: read q_target for each depth's eagle/suffix token from the
    stashed verify logits and feed (raw_prob, q_target) into the online window.

    The chain verify input is [root, c_1, ..., c_S(, tail)]; the verify-logit row
    at chain position k predicts the token at position k+1. A depth-d decision's
    token sits at position d+1, so its q_target = softmax(verify_logits[req_offset
    + d])[token_id] — read for BOTH eagle and suffix off the SAME row (loser not
    censored), conditioned on the realized chain prefix. Tail rows are skipped
    (their suffix_p is a cumulative path prob, not a single-edge prob)."""
    import torch.nn.functional as F
    vl = getattr(eagle_worker, "_oracle_stashed_verify_logits", None)
    if vl is None:
        st.warn_once("online-no-logits",
                     "verify logits not stashed; online ingest skipped")
        return
    decs = st.last_decisions.get(rid)
    if not decs:
        return
    nd = int(st.last_num_draft or vl.shape[0])
    req_offset = i * nd
    clock = (st.batch_counter if st.online.scope == "continuous"
             else st.decode_step.get(rid, 0))
    alen = int(accept_lens[i]) if i < len(accept_lens) else None
    V = vl.shape[-1]
    for depth, rec in decs.items():
        if rec.get("tail"):
            continue
        # conditional (accept-conditioned): ingest a depth only if the realized
        # chain prefix was accepted through it (accept_len >= depth) -> the
        # per-step CONDITIONAL accept population, not the dead-chain pool.
        if (st.online.conditional and alen is not None
                and int(depth) > alen):
            continue
        row = req_offset + int(depth)
        if row < 0 or row >= vl.shape[0]:
            continue
        probs = F.softmax(vl[row].float(), dim=-1)
        # target's committed token at this row (= argmax | realized prefix).
        # accept_rate label = (drafted token == committed); logged for both
        # objectives so the offline reliability plot can recover the accept
        # event for BOTH proposers (loser included), not just the chosen one.
        committed = int(probs.argmax())
        ep = rec.get("eagle_p")
        et = rec.get("eagle_token")
        q_eagle = None
        mf = getattr(st.online, "multifeat", False)
        if ep is not None and et is not None and 0 <= int(et) < V:
            q_eagle = float(probs[int(et)])  # target prob (diag + target_p label)
            y = (1.0 if int(et) == committed else 0.0) \
                if st.online.label == "accept_rate" else q_eagle
            if mf:
                st.online.ingest(
                    "eagle", st.online.featvec("eagle", ep, int(depth), None, None, None),
                    y, clock)
            else:
                st.online.ingest("eagle", int(depth), ep, y, clock)
        sp = rec.get("suffix_p")
        stk = rec.get("suffix_token")
        q_suffix = None
        if sp is not None and stk is not None and 0 <= int(stk) < V:
            q_suffix = float(probs[int(stk)])
            y = (1.0 if int(stk) == committed else 0.0) \
                if st.online.label == "accept_rate" else q_suffix
            if mf:
                st.online.ingest(
                    "suffix", st.online.featvec(
                        "suffix", sp, int(depth), rec.get("suffix_count"),
                        rec.get("suffix_total"), rec.get("match_len")),
                    y, clock)
            else:
                st.online.ingest("suffix", int(depth), sp, y, clock)
        if st.online_pairs_path is not None:
            st.online_pending.append({
                "type": "online_pair", "rid": rid,
                "decode_step": rec.get("decode_step"), "depth": int(depth),
                "eagle_p": ep,
                "q_eagle": round(q_eagle, 6) if q_eagle is not None else None,
                "eagle_token": et,
                "suffix_p": sp,
                "q_suffix": round(q_suffix, 6) if q_suffix is not None else None,
                "suffix_token": stk,
                "committed_token": committed,
                "eagle_p_online": rec.get("eagle_p_cal"),
                "suffix_p_online": rec.get("suffix_p_cal"),
                "chosen": rec.get("chosen"), "accept_len": alen,
            })
    st.last_decisions.pop(rid, None)


def _patch_forward(eagle_worker) -> None:
    original_forward = eagle_worker.forward_batch_generation

    def chain_forward(batch: "ScheduleBatch"):
        result = original_forward(batch)
        st = _STATE
        try:
            is_decode = not (
                batch.forward_mode.is_extend()
                or getattr(batch, "is_extend_in_batch", False)
            )
            if not is_decode:
                return result

            # sglang 0.5.12 renamed accept_length_per_req_cpu ->
            # num_correct_drafts_per_req_cpu (GenerationBatchResult). Try both.
            accept_lens = list(
                getattr(result, "accept_length_per_req_cpu", None)
                or getattr(result, "num_correct_drafts_per_req_cpu", None)
                or [])

            for i, req in enumerate(batch.reqs):
                rid = req.rid
                if st.mode == "record":
                    if req.finished():
                        _dump_gt(st, req)
                    continue
                if rid not in st.active:
                    continue

                # Incremental trie update (ArcticInference official
                # semantics): feed exactly the tokens verify committed this
                # step, tracked via output_ids length delta.
                out = req.output_ids or []
                prev = st.last_out_len.get(rid, 0)
                if len(out) > prev:
                    # Oracle GT alignment: greedy FP flips would otherwise
                    # push the realized output off the recorded path and
                    # invalidate gt indexing for the rest of the turn. Force
                    # divergent committed tokens back to GT (same mechanism
                    # as oracle_patch's REPLAY verify-override: the token
                    # enters the next forward via its embedding, so the
                    # context stays self-consistent). The next-step chain
                    # seed (spec_info.verified_id) is mirrored below.
                    if st.mode == "oracle" or st.pin_active:
                        g = st.gt.get(rid)
                        if g is not None:
                            forced = False
                            for k in range(prev, min(len(out), len(g))):
                                if int(out[k]) != int(g[k]):
                                    req.output_ids[k] = int(g[k])
                                    st.gt_stats["forced"] = (
                                        st.gt_stats.get("forced", 0) + 1)
                                    forced = True
                            if len(out) > len(g):
                                st.gt_stats["past_gt_end"] = (
                                    st.gt_stats.get("past_gt_end", 0)
                                    + len(out) - len(g))
                            if forced:
                                try:
                                    di = getattr(batch, "spec_info", None)
                                    vid = getattr(di, "verified_id", None)
                                    if vid is not None \
                                            and vid.numel() == len(batch.reqs):
                                        vid[i] = int(req.output_ids[-1])
                                except Exception as e:
                                    st.warn_once("gt-force-vid", str(e))
                            out = req.output_ids
                    try:
                        st.cache.add_active_response(rid, list(out[prev:]))
                    except Exception as e:
                        st.warn_once("add_active_response", str(e))
                    st.last_out_len[rid] = len(out)

                if i < len(accept_lens):
                    st.pending.append({
                        "type": "step",
                        "rid": rid,
                        "decode_step": st.decode_step.get(rid, 0),
                        "accept_len": int(accept_lens[i]),
                    })

                # Online calibration: post-verify q_target join + window ingest.
                if st.online is not None and rid in st.last_decisions:
                    try:
                        _online_ingest(st, eagle_worker, i, rid, accept_lens)
                    except Exception as e:
                        st.warn_once("online-ingest", str(e))

                if req.finished():
                    try:
                        st.cache.stop_request(rid)
                    except Exception as e:
                        st.warn_once("stop_request", str(e))
                    st.active.discard(rid)
                    st.last_out_len.pop(rid, None)
                    st.decode_step.pop(rid, None)
                    st.last_seen.pop(rid, None)
                    st.gt.pop(rid, None)
                    st.gt_offtrack.pop(rid, None)
                    st.last_decisions.pop(rid, None)
                    if st.online is not None and st.online.scope == "per_request":
                        st.online.reset_request()

            if st.online is not None:
                st.online.maybe_refit(st.batch_counter)
                st.flush_online()
            if st.batch_counter % GC_INTERVAL == 0:
                _gc_stale(st)
                if st.mode == "oracle":
                    logger.info(f"chain-hybrid oracle gt_stats: {st.gt_stats}")
            st.flush()
        except Exception as e:
            st.warn_once("forward-hook", str(e))
        return result

    eagle_worker.forward_batch_generation = chain_forward


def _gc_stale(st: _ChainHybridState) -> None:
    cutoff = st.batch_counter - GC_MAX_IDLE
    stale = [rid for rid, seen in st.last_seen.items() if seen < cutoff]
    for rid in stale:
        try:
            st.cache.stop_request(rid)
        except Exception:
            pass
        st.active.discard(rid)
        st.last_out_len.pop(rid, None)
        st.decode_step.pop(rid, None)
        st.last_seen.pop(rid, None)
    if stale:
        logger.info(f"chain-hybrid: GC'd {len(stale)} stale request(s)")


# ---------------------------------------------------------------------------
# Tail support wrappers: per-step sglang-internal constant fixes
# ---------------------------------------------------------------------------

def _resolve_target_attn_backend(eagle_worker):
    """The target attention backend that owns ``num_draft_tokens``. For
    Mamba-hybrid models the model-runner backend is a HybridLinearAttnBackend
    wrapper; the full-attention sub-backend carries the constant."""
    backend = eagle_worker.target_worker.model_runner.attn_backend
    if not hasattr(backend, "num_draft_tokens") \
            and hasattr(backend, "full_attn_backend"):
        backend = backend.full_attn_backend
    return backend


def _patch_verify_for_tail(eagle_worker) -> None:
    """Wrap eagle_worker.verify to override the triton target attention
    backend's worker-level ``num_draft_tokens`` constant for steps whose
    verify input was tail-extended (it sizes qo_indptr / mask_indptr /
    max_extend_len in TARGET_VERIFY and never reads spec_info)."""
    original_verify = eagle_worker.verify
    backend = _resolve_target_attn_backend(eagle_worker)
    server_ndt = eagle_worker.server_args.speculative_num_draft_tokens

    def tail_verify(batch, *args, **kwargs):
        # sglang 0.5.9 called verify(batch, spec_info); 0.5.12 calls verify(batch)
        # with the verify input carried on the batch. Resolve spec_info either
        # way (falls back to no override — safe — if it can't be found).
        spec_info = args[0] if args else getattr(batch, "spec_info", None)
        ndt = getattr(spec_info, "draft_token_num", server_ndt)
        override = ndt != server_ndt
        if override:
            backend.num_draft_tokens = ndt
        try:
            return original_verify(batch, *args, **kwargs)
        finally:
            if override:
                backend.num_draft_tokens = server_ndt

    eagle_worker.verify = tail_verify


def _patch_draft_extend_for_tail(eagle_worker) -> None:
    """Wrap forward_draft_extend_after_decode to bump speculative_num_steps
    by the tail length for the duration of the call. The draft-extend triton
    kernel's position/verified_id stores are masked by a
    next_power_of_2(num_steps+1) constexpr — accept lengths beyond it are
    SILENTLY truncated, so the bump is required for correctness whenever
    accept_len can exceed S (t >= next_pow2(S+1) - S - 1)."""
    original_fdead = eagle_worker.forward_draft_extend_after_decode

    def tail_fdead(batch, *args, **kwargs):
        st = _STATE
        bump = st.last_tail_len if st is not None else 0
        if st is not None:
            st.last_tail_len = 0
        if bump <= 0:
            return original_fdead(batch, *args, **kwargs)
        saved = eagle_worker.speculative_num_steps
        eagle_worker.speculative_num_steps = saved + bump
        try:
            return original_fdead(batch, *args, **kwargs)
        finally:
            eagle_worker.speculative_num_steps = saved

    eagle_worker.forward_draft_extend_after_decode = tail_fdead


# ---------------------------------------------------------------------------
# Entry point (called from oracle_patch.patch_eagle_worker_full)
# ---------------------------------------------------------------------------

def _install_online_verify_stash(eagle_worker) -> None:
    """Minimal stash of the target verify logits for the online calibrator's
    post-verify q_target join. Mirrors ONLY the logit-stash branch of
    oracle_patch._patch_verify_logits (no timing, no token-replay override),
    because for chain-hybrid arms patch_eagle_worker_full's LATENCY_ONLY path
    returns before the oracle stash is installed. The target verify forward
    predicts, at each chain position k, the token following position k; so
    chain_forward reads softmax(verify_logits[req_offset + d]) for the depth-d
    eagle/suffix tokens (both against the same realized prefix)."""
    if getattr(eagle_worker, "_chain_hybrid_verify_stash_patched", False):
        return
    tw = eagle_worker.target_worker
    original_target_forward = tw.forward_batch_generation

    def patched_target_forward(*args, **kwargs):
        result = original_target_forward(*args, **kwargs)
        try:
            if kwargs.get("is_verify", False) and result.logits_output is not None:
                logits = result.logits_output.next_token_logits
                if logits is not None and logits.numel() > 0:
                    eagle_worker._oracle_stashed_verify_logits = logits.detach().cpu()
                else:
                    eagle_worker._oracle_stashed_verify_logits = None
        except Exception:
            eagle_worker._oracle_stashed_verify_logits = None
        return result

    tw.forward_batch_generation = patched_target_forward
    eagle_worker._chain_hybrid_verify_stash_patched = True
    logger.info("chain-hybrid: online verify-logit stash installed")


def patch_chain_hybrid(eagle_worker: "EAGLEWorker") -> None:
    global _STATE

    if os.environ.get("SGLANG_LATENCY_ONLY", "0") != "1":
        raise RuntimeError(
            "SGLANG_CHAIN_HYBRID=1 requires SGLANG_LATENCY_ONLY=1: oracle "
            "vanilla mode force-accepts 0 tokens per step, which would "
            "nullify the chain-hybrid experiment.")
    if getattr(eagle_worker, "topk", None) != 1:
        raise RuntimeError(
            f"SGLANG_CHAIN_HYBRID=1 requires chain mode "
            f"(--speculative-eagle-topk 1), got topk={eagle_worker.topk}.")
    if not getattr(eagle_worker.server_args, "disable_cuda_graph", False):
        raise RuntimeError(
            "SGLANG_CHAIN_HYBRID=1 requires --disable-cuda-graph: the CUDA "
            "graph draft path bypasses draft_forward, so the decision hook "
            "would silently never run.")

    # Lazy import: needs arctic_inference (container-only) and the repo root
    # on sys.path (server is launched with cwd=/workspace, same mechanism
    # that resolves simulation.oracle imports).
    from hybrid_spec_decoding.suffix_decoding.suffix_tree import (
        SuffixDecodingCache,
    )

    # ArcticInference official defaults (max_spec_factor=1.0,
    # min_token_prob=0.1); enable_undo for temporary_extension;
    # max_spec_tokens=1 — only the first suffix token is ever used.
    suffix_cache = SuffixDecodingCache(
        max_tree_depth=64,
        max_cached_requests=100000,
        max_spec_tokens=1,
        max_spec_factor=1.0,
        max_spec_offset=0.0,
        min_token_prob=0.1,
        use_tree_spec=False,
        enable_undo=True,
    )

    log_path = os.environ.get(
        "SGLANG_CHAIN_HYBRID_LOG", "/tmp/sglang_chain_hybrid_decisions.jsonl")

    # Suffix tail append (route b) config — validated up front so a
    # misconfigured run fails at boot instead of silently mis-masking.
    tail_max = int(os.environ.get("SGLANG_CHAIN_HYBRID_TAIL", "0"))
    tail_factor = float(os.environ.get("SGLANG_CHAIN_HYBRID_TAIL_FACTOR", "4.0"))
    tail_min_prob = float(
        os.environ.get("SGLANG_CHAIN_HYBRID_TAIL_MIN_PROB", "0.1"))
    tail_check = os.environ.get("SGLANG_CHAIN_HYBRID_TAIL_CHECK", "0") == "1"
    if tail_max > 0:
        if getattr(eagle_worker.server_args, "max_running_requests", None) != 1:
            raise RuntimeError(
                "SGLANG_CHAIN_HYBRID_TAIL requires --max-running-requests 1 "
                "(tail tensor surgery only supports bs=1).")
        backend = _resolve_target_attn_backend(eagle_worker)
        if not hasattr(backend, "num_draft_tokens"):
            raise RuntimeError(
                f"SGLANG_CHAIN_HYBRID_TAIL requires a target attention "
                f"backend exposing num_draft_tokens (triton); got "
                f"{type(backend).__name__}.")
        if hasattr(backend, "linear_attn_backend") or hasattr(
                eagle_worker.target_worker.model_runner.attn_backend,
                "linear_attn_backend"):
            # Mamba-hybrid archs (Qwen3.5/qwen3-next): the speculative mamba
            # intermediate caches (intermediate_ssm / intermediate_conv_window
            # in memory_pool.py) are statically allocated with the SERVER's
            # speculative_num_draft_tokens. A tail-extended verify writes up to
            # (num_steps + 1 + tail_max) intermediate states, so that pool MUST
            # be oversized to >= that (launch with a larger
            # --speculative-num-draft-tokens AND bypass server_args' topk==1
            # force-reset) or it overflows -> silent corruption.
            server_ndt = int(getattr(
                eagle_worker.server_args, "speculative_num_draft_tokens", 0) or 0)
            need = int(eagle_worker.speculative_num_steps) + 1 + tail_max
            if server_ndt < need:
                raise RuntimeError(
                    "SGLANG_CHAIN_HYBRID_TAIL on Mamba-hybrid needs the mamba "
                    f"spec cache oversized: server num_draft_tokens={server_ndt} "
                    f"< num_steps+1+tail_max={need}. Re-launch with "
                    f"--speculative-num-draft-tokens {need} (and the server_args "
                    "topk==1 reset bypass: SGLANG_CHAIN_HYBRID_TAIL>0).")
            logger.warning(
                "chain-hybrid TAIL on Mamba-hybrid: relying on OVERSIZED mamba "
                f"spec cache (num_draft_tokens={server_ndt} >= {need}). "
                "EXPERIMENTAL — sanity-check accept lengths.")

    # Decision mode (default per-depth select-1). "score_fallback" mirrors
    # the simulator's hybrid_e3:t baseline; "record" dumps GT trajectories;
    # "oracle" is the per-depth selection ceiling driven by a GT dump.
    mode = os.environ.get("SGLANG_CHAIN_HYBRID_MODE", "select1")
    if mode not in ("select1", "score_fallback", "record", "oracle"):
        raise RuntimeError(f"unknown SGLANG_CHAIN_HYBRID_MODE={mode!r}")
    score_threshold = float(
        os.environ.get("SGLANG_CHAIN_HYBRID_SCORE_THRESHOLD", "5.0"))
    fb_factor = float(os.environ.get("SGLANG_CHAIN_HYBRID_FB_FACTOR", "1.0"))
    fb_min_prob = float(
        os.environ.get("SGLANG_CHAIN_HYBRID_FB_MIN_PROB", "0.1"))

    _STATE = _ChainHybridState(
        eagle_worker, suffix_cache, log_path,
        tail_max=tail_max, tail_factor=tail_factor,
        tail_min_prob=tail_min_prob, tail_check=tail_check,
        mode=mode, score_threshold=score_threshold,
        fb_factor=fb_factor, fb_min_prob=fb_min_prob)

    # Optional TRAJECTORY PIN. SGLANG_CHAIN_HYBRID_PIN points at a standalone
    # gt_tokens.jsonl; when set on a select1/score_fallback arm, the committed
    # tokens are forced onto that trajectory (same GT machinery oracle uses) so
    # every arm follows the identical path despite greedy FP-tie flips, while
    # each arm still builds its own draft chain. Ignored in oracle mode (which
    # pins via SGLANG_CHAIN_HYBRID_GT) and rejected in record mode.
    pin_path = os.environ.get("SGLANG_CHAIN_HYBRID_PIN")
    if pin_path and mode == "record":
        raise RuntimeError("SGLANG_CHAIN_HYBRID_PIN is incompatible with "
                           "SGLANG_CHAIN_HYBRID_MODE=record")
    _STATE.pin_active = bool(pin_path) and mode in ("select1", "score_fallback")
    if _STATE.pin_active:
        _STATE.gt_map = _load_gt_map(pin_path)

    # Optional suffix-prob calibration. SGLANG_CHAIN_HYBRID_CALIB points at a
    # frozen isotonic map; absent -> raw count-ratio comparison (the original
    # select-1). The map's meta.shrink decides whether suffix probs are
    # Jeffreys-shrunk before lookup (must match how the map was fitted).
    global _CALIB, _ONLINE
    calib_path = os.environ.get("SGLANG_CHAIN_HYBRID_CALIB")
    online_method = os.environ.get("SGLANG_CHAIN_HYBRID_ONLINE_CALIB")
    if online_method:
        # ONLINE sliding-window calibration to target_p. No frozen JSON; the
        # per-(group,depth) map is learned at serving time from a window of the
        # preceding prefix. Overrides any frozen SGLANG_CHAIN_HYBRID_CALIB.
        window = int(os.environ.get("SGLANG_CHAIN_HYBRID_ONLINE_WINDOW", "256"))
        min_samples = int(
            os.environ.get("SGLANG_CHAIN_HYBRID_ONLINE_MIN_SAMPLES", "50"))
        refit_k = int(os.environ.get("SGLANG_CHAIN_HYBRID_ONLINE_REFIT_K", "0"))
        scope = os.environ.get("SGLANG_CHAIN_HYBRID_ONLINE_SCOPE", "continuous")
        if scope not in ("continuous", "per_request"):
            raise RuntimeError(f"unknown ONLINE_SCOPE={scope!r}")
        label = os.environ.get("SGLANG_CHAIN_HYBRID_ONLINE_LABEL", "target_p")
        conditional = os.environ.get(
            "SGLANG_CHAIN_HYBRID_ONLINE_CONDITIONAL", "0") == "1"
        if online_method == "multifeat":
            # COMBO: windowed MULTI-FEATURE calibrator (window+multifeat+conditional)
            _ONLINE = _OnlineMultiFeatCalibrator(
                window, min_samples=min_samples, refit_k=refit_k, scope=scope,
                label=label, conditional=conditional)
        else:
            _ONLINE = _OnlineWindowCalibrator(
                online_method, window, min_samples=min_samples,
                refit_k=refit_k, scope=scope, label=label, conditional=conditional)
        _CALIB = _ONLINE
        _STATE.online = _ONLINE
        _STATE.online_pairs_path = os.environ.get(
            "SGLANG_CHAIN_HYBRID_ONLINE_PAIRS",
            "/tmp/sglang_chain_hybrid_online_pairs.jsonl")
        # The verify-logit stash is OFF on the LATENCY_ONLY chain-hybrid path;
        # install it here so the post-verify q_target join has the logits.
        _install_online_verify_stash(eagle_worker)
        calib_desc = (
            f"ONLINE {online_method} (window={window} steps, scope={scope}, "
            f"min_samples={min_samples}, refit_k={_ONLINE.refit_k}, "
            f"label={label}{'+conditional' if conditional else ''}; "
            f"pairs -> {_STATE.online_pairs_path})")
    elif os.environ.get("SGLANG_CHAIN_HYBRID_MULTIFEAT"):
        # Direction-2 multi-feature per-proposer calibrator (suffix uses
        # prob+count+total+match_len, eagle uses prob; depth as a feature).
        _ONLINE = None
        mf_path = os.environ["SGLANG_CHAIN_HYBRID_MULTIFEAT"]
        _CALIB = _ServingMultiFeatCalibrator.load(mf_path)
        calib_desc = f"MULTI-FEAT calib (kind={_CALIB.kind}, map={mf_path})"
    elif calib_path:
        _ONLINE = None
        _CALIB = _ServingIsoCalibrator.load(calib_path)
        calib_desc = (f"calibrated (map={calib_path}, "
                      f"shrink={'jeffreys' if _CALIB.wants_shrunk else 'none'})")
    else:
        _ONLINE = None
        _CALIB = None
        calib_desc = "raw suffix_p > eagle_p (uncalibrated)"

    # Optional joint discriminator (overrides the suffix_cmp>eagle_cmp test).
    # SGLANG_CHAIN_HYBRID_DISC points at a frozen logistic/beta model fitted on
    # the comparative label with both proposers' features; absent -> calib/raw.
    global _DISC
    disc_path = os.environ.get("SGLANG_CHAIN_HYBRID_DISC")
    if disc_path:
        _DISC = _ServingDiscriminator.load(disc_path)
        calib_desc = (f"discriminator (kind={_DISC.kind}, "
                      f"P(pick suffix)>0.5, depth={'yes' if _DISC.with_depth else 'no'}, "
                      f"accept_cond={'yes' if _DISC.accept_conditioned else 'no'}, "
                      f"map={disc_path})")
    else:
        _DISC = None

    _install_select_wrapper()
    _STATE.log_eagle_dist = (
        os.environ.get("SGLANG_CHAIN_HYBRID_LOG_EAGLE_DIST", "0") == "1")
    if _STATE.log_eagle_dist:
        _install_eagle_dist_logging()
    if mode == "oracle" or _STATE.pin_active:
        _install_verify_greedy_oracle()
    _patch_draft(eagle_worker)
    _patch_forward(eagle_worker)
    if tail_max > 0:
        _patch_verify_for_tail(eagle_worker)
        _patch_draft_extend_for_tail(eagle_worker)
        tail_desc = (f"tail=on (T_max={tail_max}, factor={tail_factor}, "
                     f"min_p={tail_min_prob}, check={tail_check})")
    else:
        tail_desc = "tail=off"

    if mode == "record":
        _STATE.gt_out_path = os.environ.get(
            "SGLANG_CHAIN_HYBRID_GT_OUT", "/tmp/sglang_chain_hybrid_gt.jsonl")
        mode_desc = f"mode=record (GT dump -> {_STATE.gt_out_path})"
    elif mode == "oracle":
        gt_path = os.environ.get("SGLANG_CHAIN_HYBRID_GT")
        if not gt_path:
            raise RuntimeError(
                "SGLANG_CHAIN_HYBRID_MODE=oracle requires "
                "SGLANG_CHAIN_HYBRID_GT=<gt_tokens.jsonl from a record arm>")
        _STATE.gt_map = _load_gt_map(gt_path)
        mode_desc = (f"mode=oracle (per-depth selection ceiling, "
                     f"{len(_STATE.gt_map)} GT trajectories from {gt_path})")
    elif mode == "score_fallback":
        mode_desc = (f"mode=score_fallback (suffix iff score>="
                     f"{score_threshold}, F={fb_factor}, T={fb_min_prob})")
    else:
        mode_desc = f"mode=select1, decision={calib_desc}"
    if _STATE.pin_active:
        mode_desc += (f" [PINNED to {len(_STATE.gt_map)} standalone "
                      f"trajectories from {pin_path}]")
    logger.info(
        f"Chain-hybrid patch applied: {mode_desc}, {tail_desc}, "
        f"steps={eagle_worker.speculative_num_steps}, "
        f"decision log -> {log_path}")


# ===========================================================================
# DFlash chain-hybrid: sglang-native per-position select-1 on the draft block
# ===========================================================================
#
# DFlash (z-lab block/diffusion drafter) drafts a whole block of block_size
# tokens in ONE non-causal forward inside
# DFlashWorker._prepare_for_speculative_decoding, so there is NO per-depth
# EAGLE chain loop and NO select_top_k_tokens hook. The per-position select-1
# is instead applied by substituting tokens directly into the verify input's
# draft_token block AFTER _prepare builds it and BEFORE the target verify
# forward consumes it: the DFlash verify mask is positional / token-identity
# based, and prepare_for_verify sets batch.input_ids = verify_input.draft_token
# (same storage), so an IN-PLACE write to draft_token is seen by both the
# target forward and verify with zero re-prepare.
#
# Greedy DFlash AUTO-PINS the trajectory. verify() (dflash_info.verify ->
# compute_dflash_correct_drafts_and_bonus) commits exactly the longest block
# prefix that matches the target's greedy argmax, plus the target's argmax
# bonus at the first mismatch. So the committed sequence is the target-greedy
# continuation REGARDLESS of which tokens we substitute into the block —
# substitution changes only the ACCEPT LENGTH (= MAT), never the committed
# tokens' identity. Two consequences, both simplifying vs the EAGLE path:
#   * no trajectory PIN / GT force / verify override is needed (greedy pins it);
#   * the per-decision GT (target greedy at each block position d) is recovered
#     POST-verify, drift-free, from the request's OWN committed output_ids
#     (committed[d] == GT at depth d for d in 0..accept_len), and oracle_hit is
#     BACKFILLED onto the decision records there.
# Only the ORACLE arm needs GT at decision time (to substitute the GT-matching
# proposer), so it loads a record-arm GT dump; record mode produces it.
#
# Mirrors the offline capture (capture_dflash_vs_suffix.py) exactly: per-depth
# autoregressive suffix query with the CHOSEN chain (== GT prefix on alive
# depths, so selacc/MAT match), max_spec_tokens=1, temporary_extension; and
# dflash_p = softmax(lm_head(draft_hidden)).max() recomputed from the same
# hidden the sampler used.

def _dflash_blocksize(st) -> int:
    return int(getattr(st.worker, "block_size", 0) or 0)


def _install_dflash_prob_stash(worker) -> None:
    """Wrap _greedy_sample_from_vocab_parallel_head so each block position's
    max-softmax prob (the DFlash analog of eagle_p) is stashed next to the
    argmax token the sampler returns (the sampler returns only argmax ids).
    Recompute matches the offline capture's softmax(lm_head(draft_hidden)).max()
    on the tp==1 / no-added-vocab fast path (the 8B serving config)."""
    if getattr(worker, "_chain_hybrid_dflash_prob_patched", False):
        return
    import torch
    orig = worker._greedy_sample_from_vocab_parallel_head

    def wrapped(*, hidden_states, lm_head, chunk_size=256):
        toks = orig(hidden_states=hidden_states, lm_head=lm_head,
                    chunk_size=chunk_size)
        st = _STATE
        if st is None:
            return toks
        st._dflash_p_flat = None
        try:
            if hidden_states is not None and hidden_states.numel() > 0:
                shard = lm_head.shard_indices
                num_org = int(shard.num_org_elements)
                num_added = int(shard.num_added_elements)
                if num_added != 0:
                    st.warn_once(
                        "dflash-prob-added",
                        "lm_head exposes added vocab; dflash_p computed over the "
                        "base-vocab shard only (approximate)")
                w = lm_head.weight
                hs = (hidden_states if hidden_states.dtype == w.dtype
                      else hidden_states.to(w.dtype))
                logits = torch.matmul(hs, w[:num_org].T).float()
                mp = torch.softmax(logits, dim=-1).max(dim=-1).values
                st._dflash_p_flat = mp.detach().to("cpu")
        except Exception as e:
            st.warn_once("dflash-prob", str(e))
            st._dflash_p_flat = None
        return toks

    worker._greedy_sample_from_vocab_parallel_head = wrapped
    worker._chain_hybrid_dflash_prob_patched = True
    logger.info("chain-hybrid DFlash: per-position prob stash installed "
                "(_greedy_sample_from_vocab_parallel_head)")


def _decide_block_dflash(st, dt_gpu, dt_cpu, dfp_2d, bs, b):
    """Per-position select-1 over the DFlash block (positions 1..b-1; pos 0 is
    the committed seed). Reads tokens/probs from CPU snapshots, writes chosen
    suffix tokens back into dt_gpu IN PLACE (aliases batch.input_ids), and emits
    per-depth decision records into st.pending + st._dflash_recidx (oracle_hit
    backfilled post-verify, except oracle mode which sets it here from the GT
    dump). depth d == block position d+1, logged 0-based so the join rule
    accept_len >= depth+1 matches the EAGLE / offline convention.

    The per-(row,depth) decision body mirrors _decide_and_inject; it is
    duplicated rather than shared so the validated EAGLE/MTP path is untouched."""
    for r, (rid, ctx_tail) in enumerate(st.stash):
        chain = st.chains[r]
        ds = st.decode_step.get(rid, 0)
        recs_by_depth = st._dflash_recidx.setdefault((rid, ds), {})
        for bp in range(1, b):
            depth = bp - 1
            eagle_tok = int(dt_cpu[r, bp])
            eagle_p = (float(dfp_2d[r, depth]) if (dfp_2d is not None
                       and depth < dfp_2d.shape[1]) else 0.0)
            rec = {
                "type": "decision", "rid": rid, "decode_step": ds,
                "depth": depth, "eagle_token": eagle_tok,
                "eagle_p": round(eagle_p, 6), "eagle_p_cal": None,
                "suffix_token": None, "suffix_p": None, "suffix_count": None,
                "suffix_total": None, "suffix_p_cal": None, "match_len": None,
                "suffix_score": None, "chosen": "eagle3", "agreement": None,
            }
            suffix_tok = suffix_p = suffix_count = suffix_total = None
            if rid in st.active:
                try:
                    ctx = (list(ctx_tail) + chain)[-st.cache.max_tree_depth:]
                    if chain:
                        with st.cache.temporary_extension(rid, chain):
                            draft = st.cache.speculate(
                                rid, ctx, max_spec_tokens=1, use_tree_spec=False)
                    else:
                        draft = st.cache.speculate(
                            rid, ctx, max_spec_tokens=1, use_tree_spec=False)
                    tok_ids = getattr(draft, "token_ids", None)
                    if tok_ids is not None and len(tok_ids):
                        suffix_tok = int(tok_ids[0])
                        probs = getattr(draft, "probs", None)
                        suffix_p = float(probs[0]) if (probs is not None
                                                       and len(probs)) else 0.0
                        rec["suffix_token"] = suffix_tok
                        rec["suffix_p"] = round(suffix_p, 6)
                        rec["match_len"] = int(getattr(draft, "match_len", 0) or 0)
                        rec["suffix_score"] = round(float(
                            getattr(draft, "score", 0.0) or 0.0), 4)
                        counts = getattr(draft, "counts", None)
                        if counts and suffix_p > 0.0:
                            suffix_count = int(counts[0])
                            suffix_total = int(round(suffix_count / suffix_p))
                            rec["suffix_count"] = suffix_count
                            rec["suffix_total"] = suffix_total
                except Exception as e:
                    st.warn_once("speculate-dflash", str(e))

            # ORACLE mode: GT (from the record-arm dump) decides the chain.
            if st.mode == "oracle":
                gt_list = st.gt.get(rid)
                L = (st.gt_pos[r] if st.gt_pos is not None
                     and r < len(st.gt_pos) else None)
                gt_tok = (int(gt_list[L + depth]) if (gt_list is not None
                          and L is not None and L + depth < len(gt_list))
                          else None)
                rec["gt_token"] = gt_tok
                rec["agreement"] = (suffix_tok == eagle_tok
                                    if suffix_tok is not None else None)
                chosen_tok = eagle_tok
                hit = "nogt" if gt_tok is None else "none"
                if gt_tok is not None:
                    e_hit = eagle_tok == gt_tok
                    s_hit = (suffix_tok == gt_tok
                             if suffix_tok is not None else False)
                    if e_hit:
                        hit = "both" if s_hit else "eagle"
                    elif s_hit:
                        hit = "suffix"
                        dt_gpu[r, bp] = suffix_tok
                        chosen_tok = suffix_tok
                        rec["chosen"] = "suffix"
                rec["oracle_hit"] = hit
                chain.append(chosen_tok)
                recs_by_depth[depth] = rec
                st.pending.append(rec)
                continue

            # select1 (raw / calibrated / discriminator) — identical to the
            # EAGLE _decide_and_inject decision body.
            suffix_cmp = suffix_p
            eagle_cmp = eagle_p
            if _CALIB is not None:
                eagle_cmp = _CALIB.predict("eagle", eagle_p, eagle_p, depth=depth)
                rec["eagle_p_cal"] = round(eagle_cmp, 6)
                if suffix_p is not None:
                    p_in = suffix_p
                    if _CALIB.wants_shrunk:
                        if suffix_count is not None and suffix_total:
                            p_in = (suffix_count + 0.5) / (suffix_total + 1)
                        else:
                            st.warn_once(
                                "no-counts-dflash",
                                "calib map wants Jeffreys-shrunk probs but the "
                                "suffix draft exposed no counts; using raw "
                                "suffix_p")
                    suffix_cmp = _CALIB.predict(
                        "suffix", p_in, suffix_p, depth=depth,
                        count=suffix_count, total=suffix_total,
                        match_len=rec.get("match_len"))
                    rec["suffix_p_cal"] = round(suffix_cmp, 6)

            chosen_tok = eagle_tok
            if suffix_tok is not None:
                rec["agreement"] = suffix_tok == eagle_tok
                if suffix_tok != eagle_tok:
                    if _DISC is not None and suffix_p is not None:
                        pick_p = _DISC.predict(suffix_p, eagle_p,
                                               rec.get("match_len"),
                                               suffix_count, suffix_total,
                                               depth=depth)
                        rec["disc_p"] = round(float(pick_p), 6)
                        take = pick_p > 0.5
                    else:
                        take = (suffix_cmp is not None and suffix_cmp > eagle_cmp)
                    if take:
                        dt_gpu[r, bp] = suffix_tok
                        chosen_tok = suffix_tok
                        rec["chosen"] = "suffix"
            chain.append(chosen_tok)
            recs_by_depth[depth] = rec
            st.pending.append(rec)


def _patch_dflash_prepare(worker) -> None:
    """Wrap _prepare_for_speculative_decoding: build the per-row committed-ctx
    stash (pre-draft), then after the block is drafted run the per-position
    select-1 and substitute chosen tokens into the verify block in place."""
    original_prepare = worker._prepare_for_speculative_decoding

    def chain_prepare(batch, draft_input):
        st = _STATE
        try:
            is_decode = not (batch.forward_mode.is_extend()
                             or batch.forward_mode.is_idle())
        except Exception:
            is_decode = True
        if st is None or not is_decode:
            return original_prepare(batch, draft_input)

        st.batch_counter += 1
        st._dflash_p_flat = None
        stash = []
        chains = []
        gt_pos = []
        try:
            for req in batch.reqs:
                rid = req.rid
                if st.mode == "record":
                    stash.append((rid, ())); chains.append([]); gt_pos.append(0)
                    continue
                if rid not in st.active:
                    try:
                        st.cache.start_request(rid, list(req.origin_input_ids))
                        st.active.add(rid)
                        # trie-feed pointer starts at 0 so the prefill bonus is
                        # fed too (matches the EAGLE path's last_out_len=0).
                        st.last_out_len[rid] = 0
                        st.decode_step[rid] = 0
                    except Exception as e:
                        st.warn_once("start_request-dflash", str(e))
                    if st.mode == "oracle" or st.pin_active:
                        g = (st.gt_map or {}).get(tuple(req.origin_input_ids))
                        st.gt[rid] = list(g) if g is not None else None
                        st.gt_stats["matched" if g is not None
                                    else "unmatched"] += 1
                        st.pending.append({
                            "type": "req", "rid": rid,
                            "input_ids": [int(x) for x in req.origin_input_ids]})
                        if g is None:
                            st.warn_once(
                                "gt-unmatched-dflash",
                                "request prompt not in GT dump; running pure "
                                "DFlash for unmatched requests")
                st.decode_step[rid] = st.decode_step.get(rid, 0) + 1
                st.last_seen[rid] = st.batch_counter
                Lnow = len(req.output_ids or [])
                ctx = (list(req.origin_input_ids)
                       + list(req.output_ids))[-st.cache.max_tree_depth:]
                stash.append((rid, ctx)); chains.append([]); gt_pos.append(Lnow)
                # gt_pos for THIS step's commit-slice backfill / step accounting.
                st._dflash_gtpos[(rid, st.decode_step[rid])] = Lnow
        except Exception as e:
            st.warn_once("dflash-prep-stash", str(e))
            stash = None
        st.stash = stash
        st.chains = chains if stash is not None else None
        gt_on = (stash is not None and (st.mode == "oracle" or st.pin_active))
        st.gt_pos = gt_pos if gt_on else None
        # Per-row GT continuation over the whole verify block [block_size], used
        # by the verify-pin override to force target_predict (= the committed
        # trajectory) onto the GT-dump path so EVERY arm follows the identical
        # trajectory (greedy alone does NOT pin across arms — the substituted
        # block changes the verify batch, flipping FP near-ties; see
        # project_chain_hybrid_fp_nondeterminism). -1 past GT end.
        if gt_on:
            b = _dflash_blocksize(st)
            ov = []
            for r, (rid, _ctx) in enumerate(stash):
                g = st.gt.get(rid)
                L = gt_pos[r] if r < len(gt_pos) else None
                if g is not None and L is not None:
                    ov.append([int(g[L + j]) if (L + j) < len(g) else -1
                               for j in range(b)])
                else:
                    ov.append([-1] * b)
            st.gt_predict_override = ov
        else:
            st.gt_predict_override = None

        result = original_prepare(batch, draft_input)

        try:
            if (st.stash is not None and st.mode != "record"):
                vi = getattr(batch, "spec_info", None)
                dt_flat = getattr(vi, "draft_token", None) if vi is not None else None
                b = _dflash_blocksize(st)
                bs = len(st.stash)
                if dt_flat is not None and b > 1 and dt_flat.numel() == bs * b:
                    dt_gpu = dt_flat.view(bs, b)
                    dt_cpu = dt_gpu.detach().cpu()
                    dfp = st._dflash_p_flat
                    dfp_2d = (dfp.view(bs, b - 1) if (dfp is not None
                              and dfp.numel() == bs * (b - 1)) else None)
                    _decide_block_dflash(st, dt_gpu, dt_cpu, dfp_2d, bs, b)
                elif dt_flat is not None:
                    st.warn_once(
                        "dflash-shape",
                        f"draft_token numel={dt_flat.numel()} != bs*b="
                        f"{bs*b} (b={b}); skipping select-1 this step")
        except Exception as e:
            st.warn_once("dflash-decide", str(e))
        return result

    worker._prepare_for_speculative_decoding = chain_prepare


def _patch_dflash_forward(worker) -> None:
    """Wrap forward_batch_generation: after the verify commits, backfill the
    per-decision GT/oracle_hit from the request's own committed tokens, feed the
    suffix trie the newly committed tokens, emit the per-step accept_len record,
    and flush. record mode dumps GT trajectories at request finish."""
    original_forward = worker.forward_batch_generation

    def chain_forward(batch, *a, **k):
        result = original_forward(batch, *a, **k)
        st = _STATE
        if st is None:
            return result
        try:
            try:
                is_decode = not (batch.forward_mode.is_extend()
                                 or getattr(batch, "is_extend_in_batch", False)
                                 or batch.forward_mode.is_idle())
            except Exception:
                is_decode = True
            if not is_decode:
                return result
            for req in batch.reqs:
                rid = req.rid
                if st.mode == "record":
                    if req.finished():
                        _dump_gt(st, req)
                    continue
                if rid not in st.active:
                    continue
                ds = st.decode_step.get(rid, 0)
                out = req.output_ids or []
                gt_pos = st._dflash_gtpos.get((rid, ds), st.last_out_len.get(rid, 0))
                committed = [int(t) for t in out[gt_pos:]]
                n_committed = len(committed)
                acc_len = max(0, n_committed - 1)

                # Backfill GT + oracle_hit (drift-free: committed[d] == target
                # greedy at depth d). oracle mode already set oracle_hit.
                if st.mode != "oracle":
                    recs = st._dflash_recidx.get((rid, ds), {})
                    for depth, rec in recs.items():
                        if "oracle_hit" in rec:
                            continue
                        gt_tok = committed[depth] if depth < n_committed else None
                        rec["gt_token"] = gt_tok
                        if gt_tok is None:
                            rec["oracle_hit"] = "nogt"
                        else:
                            e_hit = rec.get("eagle_token") == gt_tok
                            s_tok = rec.get("suffix_token")
                            s_hit = (s_tok == gt_tok) if s_tok is not None else False
                            rec["oracle_hit"] = ("both" if (e_hit and s_hit)
                                                 else "eagle" if e_hit
                                                 else "suffix" if s_hit else "none")

                # Incremental trie update: feed every token committed since the
                # last feed (out[last_out_len:] includes the first-step prefill
                # bonus). add_active_response is the official Arctic semantics.
                prev_fed = st.last_out_len.get(rid, 0)
                if len(out) > prev_fed:
                    try:
                        st.cache.add_active_response(
                            rid, [int(t) for t in out[prev_fed:]])
                    except Exception as e:
                        st.warn_once("add_active_response-dflash", str(e))
                    st.last_out_len[rid] = len(out)

                st.pending.append({"type": "step", "rid": rid,
                                   "decode_step": ds, "accept_len": acc_len})
                st._dflash_recidx.pop((rid, ds), None)
                st._dflash_gtpos.pop((rid, ds), None)

                if req.finished():
                    try:
                        st.cache.stop_request(rid)
                    except Exception as e:
                        st.warn_once("stop_request-dflash", str(e))
                    st.active.discard(rid)
                    st.last_out_len.pop(rid, None)
                    st.decode_step.pop(rid, None)
                    st.last_seen.pop(rid, None)
                    st.gt.pop(rid, None)

            if st.batch_counter % GC_INTERVAL == 0:
                _gc_stale(st)
                if st.mode == "oracle":
                    logger.info(f"chain-hybrid DFlash oracle gt_stats: "
                                f"{st.gt_stats}")
            st.flush()
        except Exception as e:
            st.warn_once("dflash-forward-hook", str(e))
        return result

    worker.forward_batch_generation = chain_forward


def _install_dflash_verify_pin() -> None:
    """Force DFlash verify's target_predict onto the GT-dump trajectory (oracle +
    pin). DFlashVerifyInput.verify computes target_predict = argmax(next_token_
    logits).view(bs, block); the committed tokens are that argmax. We bias the
    logits so argmax == GT[L+j] at every block position j (st.gt_predict_override,
    set per step in the prepare wrapper), making verify NATURALLY commit the GT
    trajectory regardless of the substituted block -> every arm follows the
    IDENTICAL trajectory (fair MAT; differences come only from selection, i.e.
    the accept length on that shared trajectory). -1 entries (past GT end) are
    left as the real argmax so the request finishes naturally."""
    import sglang.srt.speculative.dflash_info as di
    if getattr(di.DFlashVerifyInput, "_chain_hybrid_pin_wrapped", False):
        return
    import torch
    orig = di.DFlashVerifyInput.verify

    def wrapped(self_vi, *, batch, logits_output, page_size):
        st = _STATE
        ov = getattr(st, "gt_predict_override", None) if st is not None else None
        if ov is not None:
            try:
                lg = logits_output.next_token_logits  # [bs*block, vocab]
                block = int(self_vi.draft_token_num)
                bs = lg.shape[0] // max(block, 1)
                bump = float(lg.max().item()) + 10.0
                for r in range(min(bs, len(ov))):
                    row = ov[r]
                    for j in range(min(block, len(row))):
                        g = row[j]
                        if g is not None and g >= 0:
                            lg[r * block + j, int(g)] = bump
            except Exception as e:
                if st is not None:
                    st.warn_once("dflash-verify-pin", str(e))
        return orig(self_vi, batch=batch, logits_output=logits_output,
                    page_size=page_size)

    di.DFlashVerifyInput.verify = wrapped
    di.DFlashVerifyInput._chain_hybrid_pin_wrapped = True
    logger.info("chain-hybrid DFlash: verify target_predict GT-pin installed")


def patch_chain_hybrid_dflash(worker) -> None:
    """Entry point for DFlash chain-hybrid select-1 (mirrors patch_chain_hybrid
    for the EAGLE path). Installed by install_hook when SGLANG_CHAIN_HYBRID=1 on
    a DFlashWorker. No topk / disable-cuda-graph / latency-only invariants:
    DFlash drafts a block per step in Python (the substitution + suffix lookup
    run regardless of target cuda-graph) and there is no force-accept to nullify.
    Cross-arm fairness needs trajectory PINNING (SGLANG_CHAIN_HYBRID_PIN / oracle
    GT): greedy alone does NOT pin across arms, since the substituted block
    changes the verify batch and flips FP near-ties."""
    global _STATE, _CALIB, _DISC, _ONLINE

    # NOTE: block_size is NOT yet assigned when this runs — the install_hook
    # dispatch fires early in DFlashWorker.__init__ (right after
    # self.draft_model is set), before block_size is resolved. It is read
    # lazily at decode time via _dflash_blocksize(st) (the prepare wrapper only
    # fires after __init__ completes), where it is guaranteed set.

    from hybrid_spec_decoding.suffix_decoding.suffix_tree import (
        SuffixDecodingCache,
    )
    suffix_cache = SuffixDecodingCache(
        max_tree_depth=64, max_cached_requests=100000, max_spec_tokens=1,
        max_spec_factor=1.0, max_spec_offset=0.0, min_token_prob=0.1,
        use_tree_spec=False, enable_undo=True)

    log_path = os.environ.get(
        "SGLANG_CHAIN_HYBRID_LOG",
        "/tmp/sglang_chain_hybrid_dflash_decisions.jsonl")
    mode = os.environ.get("SGLANG_CHAIN_HYBRID_MODE", "select1")
    if mode not in ("select1", "record", "oracle"):
        raise RuntimeError(
            f"DFlash chain-hybrid mode must be select1/record/oracle, "
            f"got {mode!r} (score_fallback/online/tail not supported).")

    _STATE = _ChainHybridState(worker, suffix_cache, log_path, mode=mode)
    _STATE._dflash_recidx = {}
    _STATE._dflash_gtpos = {}
    _STATE._dflash_p_flat = None

    # Trajectory PIN (select1/calib/disc arms): force committed tokens onto a
    # standalone GT dump so every arm follows the identical trajectory. oracle
    # mode pins via its own GT (SGLANG_CHAIN_HYBRID_GT); record must not pin.
    pin_path = os.environ.get("SGLANG_CHAIN_HYBRID_PIN")
    if pin_path and mode == "record":
        raise RuntimeError("SGLANG_CHAIN_HYBRID_PIN incompatible with mode=record")
    _STATE.pin_active = bool(pin_path) and mode == "select1"
    if _STATE.pin_active:
        _STATE.gt_map = _load_gt_map(pin_path)

    # Reuse the EAGLE serving calibrators / discriminator unchanged.
    _ONLINE = None
    calib_path = os.environ.get("SGLANG_CHAIN_HYBRID_CALIB")
    mf_path = os.environ.get("SGLANG_CHAIN_HYBRID_MULTIFEAT")
    if mf_path:
        _CALIB = _ServingMultiFeatCalibrator.load(mf_path)
        calib_desc = f"multifeat(kind={_CALIB.kind}, map={mf_path})"
    elif calib_path:
        _CALIB = _ServingIsoCalibrator.load(calib_path)
        calib_desc = (f"calib(map={calib_path}, "
                      f"shrink={'jeffreys' if _CALIB.wants_shrunk else 'none'})")
    else:
        _CALIB = None
        calib_desc = "raw suffix_p > dflash_p (uncalibrated)"

    disc_path = os.environ.get("SGLANG_CHAIN_HYBRID_DISC")
    _DISC = _ServingDiscriminator.load(disc_path) if disc_path else None

    if mode == "record":
        _STATE.gt_out_path = os.environ.get(
            "SGLANG_CHAIN_HYBRID_GT_OUT",
            "/tmp/sglang_chain_hybrid_dflash_gt.jsonl")
        mode_desc = f"mode=record (GT dump -> {_STATE.gt_out_path})"
    elif mode == "oracle":
        gt_path = os.environ.get("SGLANG_CHAIN_HYBRID_GT")
        if not gt_path:
            raise RuntimeError(
                "DFlash chain-hybrid oracle mode requires SGLANG_CHAIN_HYBRID_GT")
        _STATE.gt_map = _load_gt_map(gt_path)
        mode_desc = (f"mode=oracle (selection ceiling, {len(_STATE.gt_map)} GT "
                     f"trajectories from {gt_path})")
    else:
        mode_desc = (f"mode=select1, decision={calib_desc}"
                     + (f", disc(kind={_DISC.kind}, P>0.5, "
                        f"depth={'yes' if _DISC.with_depth else 'no'})"
                        if _DISC is not None else ""))

    if _STATE.pin_active:
        mode_desc += f" [PINNED to {len(_STATE.gt_map)} trajectories from {pin_path}]"

    _install_dflash_prob_stash(worker)
    _patch_dflash_prepare(worker)
    _patch_dflash_forward(worker)
    if mode == "oracle" or _STATE.pin_active:
        _install_dflash_verify_pin()
    logger.info(
        f"Chain-hybrid DFlash patch applied: {mode_desc}, "
        f"block_size=(resolved at runtime), decision log -> {log_path}")
