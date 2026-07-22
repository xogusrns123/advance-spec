#!/usr/bin/env python3
"""Pure-CPU offline REPLAY of the full Extension mechanism from a capture_perpos
record (+ its .traces.json). NO model forward, NO serving — the DFlash block
proposals are read from the record and the Suffix side is model-free (the tree is
rebuilt from the warm traces on CPU). Reproduces:

  * the partial-warm crossover K per proposer (slide 9 / 23 / 24), and
  * (via analyze_perpos) hazard/survival/calibration/controller (slides 4/5/7/13b/18)

Because it is arithmetic, controller variants (argmax / first-crossing / threshold),
tree budget policies (even / bestfirst), and T-estimators can be swept with zero
GPU. This is the key point: after ONE collection pass, everything is replay.

  python3 scripts/replay_extension.py --record results/perpos/multislot_k0.jsonl \
      --props dflash suffix chain tree --alloc even
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/workspace")
from fusion_tree import build_extension_chain, build_extension_tree, adaptive_nhead  # noqa: E402
from measure_k_fusion import ArcticSuffix  # noqa: E402  (CPU: arctic + numpy)
from simulation.evaluation.tree_knapsack import greedy_tree_walk_path  # noqa: E402


def _gd(conf):
    """DFlash expected accepted tokens = sum of survival of the affine hazard
    a_j = 0.69*conf_j + 0.29 (Dr.Lee's calibration)."""
    S, G = 1.0, 0.0
    for c in conf:
        S *= min(1.0, max(0.0, 0.69 * float(c) + 0.29))
        G += S
    return G


def _ad(match):
    """Leading-1 run of a match vector = block's greedy accept length."""
    i = 0
    while i < len(match) and match[i] == 1:
        i += 1
    return i


def _fit_iso(xs, ys):
    """Isotonic signal->E[accept]; identity fallback when too few distinct xs."""
    from sklearn.isotonic import IsotonicRegression
    if len(set(xs)) < 3:
        mean = (sum(ys) / len(ys)) if ys else 0.0
        return lambda v: mean
    ir = IsotonicRegression(out_of_bounds="clip")
    import numpy as _np
    ir.fit(_np.asarray(xs, float), _np.asarray(ys, float))
    return lambda v: float(ir.predict([v])[0])


def _fit_logistic(xs, ys):
    """Smooth monotone P(match|conf) = sigmoid(w*conf+b) (Platt). Robust on small
    calibrate splits (2 params, no isotonic step-overfitting)."""
    import numpy as _np
    from sklearn.linear_model import LogisticRegression
    ys = _np.asarray(ys, int)
    if len(set(ys.tolist())) < 2:
        p = float(ys.mean()) if len(ys) else 0.0
        return lambda v: p
    X = _np.asarray(xs, float).reshape(-1, 1)
    lr = LogisticRegression(C=1.0, solver="lbfgs")
    lr.fit(X, ys)
    return lambda v: float(lr.predict_proba([[v]])[0, 1])


def _fit_beta(xs, ys):
    """Beta calibration (Kull et al. 2017): logistic regression on
    (ln p, -ln(1-p)). Unlike a plain sigmoid(w*conf+b) — whose S-curve is
    symmetric around its midpoint — the log-odds features diverge at the [0,1]
    boundaries, so it can hug 0 at conf->0 and 1 at conf->1 with asymmetric
    shape. Negative-weight features are dropped and refit (the standard
    beta-calib reduction) to keep the map monotone; if both drop, falls back
    to plain logistic."""
    import math
    import numpy as _np
    from sklearn.linear_model import LogisticRegression
    ys_a = _np.asarray(ys, int)
    if len(set(ys_a.tolist())) < 2:
        p = float(ys_a.mean()) if len(ys_a) else 0.0
        return lambda v: p
    eps = 1e-6
    x = _np.clip(_np.asarray(xs, float), eps, 1.0 - eps)
    feats = {0: _np.log(x), 1: -_np.log(1.0 - x)}
    keep = [0, 1]
    lr = None
    while keep:
        X = _np.column_stack([feats[i] for i in keep])
        lr = LogisticRegression(C=1.0, solver="lbfgs")
        lr.fit(X, ys_a)
        neg = [i for i, c in zip(keep, lr.coef_[0]) if c < 0]
        if not neg:
            break
        keep = [i for i in keep if i not in neg]
    if not keep:
        return _fit_logistic(xs, ys)
    w = {i: float(c) for i, c in zip(keep, lr.coef_[0])}
    b0 = float(lr.intercept_[0])

    def fn(v):
        v = min(max(float(v), eps), 1.0 - eps)
        z = b0 + w.get(0, 0.0) * math.log(v) - w.get(1, 0.0) * math.log(1.0 - v)
        return 1.0 / (1.0 + math.exp(-z))
    return fn


def _fit_tail_iso(warm_traces, recs, eval_traces, calib_rids, num_spec, max_rounds,
                  return_pairs=False):
    """Suffix-side twin of the hazard calibrator: isotonic map arctic tail score ->
    E[realized tail accept], fit on the CALIBRATE split only. The raw arctic score
    over-estimates the realized accept 3-6x (bfcl: mean score 22.6 -> realized 3.8),
    so using it uncalibrated in the controller objective mis-prices the tail.

    Pairs are collected accept-conditioned (candidate handoffs k <= a_d, where the
    head survives so the tail label is realizable) along a chain-policy (deployed
    argmax affine) trajectory, with the BUDGET-AWARE score (max_spec_tokens =
    num_spec - k) so deep handoffs price their shrunken tail budget."""
    suffix = ArcticSuffix(); suffix.fit(warm_traces)
    xs, ys = [], []
    for rid in sorted(calib_rids):
        tr = eval_traces.get(rid)
        if tr is None:
            continue
        gt, pids, rby = tr["output_ids"], tr["prompt_ids"], recs[rid]
        suffix.new_eval(pids)
        m = 0
        for _ in range(max_rounds):
            rec = rby.get(m + 1)
            if rec is None or m >= len(gt):
                break
            block_full, cf, mt = rec["dflash_tok"], rec["dflash_conf"], rec["dflash_match"]
            root = gt[m]
            ctx_list = pids + gt[:m] + [root]
            for kk in range(min(_ad(mt), rec["W"]) + 1):
                budget = num_spec - kk
                if budget <= 0:
                    break
                tk, sc = suffix._spec(ctx_list + block_full[:kk], budget)
                r_acc = 0
                for t, g in zip(tk[:budget], gt[m + 1 + kk:]):
                    if t != g:
                        break
                    r_acc += 1
                xs.append(float(sc)); ys.append(float(r_acc))
            suf, T = suffix.probe(ctx_list, num_spec)
            k = adaptive_nhead(cf, T=T, num_spec=rec["W"])
            tk = suffix.speculate(ctx_list + block_full[:k], num_spec)
            trd = build_extension_chain(block_full[:k], tk[:max(0, num_spec - k)])
            pth = greedy_tree_walk_path(list(trd.tokens), list(trd.parents), gt[m + 1:])
            accepted = [trd.tokens[i] for i in pth]
            nxt = [root] + accepted
            if m + 1 + len(pth) < len(gt):
                nxt.append(gt[m + 1 + len(pth)])
            suffix.add_response(nxt)
            m += 1 + len(pth) + 1
            if m >= len(gt):
                break
    if len(set(xs)) < 3:
        mean = (sum(ys) / len(ys)) if ys else 0.0
        fn = lambda v: mean                                            # noqa: E731
    else:
        from sklearn.isotonic import IsotonicRegression
        import numpy as _np
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(_np.asarray(xs, float), _np.asarray(ys, float))
        fn = lambda v: float(ir.predict([max(0.0, float(v))])[0])      # noqa: E731
    return (fn, xs, ys) if return_pairs else fn


def split_parity(eval_traces, group_mode):
    """Three-way disjoint split shared by every replay script: returns
    (conv_of, split_par) where split_par[rid] is 0 = calibrate, 1 = test.
    rid: each request its own group. lenreset: conversations inferred from
    prompt-length resets. conv: exact conversation ids (densely re-ranked).
    convlabel: conv ids, parity by the conv's rank WITHIN its task label (keeps
    every subtask represented on both sides of interleaved datasets)."""
    conv_of = {}
    split_par = None
    if group_mode == "lenreset":
        cid, prev = -1, float("inf")
        for rid in sorted(eval_traces):
            L = len(eval_traces[rid]["prompt_ids"])
            if L < prev:
                cid += 1
            conv_of[rid] = cid
            prev = L
    elif group_mode in ("conv", "convlabel"):
        raw = {rid: eval_traces[rid].get("conv", rid) for rid in eval_traces}
        order = {c: i for i, c in enumerate(sorted(set(raw.values())))}
        conv_of = {rid: order[raw[rid]] for rid in raw}
        if group_mode == "convlabel":
            lab_of = {}
            for rid in eval_traces:
                lab_of.setdefault(raw[rid], eval_traces[rid].get("task", ""))
            rank, par = {}, {}
            for c in sorted(set(raw.values())):
                l = lab_of[c]
                par[c] = rank.get(l, 0) % 2
                rank[l] = rank.get(l, 0) + 1
            split_par = {rid: par[raw[rid]] for rid in raw}
    else:
        conv_of = {rid: rid for rid in eval_traces}
    if split_par is None:
        split_par = {rid: conv_of.get(rid, 0) % 2 for rid in eval_traces}
    return conv_of, split_par


class OnlineCalib:
    """ONLINE windowed calibration (deployable arm 1): logistic head + isotonic
    tail refit on a sliding window of the labels a real deployment observes for
    free at verify time — per-depth head matches along the CHOSEN head prefix
    (accept-conditioned, incl. the first miss) and the realized tail accept of
    the CHOSEN handoff when the head survived. No offline fit, no calib split;
    the calibrators warm up on the serving stream itself (identity until
    min_pairs). Refit every `refit_every` rounds on the last `window` pairs."""

    def __init__(self, window=16000, refit_every=200, min_head=300, min_tail=100,
                 head="logistic", tail="isotonic", tail_scale=0.0):
        from collections import deque
        self.hx = deque(maxlen=window)
        self.hy = deque(maxlen=window)
        self.tx = deque(maxlen=max(1, window // 4))
        self.ty = deque(maxlen=max(1, window // 4))
        self.refit_every = refit_every
        self.min_head, self.min_tail = min_head, min_tail
        self.head, self.tail = head, tail          # head: logistic|beta|linear
        self._h = lambda c: float(c)               # tail: isotonic|linear|raw|fixed
        # fixed: FROZEN scalar map E[acc|s]=w*s (no online refit, deploy-time constant)
        self._t = ((lambda v, _w=float(tail_scale): _w * max(0.0, float(v)))
                   if tail == "fixed" else (lambda v: float(v)))
        self._n_obs = 0

    def cal_h(self, c):
        return self._h(c)

    def cal_t(self, v):
        return self._t(v)

    def observe(self, confs, k, head_acc, tail_score, tail_acc, aux_succ=None):
        """confs: block conf list; k: chosen head length; head_acc: realized
        leading-match run within the chosen head; tail_score/tail_acc: budgeted
        score + realized accept of the grafted tail (None if head died).
        aux_succ: unused here (AdaptiveScalar's label-free succratio signal)."""
        for j in range(min(head_acc, k)):
            self.hx.append(float(confs[j])); self.hy.append(1)
        if head_acc < k:                     # observed first miss
            self.hx.append(float(confs[head_acc])); self.hy.append(0)
        if tail_acc is not None:
            self.tx.append(float(tail_score)); self.ty.append(float(tail_acc))
        self._n_obs += 1
        if self._n_obs % self.refit_every == 0:
            self._refit()

    def _refit(self):
        import numpy as _np
        if (self.head not in ("raw", "none")
                and len(self.hx) >= self.min_head and len(set(self.hy)) > 1):
            try:
                if self.head == "beta":
                    self._h = _fit_beta(list(self.hx), list(self.hy))
                elif self.head == "linear":
                    a, b = _np.polyfit(_np.asarray(self.hx, float),
                                       _np.asarray(self.hy, float), 1)
                    self._h = lambda c, _a=a, _b=b: min(1.0, max(0.0, float(_a) * float(c) + float(_b)))
                else:                                        # logistic
                    self._h = _fit_logistic(list(self.hx), list(self.hy))
            except Exception:
                pass
        if (self.tail not in ("raw", "none", "fixed")
                and len(self.tx) >= self.min_tail and len(set(self.tx)) >= 3):
            try:
                if self.tail == "linear":
                    a, b = _np.polyfit(_np.asarray(self.tx, float),
                                       _np.asarray(self.ty, float), 1)
                    self._t = lambda v, _a=a, _b=b: max(0.0, float(_a) * float(v) + float(_b))
                else:                                        # isotonic
                    from sklearn.isotonic import IsotonicRegression
                    ir = IsotonicRegression(out_of_bounds="clip")
                    ir.fit(_np.asarray(self.tx, float), _np.asarray(self.ty, float))
                    self._t = lambda v, _ir=ir: float(_ir.predict([max(0.0, float(v))])[0])
            except Exception:
                pass


class AdaptiveScalar(OnlineCalib):
    """FEEDBACK-LOOP compose scalars (adaptive arm): the fixed tail weight w
    (and optionally head weight u) becomes an online update driven by verify-
    time signals from the eval stream itself. Modes:
      ftl / ftl2d — counterfactual follow-the-leader over a w (or u x w) grid:
        every round each candidate is credited the realized accept of the
        handoff k it WOULD have picked at this state (grid argmax on the same
        cached conf + per-k tail scores; only distinct ks are walked).
        censor=True clips the credit to the verifier-revealed prefix
        (accepted + bonus = acc_chosen+1) — the deployable-information tier.
        Leader = best windowed/EWMA mean with dwell (switch_every) +
        hysteresis (delta); ties keep the incumbent.
      ratio     — w_t = gamma * (sum tail_acc / sum tail_score) over the window
        of observed handoffs: mean-matching slope x fixed decision shading
        gamma (MAT-optimal w sits 2.5-5.6x BELOW the raw slope), damped
        update w <- (1-damp)*w + damp*target.
      succratio — LABEL-FREE: w_t = gamma * (sum succ / sum raw) of the chosen
        tails (succession aggregate discount rho_succ; no accept labels;
        gamma~0.6 was derived in the u=1.4 head frame).
      sgdgain — EXPLICIT per-side gain-residual loss (user formulation), both
        scalars continuous: head L=(G_hat(u)-head_acc)^2 with G_hat = sum of
        survival prefixes of min(1,u*conf); tail L=(w*s-tail_acc)^2 at the
        chosen k. Per-round SGD; verify-time labels only (deployable).
      sgdmat — EXPLICIT total-accept loss, single continuous w: the discrete
        argmax is relaxed to pi_k = softmax(beta*val_k(w)) so V(w) =
        sum_k pi_k * acc_k is differentiable; per-round gradient ASCENT with
        the counterfactual per-k accept vector (full-info; censor clips it).
    head: raw | scale (u = head_scale; the adaptive u-grid axis in ftl2d) |
    beta / logistic (inherited OnlineCalib online head refit on verify labels;
    tail refit stays off via tail='fixed')."""

    def __init__(self, mode="ftl", head="raw", wgrid=(0.075,), ugrid=(1.4,),
                 adapt_window=8000, ewma_halflife=0, init_w=0.075, init_u=1.4,
                 min_rounds=300, switch_every=100, delta=0.003, gamma=0.3,
                 damp=0.2, censor=False, subsample=1, full_accs=False,
                 head_scale=1.4, online_window=16000, online_refit=200,
                 eta=3e-4, eta_head=1e-4, beta_sm=10.0):
        from collections import deque
        super().__init__(window=online_window, refit_every=online_refit,
                         head=head, tail="fixed", tail_scale=init_w)
        self.mode, self.adapt_window = mode, int(adapt_window)
        self.gamma, self.damp, self.censor = float(gamma), float(damp), bool(censor)
        self.subsample, self.full_accs = max(1, int(subsample)), bool(full_accs)
        self.min_rounds, self.switch_every = int(min_rounds), max(1, int(switch_every))
        self.delta, self.head_scale = float(delta), float(head_scale)
        self.eta, self.eta_head, self.beta_sm = float(eta), float(eta_head), float(beta_sm)
        if mode == "sgdmat":
            self.full_accs = True                # gradient needs the full acc vector
        if mode == "ftl2d":
            self.cands = [(float(u), float(w)) for u in ugrid for w in wgrid]
        elif mode == "ftl":
            u0 = self.head_scale if head == "scale" else None
            self.cands = [(u0, float(w)) for w in wgrid]
        elif mode == "sgdgain":                  # continuous (u, w), no grid
            self.cands = [(float(init_u), float(init_w))]
        else:                                    # ratio / succratio / sgdmat
            self.cands = [(self.head_scale if head == "scale" else None,
                           float(init_w))]
        self.leader = min(range(len(self.cands)),
                          key=lambda i: (abs(self.cands[i][1] - float(init_w)),
                                         abs((self.cands[i][0] if self.cands[i][0]
                                              is not None else init_u) - init_u)))
        u0, self.w_t = self.cands[self.leader]
        self.u_t = u0 if u0 is not None else 0.0
        self.wclamp = (0.01, 0.6)                # ratio/sgd safety clamp
        self.uclamp = (0.5, 3.0)                 # sgdgain head-scalar clamp
        self.wstep, self.ustep = 0.002, 0.005    # per-round SGD step clips
        self._pending_dw = 0.0                   # sgdmat: applied after trace
        n = len(self.cands)
        self._ew_decay = (0.5 ** (1.0 / ewma_halflife)) if ewma_halflife > 0 else None
        self._ew_num, self._ew_den = [0.0] * n, 0.0
        self._cdq = deque()                      # per-round credit vectors
        self._csums, self._ccnt = [0.0] * n, 0
        self._credited = self._n_rounds = self._hookn = 0
        self.switches = 0
        self.rx = deque(maxlen=self.adapt_window or None)   # ratio/succratio pairs
        self.ry = deque(maxlen=self.adapt_window or None)
        self.min_pairs = 100
        self.trace = []                          # (w_t, u_t, chosen_k, acc)/round

    @property
    def wants_counterfactuals(self):
        return (self.mode == "sgdmat"
                or (self.mode in ("ftl", "ftl2d") and len(self.cands) > 1))

    @property
    def wants_aux_succ(self):
        return self.mode == "succratio"

    def take_round(self):
        self._hookn += 1
        return (self._hookn % self.subsample) == 0

    def cal_h(self, c):
        if self.mode in ("ftl2d", "sgdgain") or self.head == "scale":
            return min(1.0, max(0.0, self.u_t * float(c)))
        if self.head == "raw":
            return float(c)
        return self._h(c)                        # inherited online beta/logistic

    def cal_t(self, v):
        return self.w_t * max(0.0, float(v))

    def k_candidates(self, conf, tail_scores):
        """argmax-k per grid candidate at this round's state — EXACT mirror of
        the acting controller loop (strict >, ties -> smaller k). Pure math on
        the cached conf + per-k budgeted tail scores; no suffix calls."""
        if self.mode == "sgdmat":
            return []                            # gradient uses the FULL acc vector
        W_ = len(tail_scores) - 1
        prefix_cache, ks = {}, []
        for (u, w) in self.cands:
            key = u if self.mode == "ftl2d" else None
            SG = prefix_cache.get(key)
            if SG is None:
                S, G, SG = 1.0, 0.0, [(1.0, 0.0)]
                for j in range(W_):
                    a = (min(1.0, max(0.0, u * float(conf[j])))
                         if self.mode == "ftl2d" else self.cal_h(conf[j]))
                    S *= a; G += S
                    SG.append((S, G))
                prefix_cache[key] = SG
            best_val, kc = 1.0 + w * max(0.0, float(tail_scores[0])), 0
            for j in range(1, W_ + 1):
                S, G = SG[j]
                val = 1.0 + G + S * w * max(0.0, float(tail_scores[j]))
                if val > best_val:
                    best_val, kc = val, j
            ks.append(kc)
        return ks

    def observe_grid(self, conf, tail_scores, k_of, accs_by_k):
        if self.mode == "sgdmat":
            # softmax-relaxed total-accept objective: V(w) = sum_k pi_k*acc_k,
            # pi = softmax(beta*val_k(w)), val_k = 1 + G_k + S_k*w*s_k (raw head).
            # dV/dw = beta * sum_k pi_k*acc_k*(dval_k - sum_l pi_l*dval_l),
            # dval_k/dw = S_k*s_k. Stashed and applied AFTER the trace append.
            import math
            W_ = len(tail_scores) - 1
            s = [max(0.0, float(x)) for x in tail_scores]
            S, G, vals, dvals = 1.0, 0.0, [], []
            for kk in range(W_ + 1):
                if kk > 0:
                    S *= self.cal_h(conf[kk - 1]); G += S
                vals.append(1.0 + G + S * self.w_t * s[kk])
                dvals.append(S * s[kk])
            vmax = max(vals)
            e = [math.exp(self.beta_sm * (v - vmax)) for v in vals]
            z = sum(e) or 1.0
            pi = [x / z for x in e]
            acc = [float(accs_by_k.get(kk, 0.0)) for kk in range(W_ + 1)]
            gbar = sum(p * d for p, d in zip(pi, dvals))
            grad = self.beta_sm * sum(p * a * (d - gbar)
                                      for p, a, d in zip(pi, acc, dvals))
            self._pending_dw = min(self.wstep, max(-self.wstep, self.eta * grad))
            return
        self._push_credits([float(accs_by_k[kc]) for kc in k_of])

    def _push_credits(self, credits):
        if self._ew_decay is not None:
            d = self._ew_decay
            self._ew_num = [d * a + c for a, c in zip(self._ew_num, credits)]
            self._ew_den = d * self._ew_den + 1.0
        elif self.adapt_window > 0:
            self._cdq.append(credits)
            self._csums = [a + c for a, c in zip(self._csums, credits)]
            if len(self._cdq) > self.adapt_window:
                old = self._cdq.popleft()
                self._csums = [a - c for a, c in zip(self._csums, old)]
        else:
            self._csums = [a + c for a, c in zip(self._csums, credits)]
            self._ccnt += 1
        self._credited += 1

    def _means(self):
        if self._ew_decay is not None:
            return ([x / self._ew_den for x in self._ew_num]
                    if self._ew_den > 0 else None)
        if self.adapt_window > 0:
            n = len(self._cdq)
            return [x / n for x in self._csums] if n else None
        return [x / self._ccnt for x in self._csums] if self._ccnt else None

    def _elect(self):
        means = self._means()
        if means is None:
            return
        ch = max(range(len(self.cands)), key=lambda i: means[i])
        if ch != self.leader and means[ch] > means[self.leader] + self.delta:
            self.leader, self.switches = ch, self.switches + 1
            u, self.w_t = self.cands[ch]
            if u is not None:
                self.u_t = u

    def _update_ratio(self):
        sx = sum(self.rx)
        if len(self.rx) < self.min_pairs or sx <= 0:
            return
        target = self.gamma * (sum(self.ry) / sx)
        w = (1.0 - self.damp) * self.w_t + self.damp * target
        self.w_t = min(self.wclamp[1], max(self.wclamp[0], w))

    def observe(self, confs, k, head_acc, tail_score, tail_acc, aux_succ=None):
        # act(round t, current leader) already happened; credits for ftl modes
        # arrived via observe_grid just before. All updates below only affect
        # round t+1 onward (no peeking).
        if self.head in ("beta", "logistic", "linear"):
            super().observe(confs, k, head_acc, tail_score, tail_acc)
        acc = (k + tail_acc) if tail_acc is not None else head_acc
        self.trace.append((round(float(self.w_t), 6), round(float(self.u_t), 4),
                           int(k), int(acc)))
        self._n_rounds += 1
        if self.mode == "sgdmat":
            self.w_t = min(self.wclamp[1], max(self.wclamp[0],
                                               self.w_t + self._pending_dw))
            self._pending_dw = 0.0
        elif self.mode == "sgdgain":
            # tail: L = 1/2 (w*s - tail_acc)^2, observed only when the head
            # survived to the chosen k (accept-conditioned, same as the model)
            if tail_acc is not None and tail_score > 0:
                g_t = (self.w_t * float(tail_score) - float(tail_acc)) * float(tail_score)
                step = min(self.wstep, max(-self.wstep, self.eta * g_t))
                self.w_t = min(self.wclamp[1], max(self.wclamp[0], self.w_t - step))
            # head: L = 1/2 (G_hat(u) - head_acc)^2 over the chosen prefix;
            # G_hat = sum_j S_j, S_j = prod_{i<j} min(1, u*conf_i);
            # dS_j/du = (S_j/u) * #{i<j: u*conf_i < 1}
            if k > 0:
                S, Ghat, dG, mlt = 1.0, 0.0, 0.0, 0
                for j in range(k):
                    a = min(1.0, max(0.0, self.u_t * float(confs[j])))
                    if self.u_t * float(confs[j]) < 1.0:
                        mlt += 1
                    S *= a
                    Ghat += S
                    dG += (S / self.u_t) * mlt
                g_h = (Ghat - float(head_acc)) * dG
                step = min(self.ustep, max(-self.ustep, self.eta_head * g_h))
                self.u_t = min(self.uclamp[1], max(self.uclamp[0], self.u_t - step))
        elif self.mode in ("ratio", "succratio"):
            if self.mode == "ratio":
                if tail_acc is not None and tail_score > 0:
                    self.rx.append(float(tail_score)); self.ry.append(float(tail_acc))
            elif aux_succ is not None and aux_succ[0] > 0:
                self.rx.append(float(aux_succ[0])); self.ry.append(float(aux_succ[1]))
            if self._n_rounds % self.switch_every == 0:
                self._update_ratio()
        elif (self._credited >= self.min_rounds
              and self._n_rounds % self.switch_every == 0):
            self._elect()


def _bestfirst_caps(rem, k, conf, tails):
    a = [min(1.0, max(0.0, 0.69 * float(conf[j]) + 0.29)) for j in range(min(k, len(conf)))]
    S = [1.0]
    for aj in a:
        S.append(S[-1] * aj)
    p = []
    for j in range(k + 1):
        Sj = S[j] if j < len(S) else S[-1]
        p.append(Sj * (1.0 - (a[j] if j < len(a) else 1.0)) if j < k else Sj)
    tot = sum(p) or 1.0
    caps = [min(len(tails[j]), int(rem * p[j] / tot)) for j in range(k + 1)]
    left = rem - sum(caps)
    order = sorted(range(k + 1), key=lambda j: -p[j])
    while left > 0:
        progress = False
        for j in order:
            if left > 0 and caps[j] < len(tails[j]):
                caps[j] += 1; left -= 1; progress = True
        if not progress:
            break
    return caps


def build_draft(proposer, block_full, suf, tails, k, num_spec, budget=None, alloc="even", conf=None):
    """Identical to run_partialwarm_tree.build_draft (torch-free copy)."""
    B = num_spec if budget is None else budget
    if proposer == "dflash":
        return build_extension_chain(block_full[:B], [])
    if proposer == "suffix":
        return build_extension_chain([], suf[:B])
    if proposer == "chain":
        tb = max(0, B - k)
        return build_extension_chain(block_full[:k], tails[k][:tb])
    rem = max(0, B - k)
    if alloc == "none":
        caps = [num_spec] * (k + 1)
    elif alloc == "bestfirst" and conf is not None:
        caps = _bestfirst_caps(rem, k, conf, tails)
    else:
        per = rem // (k + 1) if k + 1 else 0
        caps = [per] * (k + 1)
    return build_extension_tree(block_full[:k], [t[:c] for t, c in zip(tails, caps)])


_KACC = defaultdict(lambda: [0.0, 0])          # chosen-k audit (env KLOG)
_REGRET = []                                   # (chosen_k, best_k, acc_chosen, acc_best) if REGRET_AUDIT


def _walk_acc(block_full, tails, kk, num_spec, gt_rest):
    """Realized accept of the k=kk handoff chain against gt_rest. Pure function
    (no suffix/tree state touched) — safe for counterfactual evaluation."""
    tr = build_extension_chain(block_full[:kk], tails[kk][0][:max(0, num_spec - kk)])
    return len(greedy_tree_walk_path(list(tr.tokens), list(tr.parents), gt_rest))


def _klog(key, v):
    a = _KACC[key]; a[0] += float(v); a[1] += 1


def replay_proposer(records_by_rid, gt, prompt_ids, suffix, proposer, num_spec,
                    max_rounds, alloc, controller, calib=None):
    """Replay one proposer's trajectory along gt (teacher-forced greedy). Returns
    the list of per-round accept lengths K."""
    W = records_by_rid[1]["W"] if 1 in records_by_rid else next(iter(records_by_rid.values()))["W"]
    suffix.new_eval(prompt_ids)
    Ks = []
    m = 0                                          # committed gt output tokens
    for _ in range(max_rounds):
        pos = m + 1                                # record key: block rooted at output-idx m
        rec = records_by_rid.get(pos)
        if rec is None or m >= len(gt):
            break
        block_full = rec["dflash_tok"]
        conf = rec["dflash_conf"]
        root = gt[m]
        ctx_list = prompt_ids + gt[:m] + [root]
        suf, T = suffix.probe(ctx_list, num_spec)
        if proposer == "oracle":
            # DECISION ORACLE (slide 11): same machine (DFlash head + Suffix tail),
            # knows the TRUE realized accept, picks the head length k that maximizes
            # it. Linear/chain structure (the deployed composition's ceiling).
            best_acc, path, tree = -1, [], build_extension_chain([], [])
            best_kk_f, best_kk_l = 0, 0
            for kk in range(W + 1):
                tk = suffix.speculate(ctx_list + block_full[:kk], num_spec)
                tr = build_extension_chain(block_full[:kk], tk[:max(0, num_spec - kk)])
                pth = greedy_tree_walk_path(list(tr.tokens), list(tr.parents), gt[m + 1:])
                if len(pth) > best_acc:
                    best_acc, path, tree, best_kk_f, best_kk_l = len(pth), pth, tr, kk, kk
                elif len(pth) == best_acc:
                    best_kk_l = kk
            acc = best_acc
            _klog("oracle_first", best_kk_f); _klog("oracle_last", best_kk_l)
        elif proposer == "sel_oracle":
            # SELECTION oracle: ceiling of the binary select — a PERFECT choice between
            # its two options (k=0 pure suffix tail, k=W DFlash head + suffix tail).
            # Cannot use intermediate head lengths (that is the handoff 'oracle').
            Wk = min(num_spec, len(block_full))
            best_acc, path, tree = 0, [], build_extension_chain([], [])
            for kk in (0, Wk):
                tk = suffix.speculate(ctx_list + block_full[:kk], num_spec)
                tr = build_extension_chain(block_full[:kk], tk[:max(0, num_spec - kk)])
                pth = greedy_tree_walk_path(list(tr.tokens), list(tr.parents), gt[m + 1:])
                if len(pth) > best_acc:
                    best_acc, path, tree = len(pth), pth, tr
            acc = best_acc
        elif proposer == "select":
            # PER-DEPTH competition (raw): walk DFlash block depths, extend the head while
            # its per-depth hit-prob a_d = 0.69*conf+0.29 beats a*=T/(1+T); hand off at the
            # FIRST crossing, then attach the suffix tail. (DFlash's block is a fixed
            # autoregressive chain, so per-depth competition == choosing the handoff point.)
            astar = T / (1.0 + T) if (1.0 + T) > 0 else 0.0
            k = 0
            for j in range(min(num_spec, len(conf))):
                if min(1.0, max(0.0, 0.69 * float(conf[j]) + 0.29)) > astar:
                    k = j + 1
                else:
                    break
            tk = suffix.speculate(ctx_list + block_full[:k], num_spec)
            tree = build_extension_chain(block_full[:k], tk[:max(0, num_spec - k)])
            path = greedy_tree_walk_path(list(tree.tokens), list(tree.parents), gt[m + 1:])
            acc = len(path)
        elif proposer == "select_calib":
            # PER-DEPTH competition with CALIBRATED signals, SAME first-crossing
            # controller as `select` (clean ablation: controller fixed, signals
            # swapped): extend the head while cal_h(conf_j) > Tc/(1+Tc), where
            # cal_h = logistic P(match|conf) and Tc = cal_t(probe score) is the
            # isotonic-corrected tail expectation at the current position (the
            # raw arm uses the affine hazard and the RAW arctic score there).
            if calib is not None:
                cal_h, cal_t = calib
            else:
                cal_h = lambda c: min(1.0, max(0.0, 0.69 * float(c) + 0.29))  # noqa: E731
                cal_t = float                                                  # noqa: E731
            Tc = cal_t(T)
            astar = Tc / (1.0 + Tc) if (1.0 + Tc) > 0 else 0.0
            k = 0
            for j in range(min(num_spec, len(conf))):
                if cal_h(conf[j]) > astar:
                    k = j + 1
                else:
                    break
            tk = suffix.speculate(ctx_list + block_full[:k], num_spec)
            tree = build_extension_chain(block_full[:k], tk[:max(0, num_spec - k)])
            path = greedy_tree_walk_path(list(tree.tokens), list(tree.parents), gt[m + 1:])
            acc = len(path)
        elif proposer == "calib":
            # PER-DEPTH competition, BOTH sides calibrated: argmax_k [1 + G_k + S_k*T_k]
            # (composition's own objective) where
            #   a_j = cal_h(conf[j])  — DFlash hazard, logistic P(match|conf),
            #                            accept-conditioned, fit on the calibrate split;
            #   T_k = cal_t(score_k)  — PER-CANDIDATE-k suffix tail expectation: the arctic
            #                            score of the BUDGETED tail at ctx+head[:k], mapped
            #                            through isotonic score->E[realized accept].
            # The scalar-T version priced every handoff with the k=0 probe; per-k T_k lets
            # the controller see a hot (or dead) tail exactly where it would graft it, and
            # the isotonic map fixes arctic's 3-6x score over-estimate. Tail always grafted.
            if isinstance(calib, OnlineCalib):
                cal_h, cal_t = calib.cal_h, calib.cal_t
            elif calib is not None:
                cal_h, cal_t = calib
            else:
                cal_h = lambda c: min(1.0, max(0.0, 0.69 * float(c) + 0.29))  # noqa: E731
                cal_t = float                                                  # noqa: E731
            W_ = min(num_spec, len(conf))
            want_aux = getattr(calib, "wants_aux_succ", False)
            tails, auxs = [], ([] if want_aux else None)
            for kk in range(W_ + 1):
                budget = num_spec - kk
                # dfprior: hand the tail scorer DFlash's remaining block tokens+confs
                # (positions kk, kk+1, ... aligned to tail edges 0,1,...) as the prior
                tails.append(suffix._spec(ctx_list + block_full[:kk], budget,
                                          df_tok=block_full[kk:], df_conf=conf[kk:])
                             if budget > 0 else ([], 0.0))
                if want_aux:
                    auxs.append(getattr(suffix, "last_aux_succ", None)
                                if budget > 0 else None)
            S_k, G_k = 1.0, 0.0
            best_val, k = 1.0 + cal_t(tails[0][1]), 0     # k=0: 1 + 0 + 1*T_0
            for j in range(W_):
                S_k *= cal_h(conf[j]); G_k += S_k
                val = 1.0 + G_k + S_k * cal_t(tails[j + 1][1])
                if val > best_val:
                    best_val, k = val, j + 1
            tree = build_extension_chain(block_full[:k], tails[k][0][:max(0, num_spec - k)])
            path = greedy_tree_walk_path(list(tree.tokens), list(tree.parents), gt[m + 1:])
            acc = len(path)
            _klog("calib_online" if isinstance(calib, OnlineCalib) else "calib_k", k)
            if os.environ.get("REGRET_AUDIT"):
                # realized-optimal handoff k at THIS state (same machine, tries every k):
                # log chosen vs best so we can see if the controller hands off too EARLY
                # (chosen<best) or too LATE at a given tail price. acc = realized@chosen.
                best_r, best_kk = -1, 0
                for kk in range(W_ + 1):
                    p_len = _walk_acc(block_full, tails, kk, num_spec, gt[m + 1:])
                    if p_len > best_r:
                        best_r, best_kk = p_len, kk
                _REGRET.append((int(k), int(best_kk), int(acc), int(best_r)))
            if getattr(calib, "wants_counterfactuals", False) and calib.take_round():
                # adaptive FTL/sgdmat: credit every grid candidate with the
                # realized accept of ITS argmax-k at this state (sgdmat uses
                # the FULL per-k vector for its softmax gradient). censor clips
                # labels to what the verifier reveals (accepted + bonus).
                tail_s = [t[1] for t in tails]
                k_of = calib.k_candidates(conf, tail_s)
                gt_rest = gt[m + 1: m + 2 + acc] if calib.censor else gt[m + 1:]
                accs_by_k = {k: acc}                 # chosen k: fully observed
                kws = range(W_ + 1) if calib.full_accs else set(k_of)
                for kk in kws:
                    if kk not in accs_by_k:
                        accs_by_k[kk] = _walk_acc(block_full, tails, kk,
                                                  num_spec, gt_rest)
                calib.observe_grid(conf, tail_s, k_of, accs_by_k)
            if isinstance(calib, OnlineCalib):
                # verify-time labels the deployment observes for free
                head_acc = 0
                for j in range(k):
                    if m + 1 + j < len(gt) and block_full[j] == gt[m + 1 + j]:
                        head_acc += 1
                    else:
                        break
                tail_acc = (acc - k) if head_acc == k else None
                calib.observe(conf, k, head_acc, tails[k][1], tail_acc,
                              aux_succ=(auxs[k] if auxs is not None else None))
        elif proposer == "hedge":
            # HEDGED per-depth select: same calibrated signals & head-length argmax as
            # `calib`, but instead of committing the whole tail budget to the single
            # chosen handoff point, split the remaining budget across tails at EVERY
            # head depth j<=k, sized by P(head stops at j) * E[tail_j accept] (both
            # calibrated). Insurance against an early head miss at the same total node
            # budget; every tree path is a truncated chain option, so the chain handoff
            # oracle is still the ceiling.
            if calib is not None:
                cal_h, cal_t = calib
            else:
                cal_h = lambda c: min(1.0, max(0.0, 0.69 * float(c) + 0.29))  # noqa: E731
                cal_t = float                                                  # noqa: E731
            W_ = min(num_spec, len(conf))
            tails = []
            for kk in range(W_ + 1):
                budget = num_spec - kk
                tails.append(suffix._spec(ctx_list + block_full[:kk], budget)
                             if budget > 0 else ([], 0.0))
            S_k, G_k = 1.0, 0.0
            best_val, k = 1.0 + cal_t(tails[0][1]), 0
            for j in range(W_):
                S_k *= cal_h(conf[j]); G_k += S_k
                val = 1.0 + G_k + S_k * cal_t(tails[j + 1][1])
                if val > best_val:
                    best_val, k = val, j + 1
            a = [cal_h(conf[j]) for j in range(k)]
            S = [1.0]
            for aj in a:
                S.append(S[-1] * aj)
            rem = max(0, num_spec - k)
            w = []
            for j in range(k + 1):
                pj = S[j] * ((1.0 - a[j]) if j < k else 1.0)
                w.append(pj * max(cal_t(tails[j][1]), 1e-9))
            tot = sum(w) or 1.0
            caps = [min(len(tails[j][0]), int(rem * w[j] / tot)) for j in range(k + 1)]
            left = rem - sum(caps)
            order = sorted(range(k + 1), key=lambda j: -w[j])
            while left > 0:
                prog = False
                for j in order:
                    if left > 0 and caps[j] < len(tails[j][0]):
                        caps[j] += 1; left -= 1; prog = True
                if not prog:
                    break
            tree = build_extension_tree(block_full[:k], [t[0][:c] for t, c in zip(tails, caps)])
            path = greedy_tree_walk_path(list(tree.tokens), list(tree.parents), gt[m + 1:])
            acc = len(path)
        else:
            if controller == "first_crossing":
                astar = T / (1.0 + T)
                k = 0
                for j in range(len(conf)):
                    aj = min(1.0, max(0.0, 0.69 * conf[j] + 0.29))
                    if aj > astar:
                        k = j + 1
                    else:
                        break
            else:                                   # argmax (deployed)
                k = adaptive_nhead(conf, T=T, num_spec=W)
            tails = [suffix.speculate(ctx_list + block_full[:j], num_spec) for j in range(k + 1)]
            tree = build_draft(proposer, block_full, suf, tails, k, num_spec, None, alloc, conf)
            path = greedy_tree_walk_path(list(tree.tokens), list(tree.parents), gt[m + 1:])
            acc = len(path)
        Ks.append(acc)
        # advance: root + accepted + bonus (all gt tokens)
        accepted_toks = [tree.tokens[i] for i in path]
        nxt = [root] + accepted_toks
        bonus_idx = m + 1 + acc
        if bonus_idx < len(gt):
            nxt.append(gt[bonus_idx])
        suffix.add_response(nxt)
        m += 1 + acc + 1
        if m >= len(gt):
            break
    return Ks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True, help="capture_perpos jsonl")
    ap.add_argument("--props", nargs="+", default=["dflash", "suffix", "chain", "tree"])
    ap.add_argument("--alloc", default="even", choices=["even", "bestfirst", "none"])
    ap.add_argument("--controller", default="argmax", choices=["argmax", "first_crossing"])
    ap.add_argument("--max-rounds", type=int, default=int(os.environ.get("ROUNDS", "8")))
    ap.add_argument("--hazard-fit", default="logistic", choices=["logistic", "beta"],
                    help="calib arm's head-hazard calibrator: plain logistic on conf "
                         "or beta calibration (logistic on ln/ln(1-) features)")
    ap.add_argument("--raw-cals", action="store_true",
                    help="calib arm with IDENTITY calibrators (cal_h=cal_t=identity): "
                         "the compose controller on raw DFlash conf + raw arctic score "
                         "(no beta hazard / isotonic tail, no affine). Default arm.")
    ap.add_argument("--head-cal", default="", choices=["", "raw", "affine", "linear", "logistic", "beta", "card", "scale"],
                    help="general head-hazard calibrator for the calib arm (overrides "
                         "the boolean flags). Pair with --tail-cal. Needs --calib-insample. "
                         "card = LABEL-FREE agreement card (--head-card json {x,y}): "
                         "a(conf)=P(agree with tree)/b_succession — no accept labels, "
                         "no workload fit; piecewise-linear interp, clipped to [0,1].")
    ap.add_argument("--head-card", default="",
                    help="json file with the label-free head card for --head-cal card")
    ap.add_argument("--head-scale", type=float, default=1.15,
                    help="fixed multiplicative head weight for --head-cal scale: "
                         "a_j = min(1, u*conf_j) — the head twin of the tail's w*s")
    ap.add_argument("--online-calib", action="store_true",
                    help="ONLINE windowed calibration arm: logistic head + isotonic "
                         "tail refit on a sliding window of verify-time labels from "
                         "the eval stream itself (no offline fit, no calib split; "
                         "identity until warm). Overrides other calib flags.")
    ap.add_argument("--online-window", type=int, default=16000,
                    help="sliding-window size (head pairs; tail window = /4)")
    ap.add_argument("--online-refit", type=int, default=200,
                    help="refit the calibrators every N rounds")
    ap.add_argument("--online-head", default="logistic",
                    choices=["logistic", "beta", "linear", "raw"],
                    help="online head calibrator family. 'raw' keeps the head map at "
                         "identity (no online refit) — for the tail-only online arm.")
    ap.add_argument("--online-tail", default="isotonic",
                    choices=["isotonic", "linear", "raw", "fixed"],
                    help="online tail calibrator family. 'raw' keeps the tail map at "
                         "identity (no online refit) — pair with --tail-cal succession/kt "
                         "so the zero-parameter Laplace/KT rescore IS the tail estimate. "
                         "'fixed' freezes E[acc|s]=w*s with w=--online-tail-scale (a single "
                         "deploy-time tail scalar under the online head; no refit).")
    ap.add_argument("--online-tail-scale", type=float, default=0.0075,
                    help="scalar w for --online-tail fixed (E[acc|s] = w*rawscore)")
    ap.add_argument("--adaptive", default="",
                    choices=["", "ftl", "ftl2d", "ratio", "succratio",
                             "sgdgain", "sgdmat"],
                    help="FEEDBACK-LOOP compose scalars (AdaptiveScalar): the "
                         "fixed w (and u for ftl2d) becomes an online update. "
                         "ftl/ftl2d = counterfactual follow-the-leader over "
                         "--adaptive-grid (x --adaptive-ugrid); ratio = "
                         "gamma * windowed E[tail_acc]/E[tail_score]; succratio "
                         "= LABEL-FREE gamma * E[succ]/E[raw] of chosen tails; "
                         "sgdgain = per-round SGD on the EXPLICIT per-side "
                         "gain-residual loss (head (G_hat(u)-head_acc)^2, tail "
                         "(w*s-tail_acc)^2, both continuous); sgdmat = per-round "
                         "gradient ascent on the softmax-relaxed TOTAL-ACCEPT "
                         "objective V(w)=sum_k softmax(beta*val_k)*acc_k. "
                         "Requires --props calib only. Composes with --tail-cal "
                         "gbscale (score_mode wiring is independent).")
    ap.add_argument("--adaptive-head", default="raw",
                    choices=["raw", "scale", "beta", "logistic"],
                    help="head map under adaptation: raw=identity, scale=fixed "
                         "u via --head-scale, beta/logistic=inherited online "
                         "head refit (--online-window/--online-refit)")
    ap.add_argument("--adaptive-grid",
                    default="0.02,0.03,0.04,0.05,0.0625,0.075,0.09,0.11,0.14,0.18",
                    help="w candidates for ftl/ftl2d (comma floats)")
    ap.add_argument("--adaptive-ugrid", default="1.0,1.2,1.4,1.7,2.0",
                    help="u candidates for ftl2d (comma floats)")
    ap.add_argument("--adaptive-window", type=int, default=8000,
                    help="credit window in credited ROUNDS (ftl) / pairs "
                         "(ratio); 0 = infinite (pure FTL on all history)")
    ap.add_argument("--adaptive-ewma-halflife", type=int, default=0,
                    help=">0 replaces the sliding window with EWMA credits "
                         "of this half-life (rounds)")
    ap.add_argument("--adaptive-init", type=float, default=0.075,
                    help="cold-start acting w (leader starts at the nearest "
                         "grid candidate)")
    ap.add_argument("--adaptive-init-u", type=float, default=1.4,
                    help="cold-start acting u (ftl2d leader tie-break)")
    ap.add_argument("--adaptive-min-rounds", type=int, default=300,
                    help="credited rounds before the first leader switch")
    ap.add_argument("--adaptive-switch-every", type=int, default=100,
                    help="leader re-election / ratio-update cadence (rounds)")
    ap.add_argument("--adaptive-delta", type=float, default=0.003,
                    help="hysteresis: challenger must beat the incumbent's "
                         "windowed mean by this margin (tokens/round)")
    ap.add_argument("--adaptive-gamma", type=float, default=0.3,
                    help="shading factor for ratio (~0.2-0.3) / succratio "
                         "(~0.6 in the u=1.4 head frame)")
    ap.add_argument("--adaptive-damp", type=float, default=0.2,
                    help="damping eta for the ratio update "
                         "w <- (1-eta)*w + eta*gamma*rho")
    ap.add_argument("--adaptive-censor", action="store_true",
                    help="ftl: clip counterfactual credits to the verifier-"
                         "revealed prefix min(acc_k, acc_chosen+1) — the "
                         "DEPLOYABLE-information tier")
    ap.add_argument("--adaptive-subsample", type=int, default=1,
                    help="credit every Nth round only (compute knob)")
    ap.add_argument("--adaptive-full-accs", action="store_true",
                    help="walk ALL k in 0..W for credits (diagnostics), not "
                         "just the candidates' distinct argmax-ks")
    ap.add_argument("--adaptive-trace", default="",
                    help="json dump path for the per-round (w,u,k,acc) trace "
                         "+ rid/task boundaries (figures)")
    ap.add_argument("--adaptive-eta", type=float, default=3e-4,
                    help="SGD learning rate: tail step for sgdgain, gradient-"
                         "ascent step for sgdmat (per-round step clipped)")
    ap.add_argument("--adaptive-eta-head", type=float, default=1e-4,
                    help="sgdgain: head-scalar u learning rate")
    ap.add_argument("--adaptive-beta", type=float, default=10.0,
                    help="sgdmat: softmax inverse temperature of the relaxed "
                         "argmax (higher = closer to the hard controller)")
    ap.add_argument("--tail-cal", default="", choices=["", "raw", "scaled", "linear", "isotonic", "fixed",
                                                       "succession", "kt", "succscale", "ktscale", "dfprior", "genbeta", "wsplit", "gbscale"],
                    help="general tail calibrator (score->E[accept]): raw=identity, "
                         "scaled=global shrink, linear=affine fit, isotonic=deployed, "
                         "fixed=constant scalar --tail-scale (NO fitting: the copy/novel "
                         "mixture-model arm, w = P(copy|match) measured once), "
                         "succession/kt=ZERO-parameter per-edge rescore inside the tree "
                         "((k+1)/(n+2) Laplace or (k+.5)/(n+1) KT posterior instead of "
                         "the ML k/n — pure finite-sample math, no fit, no constant).")
    ap.add_argument("--tail-scale", type=float, default=0.25,
                    help="the fixed scalar w for --tail-cal fixed (E[acc|s] = w*s)")
    ap.add_argument("--df-s", type=float, default=4.0,
                    help="--tail-cal dfprior: DFlash prior concentration (pseudo-count "
                         "strength s in (k+q*s)/(n+s); q=0.5,s=2 recovers Laplace)")
    ap.add_argument("--df-u", type=float, default=1.4,
                    help="--tail-cal dfprior: head-conf calibration q_agree=min(1,u*conf)")
    ap.add_argument("--df-qdis", type=float, default=0.0,
                    help="--tail-cal dfprior: prior for the suffix token when DFlash "
                         "disagrees (0 = full veto)")
    ap.add_argument("--gb-a", type=float, default=1.0,
                    help="--tail-cal genbeta: per-edge posterior (k+a)/(n+b). a=1,b=2=Laplace")
    ap.add_argument("--gb-b", type=float, default=2.0,
                    help="--tail-cal genbeta: denominator offset b (prior mean a/b, strength b)")
    ap.add_argument("--ws-w0", type=float, default=1.0,
                    help="--tail-cal wsplit: weight on the depth-0 survival term "
                         "(score = w0*S_0 + w1*sum_{d>=1} S_d; w0=w1 = flat scalar)")
    ap.add_argument("--ws-w1", type=float, default=1.0,
                    help="--tail-cal wsplit: weight on the depth>=1 survival terms")
    ap.add_argument("--beta-head-only", action="store_true",
                    help="calib arm: beta head hazard + IDENTITY tail (isolate the "
                         "head calibrator's effect). Needs --calib-insample.")
    ap.add_argument("--iso-tail-only", action="store_true",
                    help="calib arm: IDENTITY head + isotonic tail (isolate the tail "
                         "calibrator's effect). Needs --calib-insample.")
    ap.add_argument("--affine-cals", action="store_true",
                    help="calib arm with Dr.Lee's ORIGINAL affine head hazard "
                         "(cal_h = 0.69*conf+0.29) and identity tail (raw arctic "
                         "score). No beta, no isotonic.")
    ap.add_argument("--calib-insample", action="store_true",
                    help="fit the calibrators on the FULL eval set and test on the "
                         "SAME set (in-sample calibration; user protocol: 1) tree = "
                         "warm set, 2) calibrate = eval set, 3) test = eval set). "
                         "Mutually exclusive with --three-way.")
    ap.add_argument("--three-way", action="store_true",
                    help="NO LOO. Disjoint 3-way split: tree=warm (fills suffix tree), "
                         "calibrate=even-indexed groups, test=odd-indexed. ALL proposers "
                         "evaluated ONLY on the test groups; the calibrator is fit ONCE on "
                         "the calibrate groups.")
    ap.add_argument("--eval-half", default="test", choices=["test", "calib"],
                    help="--three-way: which half the proposers are evaluated on. "
                         "'calib' lets a universal constant be tuned on the calibrate "
                         "half (decision-level fit) before the final test-half run.")
    ap.add_argument("--group-mode", default="rid",
                    choices=["rid", "lenreset", "conv", "convlabel"],
                    help="split unit for --three-way: 'rid' (each eval request its own "
                         "group — multislot single-turn prompts), 'lenreset' (multi-turn "
                         "conversations inferred from prompt-length resets — bfcl), "
                         "'conv' (EXACT conversation ids stored by capture_traj — full-"
                         "trajectory collections), or 'convlabel' (conv ids, but the "
                         "calibrate/test parity uses the conv's rank WITHIN its task "
                         "label — for subtask-interleaved datasets like specbench where "
                         "global parity aliases whole labels onto one side).")
    args = ap.parse_args()

    traces = json.load(open(Path(args.record).with_suffix(".traces.json")))
    warm_traces = traces["warm_traces"]
    eval_traces = {t["rid"]: t for t in traces["eval_traces"]}
    num_spec = traces.get("num_spec", 32)

    # records grouped by (rid) -> {pos: rec}
    recs = defaultdict(dict)
    task_of = {}
    for l in open(args.record):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        recs[r["rid"]][r["pos"]] = r
        task_of[r["rid"]] = r["task"]

    print(f"replay: {len(recs)} eval reqs, {len(warm_traces)} warm traces, "
          f"num_spec={num_spec}, controller={args.controller}, alloc={args.alloc}")

    # LOO calibrators for the "calib" selector: per position (dense capture) the
    # DFlash estimate G_d + realized accept, and Suffix warmth T + realized warm
    # accept; fit isotonic signal->E[accept] on OTHER prompts, apply to held-out.
    # group id per rid. rid mode: each eval request is its own group (single-turn
    # multislot). lenreset mode: multi-turn conversations inferred from prompt-length
    # resets (a call whose prompt is shorter than the previous rid's starts a new conv).
    conv_of, split_par = split_parity(eval_traces, args.group_mode)

    # 3-way disjoint split (NO LOO): calibrate = even-indexed convs, test = odd.
    test_rids = None
    if args.three_way:
        test_rids = {rid for rid in recs if split_par.get(rid, 0) == 1}
        calib_rids = {rid for rid in recs if split_par.get(rid, 0) == 0}
        if args.eval_half == "calib":
            test_rids = calib_rids       # tune-on-calib pass: replay the calib half
        print(f"3-way: calibrate on {len(calib_rids)} calls "
              f"({len({conv_of[r] for r in calib_rids})} convs), eval[{args.eval_half}] on "
              f"{len(test_rids)} calls "
              f"({len({conv_of[r] for r in test_rids})} convs) — disjoint, no LOO")

    calibrators = {}
    ad = None
    ad_rid_index = []
    if args.adaptive and "calib" in args.props:
        if set(args.props) != {"calib"}:
            ap.error("--adaptive requires --props calib only (hedge/select_calib "
                     "unpack tuple calibrators)")
        wg = [float(x) for x in args.adaptive_grid.split(",") if x.strip()]
        ug = [float(x) for x in args.adaptive_ugrid.split(",") if x.strip()]
        ad = AdaptiveScalar(mode=args.adaptive, head=args.adaptive_head,
                            wgrid=wg, ugrid=ug,
                            adapt_window=args.adaptive_window,
                            ewma_halflife=args.adaptive_ewma_halflife,
                            init_w=args.adaptive_init, init_u=args.adaptive_init_u,
                            min_rounds=args.adaptive_min_rounds,
                            switch_every=args.adaptive_switch_every,
                            delta=args.adaptive_delta, gamma=args.adaptive_gamma,
                            damp=args.adaptive_damp, censor=args.adaptive_censor,
                            subsample=args.adaptive_subsample,
                            full_accs=args.adaptive_full_accs,
                            head_scale=args.head_scale,
                            online_window=args.online_window,
                            online_refit=args.online_refit,
                            eta=args.adaptive_eta, eta_head=args.adaptive_eta_head,
                            beta_sm=args.adaptive_beta)
        calibrators = {rid: ad for rid in recs}      # ONE stream across the eval set
        print(f"adaptive-scalar: mode={args.adaptive} head={args.adaptive_head} "
              f"cands={len(ad.cands)} window={args.adaptive_window}r "
              f"ewma_hl={args.adaptive_ewma_halflife} init_w={args.adaptive_init} "
              f"censor={args.adaptive_censor} gamma={args.adaptive_gamma} "
              f"switch_every={args.adaptive_switch_every} delta={args.adaptive_delta}")
    elif args.online_calib and "calib" in args.props:
        oc = OnlineCalib(window=args.online_window, refit_every=args.online_refit,
                         head=args.online_head, tail=args.online_tail,
                         tail_scale=args.online_tail_scale)
        calibrators = {rid: oc for rid in recs}      # ONE stream across the eval set
        print(f"online-calib: {args.online_head} head + {args.online_tail} tail, "
              f"window={args.online_window} pairs, refit every {args.online_refit} "
              f"rounds, cold-start identity")
    elif (args.head_cal or args.tail_cal) and {"calib", "hedge", "select_calib"} & set(args.props):
        idc = lambda x: float(x)  # noqa: E731
        if not args.three_way:
            calib_rids = set(recs)      # in-sample: fit on the full eval set
        hc, tc = args.head_cal or "raw", args.tail_cal or "raw"
        # head hazard
        if hc == "raw":
            cal_h = idc
        elif hc == "scale":
            cal_h = lambda c, _u=args.head_scale: min(1.0, max(0.0, _u * float(c)))  # noqa: E731
        elif hc == "card":
            import numpy as _np
            _cd = json.load(open(args.head_card))
            _cx, _cy = _np.asarray(_cd["x"], float), _np.asarray(_cd["y"], float)
            cal_h = lambda c, _x=_cx, _y=_cy: float(  # noqa: E731
                min(1.0, max(0.0, _np.interp(float(c), _x, _y))))
        elif hc == "affine":
            cal_h = lambda c: min(1.0, max(0.0, 0.69 * float(c) + 0.29))  # noqa: E731
        else:
            hp = [(float(cf[d]), int(mt[d]))
                  for rid in sorted(calib_rids) for r in recs[rid].values()
                  for cf, mt in [(r["dflash_conf"], r["dflash_match"])]
                  for d in range(min(_ad(mt) + 1, len(cf)))]
            if hc == "linear":                       # FITTED affine a*conf+b (not fixed)
                import numpy as _np
                a, b = _np.polyfit(_np.asarray([x for x, _ in hp], float),
                                   _np.asarray([y for _, y in hp], float), 1)
                cal_h = lambda c, _a=a, _b=b: min(1.0, max(0.0, float(_a) * float(c) + float(_b)))  # noqa: E731
            else:
                fit_h = _fit_beta if hc == "beta" else _fit_logistic
                cal_h = fit_h([x for x, _ in hp], [y for _, y in hp])
        # tail (score -> E[accept]); collect pairs once for scaled/linear/isotonic
        if tc == "raw":
            cal_t = idc
        elif tc in ("succession", "kt", "dfprior", "genbeta", "wsplit"):
            cal_t = idc      # rescoring happens inside ArcticSuffix (score_mode)
        elif tc in ("succscale", "ktscale", "gbscale"):
            # Laplace/KT/genbeta rescore (inside ArcticSuffix) x fixed scalar w
            # (gbscale --gb-a 0: per-edge k/(n+b) evidence deflation + outer w;
            #  b=0 recovers the flat scalar champion)
            cal_t = lambda v, _w=args.tail_scale: _w * max(0.0, float(v))  # noqa: E731
        elif tc == "fixed":
            # FIXED proposer constant, no fitting anywhere: under the round-level
            # copy/novel mixture (copy w.p. rho -> E[acc|copy]=s; novel -> acc~0),
            # E[acc|s] = rho*s, so a constant scalar is the Bayes-optimal form.
            cal_t = lambda v, _w=args.tail_scale: _w * max(0.0, float(v))  # noqa: E731
        else:
            fn, txs, tys = _fit_tail_iso(warm_traces, recs, eval_traces, calib_rids,
                                         num_spec, args.max_rounds, return_pairs=True)
            if tc == "isotonic":
                cal_t = fn
            elif tc == "scaled":
                ratio = (sum(tys) / sum(txs)) if sum(txs) > 0 else 1.0
                cal_t = lambda v, _r=ratio: _r * max(0.0, float(v))  # noqa: E731
            elif tc == "linear":
                import numpy as _np
                a, b = _np.polyfit(_np.asarray(txs, float), _np.asarray(tys, float), 1)
                cal_t = lambda v, _a=a, _b=b: max(0.0, float(_a) * float(v) + float(_b))  # noqa: E731
        calibrators = {rid: (cal_h, cal_t) for rid in recs}
        print(f"head-cal={hc} tail-cal={tc} — fit on {len(calib_rids)} calls "
              + ("(three-way calibrate split)" if args.three_way else "(in-sample)"))
    elif args.raw_cals and {"calib", "hedge", "select_calib"} & set(args.props):
        ident = (lambda c: float(c), lambda t: float(t))
        calibrators = {rid: ident for rid in recs}
        print("raw-cals: identity calibrators (raw conf + raw arctic score)")
    elif args.affine_cals and {"calib", "hedge", "select_calib"} & set(args.props):
        aff = (lambda c: min(1.0, max(0.0, 0.69 * float(c) + 0.29)), lambda t: float(t))
        calibrators = {rid: aff for rid in recs}
        print("affine-cals: Dr.Lee affine head (0.69*conf+0.29) + identity tail")
    elif {"calib", "hedge", "select_calib"} & set(args.props):
        # PER-DEPTH hazard calibrator: P(block matches gt at depth d | on gt through d-1)
        # as a function of conf[d]. Accept-conditioned — for each round include depths
        # 0..a_d (a_d = leading match run): (conf[d], match[d]); the label at d=a_d is 0
        # (first miss). Replaces the fixed affine hazard 0.69*conf+0.29 in the controller.
        pairs = {}
        for rid, rby in recs.items():
            pp = []
            for r in rby.values():
                conf, match = r["dflash_conf"], r["dflash_match"]
                ad = _ad(match)
                for d in range(min(ad + 1, len(conf))):
                    pp.append((float(conf[d]), int(match[d])))
            pairs[rid] = pp
        if args.calib_insample or args.three_way:
            if args.calib_insample:
                calib_rids = set(recs)          # fit on the FULL eval set (in-sample)
                print(f"calib-insample: fit on ALL {len(calib_rids)} eval calls, "
                      f"test on the same set")
            tr = [p for rid in calib_rids for p in pairs[rid]]   # fit ONCE
            fit_h = _fit_beta if args.hazard_fit == "beta" else _fit_logistic
            idc = lambda x: float(x)  # noqa: E731
            # head: beta/logistic hazard, unless --iso-tail-only (identity head)
            cal_hazard = idc if args.iso_tail_only else fit_h(
                [x for x, _ in tr], [y for _, y in tr])
            # tail: isotonic (skip the expensive fit for --beta-head-only)
            cal_tail = idc if args.beta_head_only else _fit_tail_iso(
                warm_traces, recs, eval_traces, calib_rids, num_spec, args.max_rounds)
            if args.beta_head_only:
                print("beta-head-only: beta hazard + identity tail")
            elif args.iso_tail_only:
                print("iso-tail-only: identity head + isotonic tail")
            calibrators = {rid: (cal_hazard, cal_tail) for rid in recs}
        else:
            ap.error("calib requires --three-way or --calib-insample (LOO is disabled)")

    # K per proposer, aggregated over rounds across all eval reqs (matches run_partialwarm)
    per_prop = {p: [] for p in args.props}
    per_prop_task = {p: defaultdict(list) for p in args.props}
    for p in args.props:
        suffix = ArcticSuffix(); suffix.fit(warm_traces)     # fresh warmed tree per proposer
        if args.tail_cal in ("succession", "kt", "succscale", "ktscale", "dfprior", "genbeta", "wsplit", "gbscale") and p in ("calib", "hedge", "select_calib"):
            suffix.score_mode = ("dfprior" if args.tail_cal == "dfprior"
                                 else "genbeta" if args.tail_cal in ("genbeta", "gbscale")
                                 else "wsplit" if args.tail_cal == "wsplit"
                                 else "kt" if args.tail_cal in ("kt", "ktscale") else "succ")
            suffix.df_s, suffix.df_u, suffix.df_qdis = args.df_s, args.df_u, args.df_qdis
            suffix.gb_a, suffix.gb_b = args.gb_a, args.gb_b
            suffix.ws_w0, suffix.ws_w1 = args.ws_w0, args.ws_w1
        if ad is not None and p == "calib" and ad.wants_aux_succ:
            suffix.aux_succ = True                           # label-free succ pairs
        for rid, rby in recs.items():
            if test_rids is not None and rid not in test_rids:
                continue                                     # 3-way: eval only on test convs
            tr = eval_traces.get(rid)
            if tr is None:
                continue
            if ad is not None and p == "calib":
                ad_rid_index.append([int(rid), task_of.get(rid, ""), len(ad.trace)])
            Ks = replay_proposer(rby, tr["output_ids"], tr["prompt_ids"], suffix, p,
                                  num_spec, args.max_rounds, args.alloc, args.controller,
                                  calib=calibrators.get(rid))
            per_prop[p].extend(Ks)
            per_prop_task[p][task_of[rid]].extend(Ks)

    print("\n== crossover K (mean accepted / round) ==")
    for p in args.props:
        ks = per_prop[p]
        mK = sum(ks) / len(ks) if ks else 0.0
        print(f"  {p:>8}: K={mK:.2f}  (rounds={len(ks)})")
    tasks = sorted({t for p in args.props for t in per_prop_task[p]})
    if len(tasks) > 1:
        print("\n== by task ==")
        for t in tasks:
            row = "  ".join(
                f"{p}={(sum(ks) / len(ks) if ks else 0):.2f}({len(ks)})"
                for p in args.props for ks in [per_prop_task[p][t]])
            print(f"  [{t}] {row}")

    klog_path = os.environ.get("KLOG")
    if klog_path and _KACC:
        json.dump({k: [v[0], v[1]] for k, v in _KACC.items()}, open(klog_path, "w"))
        print("\n== chosen-k audit ==")
        for k, v in _KACC.items():
            print(f"  {k:>14}: mean={v[0] / v[1]:.3f}  (n={v[1]})")

    ra_path = os.environ.get("REGRET_AUDIT")
    if ra_path and _REGRET:
        json.dump(_REGRET, open(ra_path, "w"))
        early = [r for r in _REGRET if r[0] < r[1]]     # chosen k < realized-best k
        late = [r for r in _REGRET if r[0] > r[1]]      # chosen k > realized-best k
        exact = [r for r in _REGRET if r[0] == r[1]]
        n = len(_REGRET)
        loss = sum(r[3] - r[2] for r in _REGRET)         # total MAT loss vs realized-best-k
        loss_e = sum(r[3] - r[2] for r in early)
        loss_l = sum(r[3] - r[2] for r in late)
        print("\n== handoff regret audit (chosen k vs realized-optimal k) ==")
        print(f"  rounds={n}  early(too soon)={len(early)/n:.1%}  "
              f"exact={len(exact)/n:.1%}  late={len(late)/n:.1%}")
        print(f"  mean regret (acc_best-acc_chosen)/round = {loss/n:.4f}  "
              f"[from early {loss_e/n:.4f} + late {loss_l/n:.4f}]")
        print(f"  mean chosen_k={sum(r[0] for r in _REGRET)/n:.2f}  "
              f"realized-best_k={sum(r[1] for r in _REGRET)/n:.2f}")

    if ad is not None:
        tr = ad.trace
        print("\n== adaptive scalar ==")
        print(f"  mode={args.adaptive} head={args.adaptive_head} "
              f"final_w={ad.w_t:.4f} final_u={ad.u_t:.3f} "
              f"switches={ad.switches} credited={ad._credited} rounds={len(tr)}")
        if ad.mode in ("ftl", "ftl2d") and len(ad.cands) > 1:
            means = ad._means()
            if means:
                top = sorted(range(len(ad.cands)), key=lambda i: -means[i])[:5]
                print("  top candidates (windowed mean): " + "  ".join(
                    f"(u={ad.cands[i][0]},w={ad.cands[i][1]:g})={means[i]:.3f}"
                    for i in top))
        if args.adaptive_trace:
            json.dump({"mode": args.adaptive, "head": args.adaptive_head,
                       "cands": [list(c) for c in ad.cands],
                       "final_leader": int(ad.leader), "switches": int(ad.switches),
                       "w": [t[0] for t in tr], "u": [t[1] for t in tr],
                       "k": [t[2] for t in tr], "acc": [t[3] for t in tr],
                       "rid_index": ad_rid_index},
                      open(args.adaptive_trace, "w"))
            print(f"  trace -> {args.adaptive_trace}")


if __name__ == "__main__":
    main()
