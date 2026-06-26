"""OFFLINE screen: windowed ONLINE joint discriminator vs static (frozen) disc.

Stream alive-decisive decisions in serving (file) order. The online disc keeps a
rolling window of the last W (feature, comparative-label) pairs (label observable
only for verified=alive positions, so it is naturally accept-conditioned), refits
a logistic every REFIT decisions, and predicts causally (window strictly precedes
the decision). Compare decisive accuracy + reconstructed MAT to raw, the static
our-Bayes disc (train-fit, frozen), and across window sizes. This screens whether
online adaptation buys anything BEFORE building a serving arm.
"""
from __future__ import annotations
import json
from collections import defaultdict, deque
import numpy as np
from sklearn.linear_model import LogisticRegression

DIR = "simulation/results/chain_hybrid_perdepth/qwen3_14b_tp"
EVAL = f"{DIR}/decisions_select1_oracle.jsonl"
STATIC = "/tmp/claude-20051/-home-muchwater-advance-spec/506d373c-a510-4773-92ed-a00b2aa3e05a/scratchpad/disc_ourbayes/disc_logistic.json"  # our-Bayes (6 feat) frozen
ALIVE = {"eagle", "suffix", "both"}
WINDOWS = [256, 512, 1024, 2048]
REFIT = 16
MIN_SAMPLES = 64
FEATS = lambda sp, ep, ml, c, n, d: [sp, ep, ml or 0, c or 0, n or 0, d]

# ---- load chains (for reconstruction) + alive-decisive stream (file order) ------
chains = defaultdict(list)
order = []   # list of (rid, ds, depth) alive-decisive, in file order
feats, labels = [], []
seen = set()
raw_buf = []
for line in open(EVAL):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if r.get("type") != "decision" or r.get("tail"):
        continue
    chains[(r["rid"], r["decode_step"])].append(r)
# rebuild alive-decisive stream in the order chains close (file order ~ processing)
step_acc = {}
for line in open(EVAL):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if r.get("type") == "step":
        step_acc[(r["rid"], r["decode_step"])] = r.get("accept_len")
for k in chains:
    chains[k].sort(key=lambda r: r["depth"])

# stream: walk chains in file order; within a chain, depths in order; alive only
chain_keys = []
_seen = set()
for line in open(EVAL):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if r.get("type") != "decision" or r.get("tail"):
        continue
    k = (r["rid"], r["decode_step"])
    if k not in _seen:
        _seen.add(k); chain_keys.append(k)

stream = []  # (rid,ds,depth, feat, label, ep, sp)
for k in chain_keys:
    alive = True
    for r in chains[k]:
        if not alive:
            break
        h = r.get("oracle_hit")
        if h in ("eagle", "suffix") and r["eagle_p"] is not None and r["suffix_p"] is not None:
            f = FEATS(r["suffix_p"], r["eagle_p"], r["match_len"], r["suffix_count"],
                      r["suffix_total"], r["depth"])
            stream.append((k[0], k[1], r["depth"], f, 1 if h == "suffix" else 0,
                           r["eagle_p"], r["suffix_p"]))
        if h not in ALIVE:
            alive = False
print(f"alive-decisive stream n={len(stream)}")

# ---- static disc (frozen our-Bayes logistic) -----------------------------------
sb = json.load(open(STATIC))
s_mean = np.array(sb["mean"]); s_std = np.array(sb["std"])
s_coef = np.array(sb["coef"]); s_int = sb["intercept"]
def static_pick(f):
    z = s_int + np.dot(s_coef, (np.array(f) - s_mean) / s_std)
    return 1 if z > 0 else 0   # P>0.5

def fit_window(W):
    X = np.array([s[3] for s in W]); y = np.array([s[4] for s in W])
    if len(set(y.tolist())) < 2:
        return None
    mu = X.mean(0); sd = X.std(0) + 1e-9
    clf = LogisticRegression(max_iter=500)
    clf.fit((X - mu) / sd, y)
    return (mu, sd, clf)

def online_run(W):
    win = deque(maxlen=W)
    model = None
    picks = {}
    n_acc = 0; n_dec = 0
    for i, s in enumerate(stream):
        f, lab = s[3], s[4]
        # predict causally with current model (fit on window strictly before i)
        if model is not None and len(win) >= MIN_SAMPLES:
            mu, sd, clf = model
            z = clf.decision_function(((np.array(f) - mu) / sd).reshape(1, -1))[0]
            pk = 1 if z > 0 else 0
        else:
            pk = 1 if s[6] > s[5] else 0     # fallback: raw sp>ep
        picks[(s[0], s[1], s[2])] = "suffix" if pk else "eagle"
        n_acc += (pk == lab); n_dec += 1
        # ingest realized label, then maybe refit
        win.append(s)
        if len(win) >= MIN_SAMPLES and i % REFIT == 0:
            m = fit_window(win)
            if m:
                model = m
    return n_acc / n_dec, picks

# ---- reconstruction ------------------------------------------------------------
def is_correct(pk, hit):
    if hit == "both":
        return True
    if hit == "eagle":
        return pk == "eagle"
    if hit == "suffix":
        return pk == "suffix"
    return False
def reconstruct(pickmap):
    Ls = []
    for k in chain_keys:
        served = step_acc.get(k)
        if served is None:
            continue
        L = served
        for r in chains[k]:
            hit = r.get("oracle_hit")
            if hit not in ALIVE:
                L = r["depth"]; break
            if hit in ("eagle", "suffix"):
                pk = pickmap.get((k[0], k[1], r["depth"]),
                                 "suffix" if (r["suffix_p"] or 0) > (r["eagle_p"] or 0) else "eagle")
                if not is_correct(pk, hit):
                    L = r["depth"]; break
        Ls.append(L)
    return float(np.mean(Ls))

# raw + static references
raw_pick = {(s[0], s[1], s[2]): ("suffix" if s[6] > s[5] else "eagle") for s in stream}
stat_pick = {(s[0], s[1], s[2]): ("suffix" if static_pick(s[3]) else "eagle") for s in stream}
raw_acc = np.mean([(1 if s[6] > s[5] else 0) == s[4] for s in stream])
stat_acc = np.mean([static_pick(s[3]) == s[4] for s in stream])
print(f"\n{'method':24s} {'decisive_acc':>12s} {'recon_MAT':>10s}")
print(f"{'raw (sp>ep)':24s} {raw_acc:12.3f} {reconstruct(raw_pick):10.4f}")
print(f"{'static disc (frozen)':24s} {stat_acc:12.3f} {reconstruct(stat_pick):10.4f}")
for W in WINDOWS:
    acc, picks = online_run(W)
    print(f"{'online disc W=%d' % W:24s} {acc:12.3f} {reconstruct(picks):10.4f}")
print("\n(recon MAT scale: served raw 1.318 / static disc 1.365 ~ recon+0.026; "
      "compare RELATIVE across rows here)")
