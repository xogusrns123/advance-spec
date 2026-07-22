"""Translate the decisive-selacc ladder into MAT: how much of the calib->oracle
MAT gap is REDUCIBLE on the currently-logged feature set?

Reuses irreducible_cases (same OOF joint-GBM / marginal-GBM on the decisive,
accept-conditioned set, counterfactual all-candidate labels from the oracle log).
Then simulates the counterfactual per-step accept length under each selection
policy by walking every chain on the GT trajectory:

  depth survives iff:
    oracle_hit == both               -> yes (either proposer hits)
    oracle_hit in (none, nogt)       -> no  (proposal-limited death; policy-invariant)
    oracle_hit == eagle              -> yes iff policy picks the eagle slot
    oracle_hit == suffix             -> yes iff policy picks suffix
  accept length = # consecutive survivals from depth 0.

Policies differ ONLY at decisive cells, so we attach the precomputed raw /
marginal-GBM (calib ceiling) / joint-GBM (Bayes) picks there; forced/both/dead
cells behave identically across policies. sim-oracle must reproduce the served
oracle MAT (cross-check).

Host, sklearn ok, pure stdout.
"""
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import irreducible_cases as ic  # noqa: E402

ROOT = "simulation/results/chain_hybrid_perdepth"
CELLS = {
    "14B 2-way (eagle3+sfx)": ("qwen3_14b_ar", ic.CELLS["14B 2-way (eagle3+sfx)"][1]),
    "27B 2-way (mtp+sfx)": ("qwen35_27b_ar", ic.CELLS["27B 2-way (mtp+sfx)"][1]),
}


def served_mat(path):
    acc = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        import json
        o = json.loads(line)
        if o.get("type") == "step":
            acc[(str(o["rid"]), o["decode_step"])] = o.get("accept_len")
    Ls = [v for v in acc.values() if v is not None]
    return sum(Ls) / max(len(Ls), 1)


def decisive_picks(chains, props):
    """Return {(rid,depth): {'raw':name,'calib':name,'joint':name}} over the
    decisive rows, via the SAME OOF procedure as irreducible_cases.analyze."""
    rows, names = ic.build_decisive(chains, props)
    if len(rows) < 30:
        return {}, names
    Xfull = ic.full_vector(rows, names)
    N = len(rows)
    pj = {nm: np.full(N, np.nan) for nm in names}
    pm = {nm: np.full(N, np.nan) for nm in names}
    for nm, _, _, extra in props:
        Xo, y, g, idx = ic.own_vector(rows, nm, extra)
        if not len(y):
            continue
        pm_k = ic.oof_gbm(Xo, y, g)
        pj_k = ic.oof_gbm(Xfull[idx], y, g)
        for j, i in enumerate(idx):
            pm[nm][i] = pm_k[j]
            pj[nm][i] = pj_k[j]

    def amax(score, e, i):
        return max(e["avail"], key=lambda nm: (score[nm][i] if not np.isnan(score[nm][i]) else -1))

    out = {}
    for i, e in enumerate(rows):
        out[(e["rid"], e["depth"])] = {
            "raw": max(e["avail"], key=lambda nm: e["prob"][nm]),
            "calib": amax(pm, e, i),
            "joint": amax(pj, e, i),
        }
    return out, names


def policy_mat(chains, picks, policy, suffix_name="suffix"):
    """Counterfactual per-step accept length under `policy` ('raw'|'calib'|'joint'
    |'oracle'). Walk each chain; a decisive depth survives iff the policy pick
    matches the sole hitter (== oracle_hit). oracle = always pick the hitter."""
    Ls = []
    for (rid, ds), rs in chains.items():
        acc = 0
        for r in rs:
            oh = r.get("oracle_hit")
            if oh == "both":
                acc += 1
                continue
            if oh in ("none", "nogt", None):
                break
            # decisive or forced single-hitter (oh in {eagle, suffix})
            if policy == "oracle":
                acc += 1  # pick the hitter by construction
                continue
            p = picks.get((rid, int(r["depth"])))
            if p is None:
                # forced (only one proposer available) -> the hitter is forced in;
                # survives (oh names the hitter, which is the only candidate).
                acc += 1
                continue
            picked_suffix = (p[policy] == suffix_name)
            survive = (oh == "suffix") == picked_suffix
            if survive:
                acc += 1
            else:
                break
        Ls.append(acc)
    return sum(Ls) / max(len(Ls), 1)


def main():
    for name, (dirname, props) in CELLS.items():
        d = f"{ROOT}/{dirname}"
        olog = f"{d}/decisions_select1_oracle.jsonl"
        chains = ic.load_chains(olog)
        bad = ic.loopy_rids(d)
        chains = {k: v for k, v in chains.items() if k[0] not in bad}
        picks, names = decisive_picks(chains, props)
        sfx = "suffix"
        sim = {p: policy_mat(chains, picks, p, sfx)
               for p in ("raw", "calib", "joint", "oracle")}
        srv_raw = served_mat(f"{d}/decisions_select1.jsonl")
        srv_orc = served_mat(olog)
        srv_bayes = None
        bp = Path(f"{d}/decisions_select1_bayes.jsonl")
        if bp.exists():
            srv_bayes = served_mat(str(bp))
        gap = sim["oracle"] - sim["raw"]
        print(f"\n===== {name} =====")
        print(f"  SIM (counterfactual, GT trajectory, logged features):")
        print(f"     raw          = {sim['raw']:.3f}")
        print(f"     calib (mrgGBM)= {sim['calib']:.3f}   ({100*(sim['calib']-sim['raw'])/gap:+.1f}% of sim gap)")
        print(f"     joint-GBM    = {sim['joint']:.3f}   ({100*(sim['joint']-sim['raw'])/gap:+.1f}% of sim gap)  <- logged-feature CEILING")
        print(f"     oracle       = {sim['oracle']:.3f}")
        print(f"  SERVED (real accept_len):")
        print(f"     raw={srv_raw:.3f}  bayes={srv_bayes if srv_bayes is None else round(srv_bayes,3)}  oracle={srv_orc:.3f}")
        print(f"  cross-check: sim-oracle {sim['oracle']:.3f} vs served-oracle {srv_orc:.3f} "
              f"(delta {sim['oracle']-srv_orc:+.3f})")
        # how much of the logged-feature ceiling does served bayes already capture?
        if srv_bayes is not None:
            cap = (srv_bayes - srv_raw) / max(sim["joint"] - srv_raw, 1e-9)
            print(f"  served-bayes captures {100*cap:.0f}% of (sim-joint-ceiling - served-raw)")


if __name__ == "__main__":
    main()
