"""Per-task raw->oracle gap breakdown (chain all-competition, fixed budget) to test
whether the small oracle gap is a few-tasks artifact. Reports each task's eagle/raw/
oracle MAT, raw/oracle ratio and gap%, plus suffix mean accept."""
from __future__ import annotations
import argparse, json, sys
import numpy as np
sys.path.insert(0, "/workspace")
from arctic_inference.suffix_decoding import SuffixDecodingCache
from simulation.scripts.goal2_realistic_policy import _eagle_chain, _suffix_chain, _mlen
from simulation.scripts.goal2_calib_ladder import _walk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--capture-steps", type=int, default=8)
    ap.add_argument("--capture-topk", type=int, default=8)
    args = ap.parse_args()
    S, K, B = args.capture_steps, args.capture_topk, args.budget
    d = json.load(open(args.record))
    cache = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100000)
    rid = 0
    hdr = f"{'task':>5}{'turns':>6}{'npos':>6}{'eagle':>8}{'raw':>8}{'oracle':>8}{'raw/orac%':>10}{'raw_gap%':>9}{'a_s':>7}"
    print(f"budget={B}\n" + hdr)
    rows = []
    for qi, q in enumerate(d["questions"]):
        E, R, O, AS = [], [], [], []
        nt = 0
        for turn in q["agent_metrics"]["steps"]:
            ents = (turn.get("spec_decode") or {}).get("oracle_vanilla_entries") or []
            gt, pools = [], []
            for e in ents:
                if not e.get("tokens"):
                    continue
                gt.append(e["tokens"][0][0]); pools.append(e.get("eagle3_pool_full") or {})
            if len(gt) < 2:
                continue
            nt += 1
            cache.start_request(rid, np.asarray(gt[:1], dtype=np.int32)); ctx = [gt[0]]
            for p in range(1, len(gt)):
                fut = gt[p:]
                et, ec = _eagle_chain(pools[p], S, K, B) if p < len(pools) else ([], [])
                a_e = _mlen(et, fut)
                try:
                    dr = cache.speculate(rid, np.asarray(ctx, dtype=np.int32), max_spec_tokens=B,
                                         max_spec_factor=100.0, min_token_prob=0.0, use_tree_spec=True)
                    st, sc = _suffix_chain(dr, B) if dr.token_ids else ([], [])
                except Exception:
                    st, sc = [], []
                a_s = _mlen(st, fut)
                E.append(a_e); AS.append(a_s)
                R.append(_walk(a_e, len(et), ec, a_s, len(st), sc, "raw"))
                O.append(_walk(a_e, len(et), ec, a_s, len(st), sc, "oracle"))
                cache.add_active_response(rid, [int(gt[p])]); ctx.append(gt[p])
            cache.stop_request(rid); rid += 1
        e, r, o, as_ = map(np.mean, (E, R, O, AS))
        gap = 100 * (r - e) / (o - e) if o > e else 0.0
        rows.append((e, r, o))
        print(f"{qi:>5}{nt:>6}{len(E):>6}{e:>8.3f}{r:>8.3f}{o:>8.3f}{100*r/o:>9.1f}%{gap:>8.1f}%{as_:>7.3f}")
    ee = np.array([x[0] for x in rows]); rr = np.array([x[1] for x in rows]); oo = np.array([x[2] for x in rows])
    ratio = 100 * rr / oo
    print(f"\nacross tasks: raw/oracle% mean={ratio.mean():.1f}  min={ratio.min():.1f}  "
          f"max={ratio.max():.1f}  std={ratio.std():.1f}")
    print(f"oracle-eagle headroom per task: {[round(float(x),3) for x in (oo-ee)]}")


if __name__ == "__main__":
    main()
