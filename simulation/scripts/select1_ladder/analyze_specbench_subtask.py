"""Per-SUBTASK REALIZED diagnostic for a SpecBench select-1 capture (2-way: model + suffix).

ALL metrics REALIZED (served): per-subtask MAT = mean of the served arm's per-block accept_len
('step' decision-log entries {rid, decode_step, accept_len}) grouped by subtask. suffix-win-rate
from the oracle arm's decision entries. No block-anchored run-length.

rid->subtask mapping: the oracle arm logs 'req' entries (rid->input_ids) -> decode (Qwen tokenizer)
+ substring-match dataset turns[0]. The raw arm (a separate --replay-existing run) has NO req
entries and DIFFERENT rid hashes, but replayed the SAME record in the SAME order, so its rids are
mapped by FIRST-APPEARANCE ORDER zipped to the oracle arm's ordered rids.

Run IN docker (tokenizer):
  docker exec sglang-bench python3 /workspace/simulation/scripts/select1_ladder/analyze_specbench_subtask.py \
    --dir simulation/results/chain_hybrid_perdepth/specbench_qwen35_27b_mtp_2way --model Qwen/Qwen3.5-27B
"""
import json, argparse, os
from collections import defaultdict

ARMS = [("oracle", "decisions_select1_oracle.jsonl"),
        ("calib", "decisions_select1_calib_logistic.jsonl"),
        ("bayes", "decisions_select1_bayes.jsonl"),
        ("raw", "decisions_select1.jsonl")]
ORDER = ["math_reasoning", "mt_bench", "qa", "translation", "summarization", "rag"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="data/specbench/dataset_interleaved.jsonl")
    ap.add_argument("--prefix-len", type=int, default=60)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.dataset)]
    turn0 = lambda r: (r["turns"][0] if isinstance(r["turns"], list) else r["turns"])
    prefixes = [(turn0(r)[:args.prefix_len], r["subtask"]) for r in rows]

    steps = {a: defaultdict(list) for a, _ in ARMS}
    ordered = {a: [] for a, _ in ARMS}; seen = {a: set() for a, _ in ARMS}
    reqs = {}; oracle_dec = []; present = []
    for arm, fn in ARMS:
        p = f"{args.dir}/{fn}"
        if not os.path.exists(p): continue
        present.append(arm)
        for line in open(p):
            o = json.loads(line); t = o.get("type"); rid = o.get("rid")
            if rid is not None and rid not in seen[arm] and t in ("req", "step", "decision"):
                seen[arm].add(rid); ordered[arm].append(rid)
            if t == "req": reqs.setdefault(rid, o["input_ids"])
            elif t == "step": steps[arm][rid].append(float(o.get("accept_len", 0)))
            elif t == "decision" and arm == "oracle" and not o.get("tail"):
                oracle_dec.append((rid, o.get("eagle_token"), o.get("suffix_token"), o.get("gt_token")))
    print(f"arms present: {present}  ordered rids: " + ", ".join(f"{a}={len(ordered[a])}" for a in present))

    # oracle rid -> subtask via decode
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    osub = {}; miss = 0
    for rid in ordered.get("oracle", []):
        ids = reqs.get(rid)
        if ids is None: osub[rid] = None; miss += 1; continue
        txt = tok.decode(ids, skip_special_tokens=False)
        s = next((sub for pfx, sub in prefixes if pfx and pfx in txt), None)
        osub[rid] = s; miss += (s is None)
    print(f"oracle rid->subtask mapped={sum(v is not None for v in osub.values())}/{len(osub)} (unmapped {miss})")

    # per-arm rid->subtask: oracle by decode; arms w/o reqs by order-zip to oracle
    rid_sub = {"oracle": osub}
    oo = ordered.get("oracle", [])
    for a in present:
        if a == "oracle": continue
        oa = ordered[a]
        if len(oa) == len(oo):
            rid_sub[a] = {ra: osub.get(ro) for ra, ro in zip(oa, oo)}
        else:
            print(f"  WARN {a}: ordered rid count {len(oa)} != oracle {len(oo)}; order-zip skipped")
            rid_sub[a] = {}

    def mat_by_sub(a):
        acc = defaultdict(list)
        for rid, lens in steps[a].items():
            s = rid_sub.get(a, {}).get(rid)
            if s: acc[s].extend(lens)
        return {s: (sum(v)/len(v), len(v)) for s, v in acc.items() if v}
    mats = {a: mat_by_sub(a) for a in present}

    dec = defaultdict(lambda: [0, 0])
    for rid, et, st, gt in oracle_dec:
        s = osub.get(rid)
        if s is None or gt is None: continue
        av = [x for x in (et, st) if x is not None]; hits = [x for x in av if x == gt]
        if len(av) >= 2 and 0 < len(hits) < len(av):
            dec[s][0] += 1; dec[s][1] += (st == gt)

    # served arms in ladder order (whichever are present)
    served = [a for a in ("raw", "calib", "bayes", "oracle") if a in present]
    hdr = f"\n{'subtask':12} {'blocks':>7} {'sfx-win%':>9}"
    for a in served:
        hdr += f" {a+'MAT':>9}"
    hdr += f" {'bayes_rec%':>10}"
    print(hdr)
    data_subs = {r["subtask"] for r in rows}
    all_subs = [s for s in ORDER if s in data_subs] + sorted(data_subs - set(ORDER))
    for s in all_subs:
        vals = {a: mats.get(a, {}).get(s) for a in served}
        if not any(vals.values()):
            continue
        d = dec.get(s, [0, 0]); sw = 100*d[1]/d[0] if d[0] else float("nan")
        nb = next((v[1] for v in vals.values() if v), 0)
        line = f"{s:12} {nb:>7} {sw:>8.1f}%"
        for a in served:
            line += f" {(vals[a][0] if vals[a] else float('nan')):>9.3f}"
        r = vals.get("raw"); o = vals.get("oracle"); b = vals.get("bayes")
        rec = float("nan")
        if r and o and b and (o[0] - r[0]):
            rec = 100 * (b[0] - r[0]) / (o[0] - r[0])
        line += f" {rec:>9.0f}%"
        print(line)
    def overall(a):
        v = [x for lens in steps[a].values() for x in lens]; return sum(v)/len(v) if v else float("nan")
    print(f"\noverall realized MAT: " + "  ".join(f"{a}={overall(a):.3f}" for a in present))

if __name__ == "__main__":
    main()
