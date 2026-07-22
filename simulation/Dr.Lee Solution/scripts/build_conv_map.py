#!/usr/bin/env python3
"""Exact per-row conversation/task map for a full-trajectory gt_tokens.jsonl.

The lenreset heuristic (prompt-length drop) under/over-detects conversation
boundaries on the full collections (bfcl: 145 detected vs 169 real tasks), so
the warm/eval and calibrate/test splits would not be exactly task-disjoint.
This builder recovers the EXACT row->conversation assignment from the agent
results file instead:

  bfcl  agent_results_record.json: questions[*].agent_metrics.steps carry
        (prompt_tokens, completion_tokens) per llm call, which equal
        (len(input_ids), len(output_ids)) of the gt row -> greedy sequential
        exact alignment. Error steps (400, no token counts) are skipped;
        unmatched rows (supervised-restart residue) become orphans.
  swe   agent_results_all.json: questions[*].num_turns segments the row stream
        (one row per mini-swe-agent step). Each segment start is validated
        against a prompt-length reset; trailing leftover rows (the run was
        stopped mid-instance) become orphans.

Writes conv_map.json:
  {"kind", "n_rows",
   "rows":   [ {"conv": int, "task": str} | null, ... ]   one per gt row,
   "convs":  [ {"conv", "task_id", "label", "n_rows"} ],
   "orphan_rows": [...] }

  python3 scripts/build_conv_map.py --kind bfcl \
    --agent-json .../bfcl_v4_full_traj/qwen35_27b_dflash/agent_results_record.json \
    --gt-tokens  .../bfcl_v4_full_traj/qwen35_27b_dflash/gt_tokens.jsonl \
    --out results/perpos_bfcl_full/conv_map.json
"""
from __future__ import annotations
import argparse
import glob
import json
import os
from pathlib import Path

BFCL_LABEL = {
    "bfcl/memory_kv_prereq": "memory_kv",
    "bfcl/memory_rec_sum_prereq": "memory_rec_sum",
    "bfcl/memory_vector_prereq": "memory_vector",
    "bfcl/web_search_base": "web_search",
}


def _swe_label(instance_id: str) -> str:
    # "astropy__astropy-12907" -> "astropy", "pylint-dev__pylint-4551" -> "pylint"
    return instance_id.split("__")[1].rsplit("-", 1)[0]


def map_bfcl(questions, rl, lookahead=10):
    """Greedy sequential exact alignment on (prompt_tokens, completion_tokens)."""
    rows = [None] * len(rl)
    convs, orphans = [], []
    i = 0
    for ci, t in enumerate(questions):
        steps = [(s["prompt_tokens"], s["completion_tokens"])
                 for s in t["agent_metrics"]["steps"]
                 if s.get("type") == "llm" and "prompt_tokens" in s]
        label = BFCL_LABEL.get(t["category"], t["category"].split("/")[-1])
        n = 0
        for pt_ct in steps:
            j = i
            while j < len(rl) and rl[j] != pt_ct and j - i < lookahead:
                j += 1
            if j >= len(rl) or rl[j] != pt_ct:
                raise SystemExit(f"ALIGN FAIL: task {ci} {t['bfcl_id']} expects "
                                 f"{pt_ct}, rows[{i}:{i+4}]={rl[i:i+4]}")
            orphans.extend(range(i, j))
            rows[j] = {"conv": ci, "task": label}
            i = j + 1
            n += 1
        convs.append({"conv": ci, "task_id": t["bfcl_id"], "label": label, "n_rows": n})
    orphans.extend(range(i, len(rl)))
    return rows, convs, orphans


def _norm_text(s: str) -> str:
    """Comparable prefix of an assistant response: drop think blocks / special
    tokens / ALL whitespace, keep the first 100 chars."""
    import re
    s = re.sub(r"<think>.*?</think>", "", s or "", flags=re.S)
    s = s.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    return re.sub(r"\s+", "", s)[:100]


def map_content(questions, row_texts, steps_of, task_of, label_of, window=400):
    """Greedy sequential CONTENT alignment (swe/spider): per question, per step,
    bind the next gt row whose decoded output prefix equals the step's recorded
    response text (both _norm_text'd). Interleaved non-matching rows (aborted-
    attempt/retry residue) become orphans. Replaces num_turns block
    segmentation, which breaks when retries leave partial-attempt rows in the
    stream or when consecutive instances lack a prompt-length reset (swe:
    17889 -> 18227 at an astropy->django boundary)."""
    rows = [None] * len(row_texts)
    convs, orphans = [], []
    i = 0
    for ci, t in enumerate(questions):
        label = label_of(t)
        n = 0
        for si, st in enumerate(steps_of(t)):
            st = _norm_text(st)
            if not st:
                continue
            j = i
            while j < len(row_texts) and j - i < window and row_texts[j] != st:
                j += 1
            if j >= len(row_texts) or row_texts[j] != st:
                raise SystemExit(
                    f"ALIGN FAIL: q{ci} {task_of(t)} step {si}: no content match "
                    f"in rows[{i}:{i+window}]; want '{st[:60]}...', "
                    f"row[{i}]='{row_texts[i][:60] if i < len(row_texts) else ''}...'")
            orphans.extend(range(i, j))
            rows[j] = {"conv": ci, "task": label}
            i = j + 1
            n += 1
        convs.append({"conv": ci, "task_id": task_of(t), "label": label, "n_rows": n})
    orphans.extend(range(i, len(row_texts)))
    return rows, convs, orphans


def map_swe(questions, row_texts):
    return map_content(
        questions, row_texts,
        steps_of=lambda t: [s.get("content") or "" for s in t["agent_metrics"]["steps"]],
        task_of=lambda t: t["instance_id"],
        label_of=lambda t: _swe_label(t["instance_id"]))


def map_spider(questions, row_texts):
    return map_content(
        questions, row_texts,
        steps_of=lambda t: [s.get("response") or ""
                            for s in t["trajectory"]["trajectory"]],
        task_of=lambda t: t["instance_id"],
        label_of=lambda t: "spider_dbt")


def map_tau2(sims_dir, rl, lookahead=60):
    """tau2-bench (Sierra). Only the AGENT (assistant) generations are the model
    under test — the user turns are LLM-simulated and out of scope. Each
    simulation (one task conversation) lives in sims/<domain>_rNN.json/results.json;
    the domain (airline/retail/telecom) is the subtask label. Assistant messages
    carry usage=(prompt_tokens, completion_tokens) that EXACTLY equal a gt row's
    (len input_ids, len output_ids).

    The RR harness runs tasks in CONCURRENT chunks (chunk=5), so the gt stream
    INTERLEAVES the generations of the ~5 co-running conversations. A single
    sequential pass therefore fails; instead do a multi-way merge — each row is
    assigned to whichever (earliest-started) still-open conversation is expecting
    exactly that (prompt,completion) length next. Concurrent chunks are per-domain,
    so the label is unambiguous even when the exact conv id has a rare collision;
    user-sim / tool / residue rows match no open turn and become orphans."""
    sims = []
    for f in sorted(glob.glob(os.path.join(sims_dir, "*.json", "results.json"))):
        domain = os.path.basename(os.path.dirname(f)).split("_r")[0]
        for sim in json.load(open(f)).get("simulations", []):
            sims.append((sim.get("start_time", ""), domain, sim))
    sims.sort(key=lambda x: x[0])                       # chronological start order
    conv_turns, meta = [], []
    for st, domain, sim in sims:
        conv_turns.append([(m["usage"]["prompt_tokens"], m["usage"]["completion_tokens"])
                           for m in sim["messages"]
                           if m["role"] == "assistant" and m.get("usage")])
        meta.append((str(sim.get("task_id", len(meta))), domain))
    ptr = [0] * len(conv_turns)
    rows, orphans = [None] * len(rl), []
    for ri, key in enumerate(rl):
        ci = next((c for c in range(len(conv_turns))
                   if ptr[c] < len(conv_turns[c]) and conv_turns[c][ptr[c]] == key), None)
        if ci is None:
            orphans.append(ri)
            continue
        rows[ri] = {"conv": ci, "task": meta[ci][1]}
        ptr[ci] += 1
    convs = [{"conv": c, "task_id": meta[c][0], "label": meta[c][1], "n_rows": ptr[c]}
             for c in range(len(conv_turns))]
    unmatched = sum(len(t) - p for t, p in zip(conv_turns, ptr))
    if unmatched:
        print(f"[tau2] WARNING: {unmatched} agent turns unmatched "
              f"(convs short of their message count)")
    return rows, convs, orphans


def map_specbench(questions, rl, lookahead=10):
    """Exact (prompt_tokens, completion_tokens) alignment over questions[].turns
    (specbench_full_traj agent_results.json). Interior orphans (collection-start
    retries) are skipped via bounded lookahead, like bfcl."""
    rows = [None] * len(rl)
    convs, orphans = [], []
    i = 0
    for ci, t in enumerate(questions):
        label = t["category"]
        n = 0
        for turn in t["turns"]:
            if turn.get("prompt_tokens") is None:
                continue
            pt_ct = (int(turn["prompt_tokens"]), int(turn["completion_tokens"]))
            j = i
            while j < len(rl) and rl[j] != pt_ct and j - i < lookahead:
                j += 1
            if j >= len(rl) or rl[j] != pt_ct:
                raise SystemExit(f"ALIGN FAIL: q{ci} id={t['question_id']} expects "
                                 f"{pt_ct}, rows[{i}:{i+4}]={rl[i:i+4]}")
            orphans.extend(range(i, j))
            rows[j] = {"conv": ci, "task": label}
            i = j + 1
            n += 1
        convs.append({"conv": ci, "task_id": t["question_id"], "label": label, "n_rows": n})
    orphans.extend(range(i, len(rl)))
    return rows, convs, orphans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True,
                    choices=["bfcl", "swe", "specbench", "spider", "tau2"])
    ap.add_argument("--agent-json", help="agent results json (bfcl/swe/specbench/spider)")
    ap.add_argument("--parts-dir", help="swe/spider: load questions by merging "
                    "parts/part_*.json (chunked collection not yet consolidated)")
    ap.add_argument("--sims-dir", help="tau2: dir of <domain>_rNN.json/results.json")
    ap.add_argument("--gt-tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-27B",
                    help="tokenizer for swe/spider content alignment")
    ap.add_argument("--lookahead", type=int, default=10,
                    help="bfcl/specbench: max interleaved orphan rows to skip "
                         "per step (raise when failed-task residue blocks are "
                         "large, e.g. bfcl web_search retries)")
    ap.add_argument("--require-marker", default="",
                    help="swe: keep ONLY conversations whose step contents contain "
                         "this marker (e.g. COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT — "
                         "the model self-terminated). Filtering is applied AFTER "
                         "alignment (dropped convs' rows become orphans), so the "
                         "content alignment stays clean.")
    args = ap.parse_args()

    def _load_questions():
        if args.parts_dir:
            qs = []
            for f in sorted(glob.glob(os.path.join(args.parts_dir, "part_*.json"))):
                qs += json.load(open(f)).get("questions", [])
            return qs
        return json.load(open(args.agent_json))["questions"]

    gt_rows = [json.loads(l) for l in open(args.gt_tokens) if l.strip()]

    if args.kind == "tau2":
        if not args.sims_dir:
            ap.error("tau2 requires --sims-dir")
        rl = [(len(r["input_ids"]), len(r["output_ids"])) for r in gt_rows]
        rows, convs, orphans = map_tau2(args.sims_dir, rl, lookahead=args.lookahead)
        args.agent_json = args.agent_json or "(tau2 sims)"
    elif args.kind in ("swe", "spider"):
        questions = _load_questions()
        # content alignment: decoded prefix (first 80 output tokens) per row
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        row_texts = [_norm_text(tok.decode((r.get("output_ids") or [])[:80]))
                     for r in gt_rows]
        mapper = {"swe": map_swe, "spider": map_spider}[args.kind]
        rows, convs, orphans = mapper(questions, row_texts)
    else:
        questions = _load_questions()
        rl = [(len(r["input_ids"]), len(r["output_ids"])) for r in gt_rows]
        mapper = {"bfcl": map_bfcl, "specbench": map_specbench}[args.kind]
        rows, convs, orphans = mapper(questions, rl, lookahead=args.lookahead)

    # completion filter (applied AFTER alignment): drop convs whose question's step
    # contents lack the marker; their rows become orphans so capture/replay excludes
    # them. task_id == instance_id for swe, so match convs to questions by task_id.
    if args.require_marker:
        def _steps_text(q):
            am = q.get("agent_metrics", {})
            steps = am.get("steps", []) if isinstance(am, dict) else []
            return "\n".join(s.get("content") or "" for s in steps)
        keep_ids = {q.get("instance_id") for q in questions
                    if args.require_marker in _steps_text(q)}
        n_before = len(convs)
        dropped = [c["conv"] for c in convs if c["task_id"] not in keep_ids]
        drop_set = set(dropped)
        for i, r in enumerate(rows):
            if r is not None and r["conv"] in drop_set:
                rows[i] = None
                orphans.append(i)
        orphans = sorted(set(orphans))
        convs = [c for c in convs if c["conv"] not in drop_set]
        print(f"[{args.kind}] require-marker '{args.require_marker}': "
              f"kept {len(convs)}/{n_before} convs (dropped {len(dropped)} not self-terminated)")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"kind": args.kind, "n_rows": len(rows), "rows": rows,
               "convs": convs, "orphan_rows": orphans}, open(outp, "w"))
    from collections import Counter
    per = Counter(c["label"] for c in convs)
    print(f"[{args.kind}] {len(gt_rows)} rows -> {len(convs)} convs "
          f"({dict(per)}), orphans={len(orphans)} "
          f"{orphans if len(orphans) < 40 else str(orphans[:40]) + '...'}")
    print(f"saved -> {outp}")


if __name__ == "__main__":
    main()
