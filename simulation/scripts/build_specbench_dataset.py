#!/usr/bin/env python3
"""Build the standard Spec-Bench dataset for oracle trajectory collection.

Spec-Bench (Xia et al., "Unlocking Efficiency in Large Language Model
Inference: A Comprehensive Survey of Speculative Decoding", ACL 2024;
https://github.com/hemingkx/Spec-Bench) is the de-facto standard
speculative-decoding benchmark. It is a single file of 480 questions
spanning **six** sub-tasks, 80 each:

    * mt_bench         multi-turn conversation (MT-Bench, 2 turns each;
                       8 fine-grained categories x 10)
    * translation      WMT14 DE->EN          (1 turn)
    * summarization    CNN/Daily Mail        (1 turn)
    * qa               Natural Questions     (1 turn)
    * math_reasoning   GSM8K                 (1 turn)
    * rag              Natural Questions+ctx (1 turn)

The upstream `question.jsonl` labels every row with a fine-grained
`category` (14 distinct values — the 8 MT-Bench categories plus the 5
single-turn task names). This builder preserves that `category`
verbatim for fidelity and additionally tags each row with the coarse
6-way `subtask` that Spec-Bench reports speed-ups against.

Output (matches simulation.agents.specbench_agent + run_experiment.py
WORKLOAD_REGISTRY["specbench"]):

    data/specbench/dataset.jsonl              480 rows, sorted by question_id
    data/specbench/dataset_interleaved.jsonl  same rows, round-robin by subtask
    data/specbench/SOURCE.txt                 provenance (url, commit, sha256)

Round-robin-by-subtask matters: round-robin capture often early-stops at
N requests, and interleaving by the 6 reported subtasks (rather than the
14 fine categories) keeps any prefix balanced across what Spec-Bench
reports. Interleaving by the 14 categories would over-sample MT-Bench
8:1 in every early-stopped prefix.

Usage:
    python3 simulation/scripts/build_specbench_dataset.py
    python3 simulation/scripts/build_specbench_dataset.py \
        --from-file /path/to/question.jsonl   # offline / vendored source
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "specbench"

# Pinned to the commit that last touched question.jsonl, so re-runs are
# byte-reproducible regardless of upstream churn.
SOURCE_COMMIT = "66230f10cb0a02aced5ef3ce1e85163c16160454"
SOURCE_URL = (
    "https://raw.githubusercontent.com/hemingkx/Spec-Bench/"
    f"{SOURCE_COMMIT}/data/spec_bench/question.jsonl"
)
EXPECTED_SHA256 = "4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04"

# Fine-grained `category` -> coarse Spec-Bench `subtask`. The 8 MT-Bench
# categories collapse into "mt_bench"; the 5 single-turn tasks map to
# themselves. (Note: MT-Bench's own "math" category is distinct from the
# GSM8K "math_reasoning" subtask — they must not be merged.)
MT_BENCH_CATEGORIES = {
    "writing", "roleplay", "reasoning", "math",
    "coding", "extraction", "stem", "humanities",
}
SINGLE_TURN_SUBTASKS = {"translation", "summarization", "qa", "math_reasoning", "rag"}
SUBTASK_ORDER = ["mt_bench", "translation", "summarization", "qa", "math_reasoning", "rag"]


def subtask_of(category: str) -> str:
    if category in MT_BENCH_CATEGORIES:
        return "mt_bench"
    if category in SINGLE_TURN_SUBTASKS:
        return category
    raise ValueError(f"Unknown Spec-Bench category: {category!r}")


def fetch_source(from_file: str | None) -> bytes:
    if from_file:
        return Path(from_file).read_bytes()
    print(f"Downloading {SOURCE_URL}")
    with urllib.request.urlopen(SOURCE_URL, timeout=120) as r:
        return r.read()


def normalize(raw: bytes) -> list[dict]:
    rows: list[dict] = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        category = d["category"]
        rec = {
            "question_id": d["question_id"],
            "category": category,
            "subtask": subtask_of(category),
            "turns": d["turns"],
        }
        # `reference` is present for some tasks (translation, summarization,
        # math_reasoning, rag, and a few MT-Bench rows). Preserve when given.
        if "reference" in d:
            rec["reference"] = d["reference"]
        rows.append(rec)
    return rows


def interleave_by_subtask(rows: list[dict]) -> list[dict]:
    """Round-robin by subtask: sub0[0], sub1[0], ..., sub0[1], sub1[1], ...

    First len(subtasks) items cover every reported Spec-Bench subtask.
    """
    buckets: OrderedDict[str, list[dict]] = OrderedDict(
        (s, []) for s in SUBTASK_ORDER
    )
    for r in rows:
        buckets[r["subtask"]].append(r)
    out: list[dict] = []
    max_len = max(len(b) for b in buckets.values())
    for i in range(max_len):
        for s in SUBTASK_ORDER:
            if i < len(buckets[s]):
                out.append(buckets[s][i])
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-file", default=None,
                    help="Use a local question.jsonl instead of downloading.")
    ap.add_argument("--no-verify-sha", action="store_true",
                    help="Skip the source sha256 check (use with --from-file).")
    args = ap.parse_args()

    raw = fetch_source(args.from_file)
    sha = hashlib.sha256(raw).hexdigest()
    if not args.no_verify_sha and not args.from_file and sha != EXPECTED_SHA256:
        raise SystemExit(
            f"Source sha256 mismatch:\n  got      {sha}\n  expected {EXPECTED_SHA256}\n"
            "Upstream changed; review before trusting the pinned commit.")

    rows = normalize(raw)
    if len(rows) != 480:
        raise SystemExit(f"Expected 480 Spec-Bench questions, got {len(rows)}")

    rows_sorted = sorted(rows, key=lambda r: r["question_id"])
    interleaved = interleave_by_subtask(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "dataset.jsonl", rows_sorted)
    write_jsonl(OUT_DIR / "dataset_interleaved.jsonl", interleaved)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (OUT_DIR / "SOURCE.txt").write_text(
        "Spec-Bench (hemingkx/Spec-Bench) standard benchmark\n"
        f"url:    {SOURCE_URL}\n"
        f"commit: {SOURCE_COMMIT}\n"
        f"sha256: {sha}\n"
        f"built:  {now}\n"
        f"rows:   {len(rows)} (6 subtasks x 80)\n"
    )

    # Report
    from collections import Counter
    subc = Counter(r["subtask"] for r in rows)
    catc = Counter(r["category"] for r in rows)
    print(f"wrote {OUT_DIR/'dataset.jsonl'} ({len(rows_sorted)} rows)")
    print(f"wrote {OUT_DIR/'dataset_interleaved.jsonl'} ({len(interleaved)} rows)")
    print("subtasks:", dict(subc))
    print("first 6 interleaved subtasks:",
          [r["subtask"] for r in interleaved[:6]])
    print("categories:", dict(catc))


if __name__ == "__main__":
    main()
