#!/usr/bin/env python3
"""Render chain-hybrid agent_results_*.json into a human-readable markdown
trajectory dump: per task, the user question, then each step's reasoning,
tool call(s), and tool result(s), then the final answer. Runaway steps
(degenerate repetition to max_tokens) are flagged and truncated head+tail.

Usage:
    python3 simulation/scripts/extract_trajectories.py <agent_results.json> \
        [--out path.md] [--reason-chars 900] [--max-iter 8]
"""
from __future__ import annotations
import argparse, ast, json, re
from pathlib import Path

RUNAWAY_CHARS = 20000


def parse_results(raw):
    """exec_results elements are str(list[dict]); pull a compact view."""
    out = []
    for item in (raw or []):
        try:
            obj = ast.literal_eval(item) if isinstance(item, str) else item
        except Exception:
            out.append(str(item)[:200] + " …"); continue
        if isinstance(obj, list):
            for r in obj[:6]:
                if isinstance(r, dict):
                    t = r.get("title") or r.get("href") or ""
                    b = (r.get("body") or "")[:140].replace("\n", " ")
                    out.append(f"- {t}" + (f" — {b}…" if b else ""))
                else:
                    out.append(f"- {str(r)[:160]}")
            if len(obj) > 6:
                out.append(f"- … (+{len(obj) - 6} more results)")
        elif isinstance(obj, dict):
            for k in ("content", "text", "markdown", "body"):
                if obj.get(k):
                    out.append(f"  {str(obj[k])[:300].strip()} …"); break
            else:
                out.append(str(obj)[:200] + " …")
        else:
            out.append(str(obj)[:300] + " …")
    return out


def fmt_step(s, reason_chars):
    L = []
    ct = (s.get("content") or "").strip()
    tokc = s.get("completion_tokens")
    runaway = len(ct) > RUNAWAY_CHARS or (tokc is not None and tokc >= 8192)
    head = f"#### Step {s.get('step')}  ({tokc} tok)" + ("  ⚠️ **RUNAWAY**" if runaway else "")
    L.append(head)
    if runaway:
        L.append("> 🔁 degenerate repetition to max_tokens — showing head + tail only\n")
        L.append("```\n" + ct[:600] + "\n\n[… " + str(len(ct) - 1200) +
                 " chars of looping reasoning omitted …]\n\n" + ct[-600:] + "\n```")
    elif ct:
        L.append("> " + ct[:reason_chars].replace("\n", "\n> ") +
                 (" …" if len(ct) > reason_chars else ""))
    for call in (s.get("decoded_calls") or []):
        L.append(f"\n**🔧 tool call:** `{call}`")
    res = parse_results(s.get("exec_results"))
    if res:
        L.append("\n**📄 result:**")
        L.extend(res)
    if s.get("exec_error"):
        L.append(f"\n**❌ exec_error:** {str(s['exec_error'])[:300]}")
    return "\n".join(L), runaway


def render(path, reason_chars, max_iter):
    d = json.load(open(path))
    qs = d.get("questions", [])
    meta = d.get("metadata", {})
    out = [f"# Trajectories — {Path(path).name}", ""]
    out.append(f"model: `{meta.get('model','?')}` · benchmark: `{meta.get('benchmark','?')}` "
               f"· {len(qs)} tasks\n")

    # health summary table
    out.append("## Health summary\n")
    out.append("| task | steps | tools | runaway | hit_iter | status |")
    out.append("|---|---|---|---|---|---|")
    bodies = []
    for q in qs:
        bid = str(q.get("bfcl_id"))
        steps = q.get("agent_metrics", {}).get("steps", [])
        llm = [s for s in steps if s.get("type", "llm") == "llm"]
        tools = {}
        any_runaway = False
        sec = [f"\n## {bid}", ""]
        # user question
        msgs = steps[0].get("messages") if steps else None
        uq = ""
        for m in (msgs or []):
            if isinstance(m, dict) and m.get("role") == "user":
                uq = (m.get("content") or "").strip(); break
        if uq:
            sec.append(f"**❓ Question:** {uq[:600]}\n")
        for s in steps:
            for c in (s.get("decoded_calls") or []):
                nm = str(c).lstrip("[").split("(", 1)[0].strip()
                tools[nm] = tools.get(nm, 0) + 1
            body, rw = fmt_step(s, reason_chars)
            any_runaway = any_runaway or rw
            sec.append(body)
        # final answer = last llm step content (if not runaway)
        if llm:
            last = (llm[-1].get("content") or "").strip()
            if last and len(last) <= RUNAWAY_CHARS:
                sec.append(f"\n**✅ Final answer:**\n\n> " + last[:1200].replace("\n", "\n> "))
        hit = len(llm) >= max_iter
        status = "⚠️ RUNAWAY" if any_runaway else ("⚠️ hit iter-cap" if hit else "✅ ok")
        tstr = ", ".join(f"{k}×{v}" for k, v in tools.items()) or "—"
        out.append(f"| {bid} | {len(llm)} | {tstr} | {'yes' if any_runaway else ''} "
                   f"| {'yes' if hit else ''} | {status} |")
        bodies.append("\n".join(sec))

    n_rw = sum(1 for b in bodies if "RUNAWAY" in b)
    out.insert(3, f"**{len(qs) - n_rw}/{len(qs)} healthy · {n_rw} runaway**\n")
    out.append("\n---\n")
    out.extend(bodies)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default=None)
    ap.add_argument("--reason-chars", type=int, default=900)
    ap.add_argument("--max-iter", type=int, default=8)
    args = ap.parse_args()
    md = render(args.path, args.reason_chars, args.max_iter)
    out = args.out or str(Path(args.path).with_suffix("")) + "_readable.md"
    Path(out).write_text(md)
    print(f"wrote {out}  ({len(md)} chars)")


if __name__ == "__main__":
    main()
