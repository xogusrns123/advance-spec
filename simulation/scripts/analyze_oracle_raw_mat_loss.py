"""Standalone: oracle-raw MAT loss per depth (bars) + cumulative (line)."""
from __future__ import annotations
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = "simulation/results/chain_hybrid_perdepth/qwen3_14b_tp"
ORACLE_LOG = f"{DIR}/decisions_select1_oracle.jsonl"
OUT = "simulation/results/calib_why_analysis/figures/oracle_raw_mat_loss.png"
ALIVE = {"eagle", "suffix", "both"}
MAXD = 16

chains = defaultdict(list); step_acc = {}
for line in open(ORACLE_LOG):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    t = r.get("type")
    if t == "decision" and not r.get("tail"):
        chains[(r["rid"], r["decode_step"])].append(r)
    elif t == "step":
        step_acc[(r["rid"], r["decode_step"])] = r.get("accept_len")
for k in chains:
    chains[k].sort(key=lambda r: r["depth"])

def raw_pick(r):
    if r["suffix_p"] is None: return "eagle"
    if r["eagle_p"] is None: return "suffix"
    return "suffix" if r["suffix_p"] > r["eagle_p"] else "eagle"
def is_correct(pk, hit):
    return hit == "both" or (hit == "eagle" and pk == "eagle") or (hit == "suffix" and pk == "suffix")

loss = np.zeros(MAXD); N = 0
for k, rs in chains.items():
    Lo = step_acc.get(k)
    if Lo is None: continue
    N += 1
    for r in rs:
        hit = r.get("oracle_hit"); d = r["depth"]
        if hit not in ALIVE:
            break                                   # raw (& oracle) die together here
        if not is_correct(raw_pick(r), hit):
            if hit in ("eagle", "suffix"):          # raw's first decisive error
                loss[d] += (Lo - d)
            break
loss /= N
cum = np.cumsum(loss)
total = cum[-1]
print(f"N={N} total oracle-raw MAT loss={total:.4f}")
print("per-depth share:", " ".join(f"d{d}:{loss[d]/total*100:.0f}%" for d in range(6)))

dd = np.arange(MAXD)
fig, ax = plt.subplots(figsize=(11, 6.4))
bars = ax.bar(dd, loss, color="#2ca02c", alpha=0.85, label="oracle−raw MAT loss at depth d")
for d in range(MAXD):
    if loss[d] > 0.004:
        ax.text(d, loss[d] + 0.004, f"{loss[d]:.3f}", ha="center", fontsize=8)
ax.plot(dd, cum, "-o", color="#d62728", lw=2.2, ms=5,
        label="cumulative oracle−raw MAT loss")
ax.axhline(total, color="#d62728", ls=":", lw=1, alpha=0.6)
ax.text(15, total + 0.006, f"total = {total:.3f}", ha="right", color="#d62728", fontsize=10)
# share annotations
ax.annotate(f"depth 0 alone = {loss[0]/total*100:.0f}%",
            xy=(0, loss[0]), xytext=(2.2, 0.235),
            arrowprops=dict(arrowstyle="->", color="#2ca02c"), fontsize=9, color="#1a7a1a")
ax.annotate(f"depth 0–2 = {cum[2]/total*100:.0f}%", xy=(2, cum[2]), xytext=(4.0, 0.34),
            arrowprops=dict(arrowstyle="->", color="#d62728"), fontsize=9, color="#a11")
ax.set_xlabel("draft depth d  (where raw makes its first decisive selection error)")
ax.set_ylabel("MAT loss vs oracle (tokens)")
ax.set_xticks(dd); ax.set_ylim(0, total * 1.12)
ax.set_title("Oracle−raw MAT loss by first-error depth: front-loaded\n"
             "early-depth selection errors forfeit the whole downstream chain")
ax.legend(fontsize=9, loc="center right"); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
