"""Generate anchor_depth_analysis.ipynb. Run once: /usr/bin/python3 _gen_anchor_depth_nb.py
(use /usr/bin/python3 — the repo .venv lacks numpy). Kept in-repo so the notebook
can be regenerated/edited from source.

Analysis: INDEPENDENT single-depth-graft simulations. For each anchor depth d we
run the extension grafting the suffix ONLY at depth d (method extension_gd:D;
d=0 = root hybrid = the two trees merged at root), and measure its survival over
ALL steps — a standalone hybrid, not a slice of the full extension. Because the
greedy tree walk takes the per-step longest path, even a single-depth graft acts
like max(backbone, suffix) on the steps it activates, so shallow grafts beat BOTH
single proposers. Data: per_step_graft_bfcl.jsonl (14B/BFCLv4 graft sweep).
"""
import json, pathlib

cells = []
def md(src):  cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
def code(src): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src})

md("""# Anchor-depth survival analysis — independent single-depth grafts

How much does the **Extension (hybrid)** beat the single proposers, **per anchor
depth**, when the suffix is grafted at only that one depth? For each anchor depth
`d` we run an **independent simulation** (`extension_gd:D`) that grafts the suffix
**only** at depth `d` (d=0 = **root hybrid**, the EAGLE/suffix trees merged at the
root) and measure its survival over **all** steps. Each panel overlays:

* **EAGLE3/MTP only** (blue), **Suffix only** (orange) — the standalone singles.
* **graft@d** (purple) — backbone + suffix grafted only at depth `d`.
* **full extension** (grey dashed) — suffix grafted at every depth (reference).

The greedy tree walk takes the per-step longest path, so a single-depth graft
behaves like `max(backbone, suffix)` on the steps it activates — **shallow grafts
beat BOTH singles**. Panel titles give the area-under-survival MAT and the gain
over each single.

**Data.** `run_tree_oracle_sim.py` with `SIM_FORCE_ADVANCE_1=1` (every prefix;
suffix cache fed one token) + `SIM_PER_STEP_JSONL`, methods `single:eagle3`,
`single:suffix:4.0:0.0`, `extension:4.0:0.0`, `extension_gd:0..8` and
`extension_cumd:0..8` (matched `F=4,T=0`), budget `B=128`. Collected as one slim
per-step dump per (model, workload) — **all 6 model×workload pairs**
(`per_step_alldepth_<model>_<wl>.jsonl`), run with `SIM_PARALLEL` for speed.""")

code("""import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path('/home/muchwater/advance-spec/simulation/results')
OUT_DIR = Path('/home/muchwater/advance-spec/simulation/notebooks/figures/anchor_depth')
OUT_DIR.mkdir(parents=True, exist_ok=True)

WORKLOADS = ['specbench', 'bfcl_v4', 'swebench_verified']
WORKLOAD_LABEL = {'specbench': 'SpecBench', 'bfcl_v4': 'BFCLv4',
                  'swebench_verified': 'SWE-Bench Verified'}
MODELS = {
    'qwen3_14b':  {'dir': ROOT / 'explorations_qwen3_14b'  / 'anchor_depth',
                   'title': 'Qwen3-14B',   'backbone_label': 'EAGLE3'},
    'qwen35_27b': {'dir': ROOT / 'explorations_qwen35_27b' / 'anchor_depth',
                   'title': 'Qwen3.5-27B', 'backbone_label': 'MTP'},
}
M_BACKBONE, M_SUFFIX, M_EXTENSION = 'single:eagle3', 'single:suffix:4.0:0.0', 'extension:4.0:0.0'
# Colors: EAGLE3/MTP=blue, Suffix=orange, Extension/graft=purple, full ext=grey.
C_BACKBONE, C_SUFFIX, C_EXTENSION, C_FULL = '#1f77b4', '#ff7f0e', '#7b1fa2', '#999999'
K_VALUES = list(range(9))
MAX_P = 24

# All-depth sweep dumps (singles + full extension + extension_gd:0..8 +
# extension_cumd:0..8) — one slim per-step file per (model, workload). The same
# file serves both the independent (gd) and cumulative (cumd) sections; the
# loader filters by method. Collected for all 6 model x workload pairs.
ALLDEPTH_DUMPS = {
    (mk, wl): mc['dir'] / f'per_step_alldepth_{mk}_{wl}.jsonl'
    for mk, mc in MODELS.items() for wl in WORKLOADS
}
GRAFT_DUMPS = ALLDEPTH_DUMPS
CUMD_DUMPS = ALLDEPTH_DUMPS

plt.rcParams.update({'axes.titlesize': 10, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8,
    'savefig.dpi': 130, 'figure.dpi': 110})""")

code("""# ── Loaders. survival(arr) = P(L>=p); MAT(A) = area under survival = mean
# accepted tokens (truncated at MAX_P).

def survival(arr):
    arr = np.asarray(arr); n = len(arr)
    if n == 0:
        return np.full(MAX_P + 1, np.nan)
    return np.array([(arr >= p).sum() / n for p in range(MAX_P + 1)])

def MAT(A):
    return float(A[1:].sum())


def load_methods(path, methods):
    \"\"\"Stream a per-step JSONL; return {method: np.array(accepted)} for the
    requested methods, plus N. Rows joined per step by (request_id,call_idx,
    step_id); only steps present for ALL requested methods are kept.\"\"\"
    if not Path(path).exists():
        return None, 0
    want = set(methods)
    rec = defaultdict(dict)
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = r['method']
            if m in want:
                rec[(r['request_id'], r.get('call_idx', 0), r['step_id'])][m] = \
                    int(r.get('accepted', 0))
    out = {m: [] for m in methods}
    N = 0
    for v in rec.values():
        if all(m in v for m in methods):
            N += 1
            for m in methods:
                out[m].append(v[m])
    return {m: np.asarray(a) for m, a in out.items()}, N


# Aggregate survival per (model, workload) from the MAIN per-step captures.
AGG = {}
for mk, mc in MODELS.items():
    AGG[mk] = {}
    for wl in WORKLOADS:
        d, N = load_methods(mc['dir'] / f'per_step_{wl}.jsonl',
                            [M_BACKBONE, M_SUFFIX, M_EXTENSION])
        AGG[mk][wl] = (d, N)
        print(f'  agg {mk}/{wl}: ' + ('MISSING' if d is None else f'N={N:,}'))""")

md("""## Overview — aggregate survival (marginal over all steps)

Standalone survival of each proposer (denominator = all steps). The full
Extension (all grafts) sits on/above the envelope of the two singles.""")
code("""fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey=True)
for ri, (mk, mc) in enumerate(MODELS.items()):
    for ci, wl in enumerate(WORKLOADS):
        ax = axes[ri][ci]
        d, N = AGG[mk][wl]
        if d is None:
            ax.text(0.5, 0.5, f'{WORKLOAD_LABEL[wl]}: missing', ha='center',
                    va='center', transform=ax.transAxes); continue
        xs = np.arange(MAX_P + 1)
        ax.plot(xs, survival(d[M_BACKBONE]), color=C_BACKBONE, lw=2, marker='o', ms=3, label=f'{mc["backbone_label"]} only')
        ax.plot(xs, survival(d[M_SUFFIX]),  color=C_SUFFIX, lw=2, marker='s', ms=3, label='Suffix only')
        ax.plot(xs, survival(d[M_EXTENSION]), color=C_EXTENSION, lw=2.4, marker='D', ms=3, label='Extension (all grafts)')
        ax.set_xlim(0, MAX_P); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
        ax.set_title(f'{mc["title"]} / {WORKLOAD_LABEL[wl]} (N={N:,})')
        if ci == 0: ax.set_ylabel('survival $A(p)$')
        if ri == 1: ax.set_xlabel('position $p$')
        if ri == 0 and ci == 0: ax.legend(loc='upper right')
fig.suptitle('Aggregate survival (marginal, denom = all steps)', y=1.01, fontsize=14)
plt.tight_layout(); plt.savefig(OUT_DIR / 'overview_aggregate_survival.png', bbox_inches='tight'); plt.show()""")

md("""## Independent single-depth grafts — all model × workload

One panel per anchor depth `d`. Purple = `graft@d` (suffix grafted only at depth
`d`), vs EAGLE3-only (blue), Suffix-only (orange), full extension (grey dashed).
Purple fill marks where `graft@d` exceeds the better single. Title flags whether
it beats BOTH singles.""")
code("""GRAFT_METHODS = [M_BACKBONE, M_SUFFIX, M_EXTENSION] + [f'extension_gd:{d}:4.0:0.0' for d in K_VALUES]

def plot_graft_indep(model_key, wl):
    mc = MODELS[model_key]; bb = mc['backbone_label']
    path = GRAFT_DUMPS.get((model_key, wl))
    if path is None:
        print(f'{model_key}/{wl}: no graft sweep'); return None
    d, N = load_methods(path, GRAFT_METHODS)
    if d is None:
        print(f'{model_key}/{wl}: dump missing ({path})'); return None
    xs = np.arange(MAX_P + 1)
    Ae, As, Ax = survival(d[M_BACKBONE]), survival(d[M_SUFFIX]), survival(d[M_EXTENSION])
    me, ms = MAT(Ae), MAT(As)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=True)
    rows = []
    for gd in K_VALUES:
        ax = axes[gd // 3][gd % 3]
        Ag = survival(d[f'extension_gd:{gd}:4.0:0.0']); mg = MAT(Ag)
        rows.append({'graft_depth': gd, 'MAT_graft': mg,
                     f'vs_{bb}': mg - me, 'vs_Suffix': mg - ms})
        ax.fill_between(xs, 0, Ag, color=C_EXTENSION, alpha=0.13, zorder=0)   # area = MAT
        ax.plot(xs, Ax, color=C_FULL, lw=1.2, ls='--', label=f'full ext (area {mx:.2f})')
        ax.plot(xs, Ae, color=C_BACKBONE, lw=1.8, marker='o', ms=2.5, label=f'{bb} (area {me:.2f})')
        ax.plot(xs, As, color=C_SUFFIX, lw=1.8, marker='s', ms=2.5, label=f'Suffix (area {ms:.2f})')
        title = 'root hybrid (graft@0)' if gd == 0 else f'graft @ anchor depth {gd}'
        ax.plot(xs, Ag, color=C_EXTENSION, lw=2.6, marker='D', ms=2.5, label=f'{title} (area {mg:.2f})')
        beats = 'beats BOTH' if (mg > me and mg > ms) else (f'>{bb} only' if mg > me else '-')
        ax.text(0.97, 0.55, f'area={mg:.2f}\\n vs{bb} +{mg-me:.2f}\\n vsSfx {mg-ms:+.2f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', fc='white', ec=C_EXTENSION, alpha=0.85))
        ax.set_title(f'{title}   [{beats}]', fontsize=10)
        ax.set_xlim(0, MAX_P); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
        if gd % 3 == 0: ax.set_ylabel('survival $A(p)$')
        if gd // 3 == 2: ax.set_xlabel('position $p$')
        if gd == 0: ax.legend(loc='upper right')
    fig.suptitle(f'{mc["title"]} / {WORKLOAD_LABEL[wl]} (N={N:,}) — independent single-depth '
                 f'graft sims: graft@d vs {bb}-only / Suffix-only '
                 f'({bb}={me:.2f}, Suffix={ms:.2f}, full ext={MAT(Ax):.2f})', y=1.005, fontsize=12)
    plt.tight_layout()
    fname = f'{model_key}_{wl}_graft_indep.png'
    plt.savefig(OUT_DIR / fname, bbox_inches='tight')
    plt.show(); print('saved', OUT_DIR / fname)
    return pd.DataFrame(rows).set_index('graft_depth')

for _mk in MODELS:
    for _wl in WORKLOADS:
        _t = plot_graft_indep(_mk, _wl)
        if _t is not None:
            print(f'== {MODELS[_mk]["title"]} / {WORKLOAD_LABEL[_wl]} =='); print(_t.round(3).to_string())""")

md("""## Read-off — graft@d MAT and gains (all model x workload)

`MAT_graft` is the area under each `graft@d` survival curve (mean accepted
tokens). `vs_EAGLE3`/`vs_MTP` and `vs_Suffix` are the gains over each single
proposer — positive for both means the single-depth hybrid beats both. (Tables
are printed alongside the figures above.)""")
code("""print('see per-workload tables above')
if False:
else:
    print('No graft sweep loaded.')""")

md("""## Cumulative grafts (version 2) — all model × workload

Single-depth graft@d loses to Suffix when the backbone never reaches depth `d`
(no suffix fallback there). **Cumulative** `cum≤d` grafts the suffix at **all**
anchor depths `0..d` — so it always includes the root graft (suffix-from-root
fallback for every step) and therefore **always beats both singles**, climbing
monotonically from the root hybrid (`d=0`) to the full extension (`d=8`).
One panel per cumulative depth `d`: `cum≤d` (purple) vs EAGLE3-only (blue),
Suffix-only (orange), full extension (grey dashed) — every panel beats BOTH and
climbs toward the full extension.""")
code("""CUMD_METHODS = [M_BACKBONE, M_SUFFIX, M_EXTENSION] + [f'extension_cumd:{d}:4.0:0.0' for d in K_VALUES]

def plot_cumulative(model_key, wl):
    mc = MODELS[model_key]; bb = mc['backbone_label']
    path = CUMD_DUMPS.get((model_key, wl))
    if path is None:
        print(f'{model_key}/{wl}: no cumulative sweep'); return None
    d, N = load_methods(path, CUMD_METHODS)
    if d is None:
        print(f'{model_key}/{wl}: dump missing ({path})'); return None
    xs = np.arange(MAX_P + 1)
    Ae, As, Ax = survival(d[M_BACKBONE]), survival(d[M_SUFFIX]), survival(d[M_EXTENSION])
    me, ms, mx = MAT(Ae), MAT(As), MAT(Ax)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=True)
    rows = []
    for gd in K_VALUES:                       # one panel per cumulative depth
        ax = axes[gd // 3][gd % 3]
        Ag = survival(d[f'extension_cumd:{gd}:4.0:0.0']); mg = MAT(Ag)
        rows.append({'cum_depth': gd, 'MAT_cum': mg, f'vs_{bb}': mg - me, 'vs_Suffix': mg - ms})
        ax.fill_between(xs, 0, Ag, color=C_EXTENSION, alpha=0.13, zorder=0)   # area = MAT
        ax.plot(xs, Ax, color=C_FULL, lw=1.2, ls='--', label=f'full ext (area {mx:.2f})')
        ax.plot(xs, Ae, color=C_BACKBONE, lw=1.8, marker='o', ms=2.5, label=f'{bb} (area {me:.2f})')
        ax.plot(xs, As, color=C_SUFFIX, lw=1.8, marker='s', ms=2.5, label=f'Suffix (area {ms:.2f})')
        title = 'root hybrid (cum\\u22640)' if gd == 0 else f'cumulative \\u2264depth {gd}'
        ax.plot(xs, Ag, color=C_EXTENSION, lw=2.6, marker='D', ms=2.5, label=f'{title} (area {mg:.2f})')
        beats = 'beats BOTH' if (mg > me and mg > ms) else (f'>{bb} only' if mg > me else '-')
        ax.text(0.97, 0.55, f'area={mg:.2f}\\n vs{bb} +{mg-me:.2f}\\n vsSfx {mg-ms:+.2f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', fc='white', ec=C_EXTENSION, alpha=0.85))
        ax.set_title(f'{title}   [{beats}]', fontsize=10)
        ax.set_xlim(0, MAX_P); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
        if gd % 3 == 0: ax.set_ylabel('survival $A(p)$')
        if gd // 3 == 2: ax.set_xlabel('position $p$')
        if gd == 0: ax.legend(loc='upper right')
    fig.suptitle(f'{mc["title"]} / {WORKLOAD_LABEL[wl]} (N={N:,}) — cumulative graft \\u2264depth d (per-depth): '
                 f'cum\\u2264d vs {bb}/Suffix ({bb}={me:.2f}, Suffix={ms:.2f}, full ext={mx:.2f}); '
                 f'always beats both, climbs to full ext', y=1.005, fontsize=11)
    plt.tight_layout()
    fname = f'{model_key}_{wl}_cumulative.png'
    plt.savefig(OUT_DIR / fname, bbox_inches='tight'); plt.show(); print('saved', OUT_DIR / fname)
    import pandas as pd
    return pd.DataFrame(rows).set_index('cum_depth')

for _mk in MODELS:
    for _wl in WORKLOADS:
        _t = plot_cumulative(_mk, _wl)
        if _t is not None:
            print(f'== {MODELS[_mk]["title"]} / {WORKLOAD_LABEL[_wl]} =='); print(_t.round(3).to_string())""")

md("""## Conditional accept rate — `a_p = A(p)/A(p-1)` (same per-depth layout)

Per-position accept rate given the previous position was accepted, for the
independent grafts and the cumulative grafts. The **curve shape** is the point
(where acceptance drops — anchor handoff / backbone exhaustion); the **area under
the conditional curve is NOT meaningful** (survival is the *product* of
conditionals, not the sum), so we do not shade it. The `MAT` label is the
**survival** area (`Σ_p A(p)` = mean accepted tokens), the actual metric.""")
code("""def cond_rate(a):
    a = np.asarray(a)
    cnt = np.array([(a >= p).sum() for p in range(MAX_P + 1)], float)
    c = np.full(MAX_P + 1, np.nan); c[0] = 1.0
    for p in range(1, MAX_P + 1):
        if cnt[p - 1] >= 30:          # support gate: trust a_p only with >=30 steps
            c[p] = cnt[p] / cnt[p - 1]
    return c

def plot_cond_grid(model_key, wl, dumps, prefix, kind, label_fn, fname):
    mc = MODELS[model_key]; bb = mc['backbone_label']
    path = dumps.get((model_key, wl))
    if path is None:
        print(f'{model_key}/{wl}: no sweep'); return
    methods = [M_BACKBONE, M_SUFFIX, M_EXTENSION] + [f'{prefix}:{gd}:4.0:0.0' for gd in K_VALUES]
    d, N = load_methods(path, methods)
    if d is None:
        print(f'{model_key}/{wl}: dump missing'); return
    xs = np.arange(MAX_P + 1)
    Ce, Cs, Cx = cond_rate(d[M_BACKBONE]), cond_rate(d[M_SUFFIX]), cond_rate(d[M_EXTENSION])
    me, ms = MAT(survival(d[M_BACKBONE])), MAT(survival(d[M_SUFFIX]))
    fig, axes = plt.subplots(3, 3, figsize=(17, 13), sharex=True, sharey=True)
    for gd in K_VALUES:
        ax = axes[gd // 3][gd % 3]
        g = d[f'{prefix}:{gd}:4.0:0.0']; Cg = cond_rate(g); mg = MAT(survival(g))
        ax.plot(xs, Cx, color=C_FULL, lw=1.2, ls='--', label='full ext')
        ax.plot(xs, Ce, color=C_BACKBONE, lw=1.8, marker='o', ms=2.5, label=f'{bb}')
        ax.plot(xs, Cs, color=C_SUFFIX, lw=1.8, marker='s', ms=2.5, label='Suffix')
        ax.plot(xs, Cg, color=C_EXTENSION, lw=2.6, marker='D', ms=2.5, label=label_fn(gd))
        ax.text(0.97, 0.30, f'MAT={mg:.2f}\\n(survival area)\\n vs{bb} +{mg-me:.2f}\\n vsSfx {mg-ms:+.2f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9.5, fontweight='bold',
                bbox=dict(boxstyle='round', fc='white', ec=C_EXTENSION, alpha=0.85))
        ax.set_title(label_fn(gd), fontsize=10)
        ax.set_xlim(0, MAX_P); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
        if gd % 3 == 0: ax.set_ylabel('conditional accept $a_p$')
        if gd // 3 == 2: ax.set_xlabel('position $p$')
        if gd == 0: ax.legend(loc='lower left', fontsize=8)
    fig.suptitle(f'{mc["title"]} / {WORKLOAD_LABEL[wl]} (N={N:,}) — conditional accept $a_p$ ({kind}); '
                 f'curve shape = where accept drops; MAT label = survival area (the metric)', y=1.005, fontsize=11)
    plt.tight_layout(); plt.savefig(OUT_DIR / fname, bbox_inches='tight'); plt.show(); print('saved', OUT_DIR / fname)

for _mk in MODELS:
    for _wl in WORKLOADS:
        plot_cond_grid(_mk, _wl, GRAFT_DUMPS, 'extension_gd', 'independent graft@d',
                       lambda x: ('root hybrid (graft@0)' if x == 0 else f'graft@d={x}'),
                       f'{_mk}_{_wl}_graft_indep_cond.png')
        plot_cond_grid(_mk, _wl, CUMD_DUMPS, 'extension_cumd', 'cumulative <=depth d',
                       lambda x: ('root hybrid (cum<=0)' if x == 0 else f'cumulative <=depth {x}'),
                       f'{_mk}_{_wl}_cumulative_cond.png')""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.8"}},
      "nbformat": 4, "nbformat_minor": 5}
out = pathlib.Path('/home/muchwater/advance-spec/simulation/notebooks/anchor_depth_analysis.ipynb')
out.write_text(json.dumps(nb, indent=1))
print('wrote', out, 'with', len(cells), 'cells')
