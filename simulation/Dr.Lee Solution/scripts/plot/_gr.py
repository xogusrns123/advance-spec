"""Canonical G/R partition for the deck figures (Picture 1/2/3/5/9/10).

Redefinition (2026-07-10, user): a token is in **R (repeated)** iff it is covered
by a repeated n-gram chunk of >= L tokens (L=4) -- the union of [q, q+s[q]) over
every position q whose suffix copy depth s[q] >= L. Everything else is **G
(novel)**. This is MODEL-FREE: it uses only s (a pure repetition statistic of the
token stream: the longest match copyable from earlier context), never the DFlash
accept curve `a`. Replaces the earlier winner=argmax(s,a) / s>=1 / s>=4-start
definitions that these figures used before.
"""
from __future__ import annotations

L_DEFAULT = 4


def rg_cover(s, L=L_DEFAULT):
    """Per-position bool: True = R (inside a >= L repeated run), False = G.

    Mirrors plot_traj_warmcold.run_cover(s, L): the union of [q, q+s[q]) over q
    with s[q] >= L. s = per-position suffix copy depth along gt."""
    cov = [False] * len(s)
    for q, v in enumerate(s):
        if v >= L:
            for i in range(q, min(len(s), q + v)):
                cov[i] = True
    return cov


def gr_runs(cover):
    """Contiguous run lengths under the partition -> (G_runs, R_runs)."""
    G, R = [], []
    cur, ln = None, 0
    for c in list(cover) + [None]:
        if c == cur:
            ln += 1
        else:
            if cur is True:
                R.append(ln)
            elif cur is False:
                G.append(ln)
            cur, ln = c, 1
    return G, R


def n_boundaries(cover):
    """Number of R<->G transitions (winner-flip analogue on the new partition)."""
    return sum(1 for i in range(1, len(cover)) if cover[i] != cover[i - 1])
