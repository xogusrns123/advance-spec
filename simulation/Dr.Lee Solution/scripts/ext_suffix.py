"""compCONF controller — single source of the n_head decision rule.

controller.md §4 names ``ext_suffix.py::_adaptive_nhead_conf`` as the ONE place
the head-length rule lives (``fusion_tree.adaptive_nhead`` just forwards to it,
to prevent copy drift). This reconstructs it from the spec.

Deployed rule (composition.md slide 13b) — argmax of the total round value:

    a_k  = clip(0.69·conf_k + 0.29, 0, 1)        # DFlash conf -> per-pos accept (r=0.99 fit)
    S_k  = Π_{i≤k} a_i        (S_0 = 1)           # head survives to depth k
    G_k  = Σ_{i=1}^{k} S_i    (G_0 = 0)           # expected accepted head tokens
    K(k) = 1 + G_k + S_k·T                        # bonus + head + gated tail
    n_head* = argmax_k K(k),   k ∈ [0, W]

*Formula reconciliation:* controller.md §4's pseudocode line writes
``G = [0.0] + cumsum(S)`` with ``S = [1.0] + cumprod(a)``, which would make
``G_k = 1 + Σ_{i=1}^{k-1} S_i`` — inconsistent with (a) slide 9's worked
numeric check (k=3, a=[.715,.671,.640] → S=[.715,.480,.307], G_3 = .715+.480+
.307 = 1.502) and (b) slide 13b's explicit ``G_k = Σ_{i≤k} S_i``. We implement
the worked-example / slide-13b definition ``G_k = Σ_{i=1}^{k} S_i =
cumsum(cumprod(a))`` (verified against the 1.502 check in ``_selftest``).

k=0 gives K(0) = 1 + 0 + 1·T = 1 + T (pure suffix). Large T (warm) pushes the
S_k·T term to prefer small k (short head); T≈0 (cold) makes G_k dominate → long
head. No separate cold gate is needed: at T≈0 the argmax naturally returns the
full head.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

# conf -> accept-probability affine fit (composition.md slide 18, r=0.99).
A_SLOPE = 0.69
A_BIAS = 0.29


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _round_values(conf: Sequence[float], T: float, W: int) -> List[float]:
    """K(k) for k = 0..W from the head confidences and tail warmth T."""
    a = [_clip01(A_SLOPE * float(conf[j]) + A_BIAS) for j in range(W)]
    # S[k] = S_k (survival to depth k), S[0] = 1.
    S = [1.0]
    for aj in a:
        S.append(S[-1] * aj)
    # G[k] = sum_{i=1}^k S_i  (cumulative survival = expected head accepts).
    G = [0.0]
    for k in range(1, W + 1):
        G.append(G[-1] + S[k])
    return [1.0 + G[k] + S[k] * float(T) for k in range(W + 1)]


def _adaptive_nhead_conf(conf: Sequence[float], T: float, num_spec: int) -> int:
    """compCONF: n_head* = argmax_k [1 + G_k + S_k·T].  Head capped at the
    DFlash draft horizon W = min(num_spec, len(conf))."""
    W = min(int(num_spec), len(conf))
    if W <= 0:
        return 0
    K = _round_values(conf, T, W)
    best_k, best_v = 0, K[0]
    for k in range(1, W + 1):
        if K[k] > best_v:            # strict > → ties resolve to the SHORTER head
            best_v, best_k = K[k], k
    return best_k


def _adaptive_nhead(T: float, num_spec: int) -> int:
    """Fallback when DFlash conf capture failed (controller.md §4: "T만, conf
    없음 → head를 길게 뽑는 경향"). With no per-position confidence we assume an
    optimistic flat survival (a_k = 1), so K(k) = 1 + k + T is increasing in k
    → returns the full head (long-head bias)."""
    return int(num_spec)


def adaptive_nhead(conf: Optional[Sequence[float]], T: float, num_spec: int) -> int:
    """Public entry used by ``fusion_tree.adaptive_nhead`` / the round loop.
    conf-based (compCONF) is the primary path; conf=None triggers the
    T-only fallback."""
    if conf is None:
        return _adaptive_nhead(T, num_spec)
    return _adaptive_nhead_conf(conf, T, num_spec)


def _selftest() -> None:
    # slide 9 worked example: k=3, a=[.715,.671,.640] -> G_3 = 1.502.
    conf = [(0.715 - A_BIAS) / A_SLOPE, (0.671 - A_BIAS) / A_SLOPE,
            (0.640 - A_BIAS) / A_SLOPE]
    K = _round_values(conf, T=0.0, W=3)
    G3 = K[3] - 1.0  # T=0 → K(3) = 1 + G_3
    assert abs(G3 - 1.502) < 1e-3, f"G_3={G3} != 1.502"
    # warm T pushes to short head, cold T=0 → long head.
    warm = _adaptive_nhead_conf([0.6] * 15, T=9.8, num_spec=15)
    cold = _adaptive_nhead_conf([0.6] * 15, T=0.0, num_spec=15)
    assert warm < cold, (warm, cold)
    print(f"[ext_suffix selftest] G_3={G3:.3f} (want 1.502)  warm_k={warm}  cold_k={cold}  OK")


if __name__ == "__main__":
    _selftest()
