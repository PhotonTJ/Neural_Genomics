"""Dynamic time warping over depth profiles, with the constraints the paper specifies.

Two things matter and are easy to get wrong:

  * a low cost bought by aggressive re-indexing is not stability, so we always return the
    warping extent (mean |i-j| along the path) alongside the cost;
  * DTW costs are only comparable between trajectories normalised together. `normalise`
    is applied to the *stacked* set, never per-curve, and `pairwise_matrix` takes the
    whole set at once for that reason.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- prep
def resample(profile, n):
    """Linear interpolation onto a common grid of n points over relative depth."""
    profile = np.asarray(profile, dtype=float)
    src = np.linspace(0.0, 1.0, len(profile))
    dst = np.linspace(0.0, 1.0, n)
    return np.interp(dst, src, profile)


def stack_triad(triads, n=32):
    """[{kappa,length,belief}, ...] -> array (n_models, n, 3), depth-resampled."""
    rows = []
    for t in triads:
        rows.append(np.stack([resample(t["kappa"], n),
                              resample(t["length"], n),
                              resample(t["belief"], n)], -1))
    return np.stack(rows, 0)


def normalise_set(X, mode="minmax"):
    """Normalise a whole set jointly, per channel. X: (n_models, n, 3)."""
    if mode == "none":
        return X.copy()
    Y = X.astype(float).copy()
    for c in range(Y.shape[-1]):
        ch = Y[..., c]
        if mode == "minmax":
            lo, hi = np.nanmin(ch), np.nanmax(ch)
            Y[..., c] = (ch - lo) / (hi - lo) if hi > lo else 0.0
        elif mode == "median":
            m = np.nanmedian(np.abs(ch))
            Y[..., c] = ch / m if m > 0 else ch
        else:
            raise ValueError(f"unknown normalisation {mode!r}")
    return Y


# --------------------------------------------------------------------------- dtw
def dtw(a, b, band=0.2):
    """Sakoe-Chiba banded DTW. a: (n, d), b: (m, d).

    Returns (normalised_cost, warping_extent, path).
    """
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    n, m = len(a), len(b)
    w = max(int(round(band * max(n, m))), abs(n - m)) if band else max(n, m)

    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = max(1, i - w), min(m, i + w)
        for j in range(lo, hi + 1):
            c = float(np.sum((a[i - 1] - b[j - 1]) ** 2))
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    # backtrack
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = int(np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]))
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    if not path:
        return float("inf"), float("inf"), []
    cost = sum(float(np.sum((a[i] - b[j]) ** 2)) for i, j in path) / len(path)
    extent = float(np.mean([abs(i - j) for i, j in path]))
    return cost, extent, path


def pairwise_matrix(X, band=0.2, normalise="minmax"):
    """Jointly normalise a set, then all-pairs DTW.

    X: (n_models, n, d) -> (cost, extent) each (n_models, n_models).
    """
    Y = normalise_set(X, normalise)
    k = len(Y)
    C = np.zeros((k, k))
    E = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            c, e, _ = dtw(Y[i], Y[j], band=band)
            C[i, j] = C[j, i] = c
            E[i, j] = E[j, i] = e
    return C, E


def offdiag(M):
    k = len(M)
    return np.array([M[i, j] for i in range(k) for j in range(k) if i != j])


def threshold(M, q=0.90):
    """Acceptance threshold: upper quantile of the off-diagonal distribution."""
    v = offdiag(M)
    return float(np.quantile(v, q)) if v.size else float("nan")
