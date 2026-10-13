import numpy as np
from litho_setup import *


def mnar_mask(df, seed, q=MASK_Q, level=MASK_LEVEL):
    """Componentwise MNAR self-masking. Returns bool (n, len(TARGETS)), True = masked.
    Per lithology and per log: candidates = tail values (<= q or >= 1-q quantile);
    `level` * n_rows_in_litho rows are drawn at random from candidates.
    Positional indexing only (no dependence on df.index)."""
    rng = np.random.default_rng(seed)
    litho = df[LITHO_COL].to_numpy()
    M = np.zeros((len(df), len(TARGETS)), dtype=bool)
    for j, t in enumerate(TARGETS):
        v = df[t].to_numpy()
        for lit in np.unique(litho):
            idx = np.flatnonzero(litho == lit)
            lo, hi = np.quantile(v[idx], [q, 1 - q])
            cand = idx[(v[idx] <= lo) | (v[idx] >= hi)]
            n_mask = min(int(round(level * len(idx))), len(cand))
            if n_mask > 0:
                M[rng.choice(cand, n_mask, replace=False), j] = True
    return M


def apply_mask(X, M):
    """X: (n, len(ALL_COLS)) array; masks target columns (the last len(TARGETS))."""
    Xm = X.copy()
    Xm[:, N_PRED:][M] = np.nan
    return Xm


def mask_stats(M):
    k = M.sum(1)
    return {f"k={i}": float((k == i).mean()) for i in range(M.shape[1] + 1)}
