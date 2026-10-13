import numpy as np
import pandas as pd
from litho_setup import *


def load_data():
    df = pd.read_excel(DATA_PATH) if str(DATA_PATH).endswith(("xlsx", "xls")) else pd.read_csv(DATA_PATH)
    need = [WELL_COL, LITHO_COL] + ALL_COLS
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"Missing columns in data: {miss}")
    n0 = len(df)
    df = df.dropna(subset=need)
    for c in LOG10_COLS:
        df = df[df[c] > 0]
    df = df.reset_index(drop=True)
    if len(df) != n0:
        print(f"[data] dropped {n0 - len(df)} incomplete/non-positive rows -> {len(df)} rows")
    for c in LOG10_COLS:
        df[c] = np.log10(df[c])
    return df


def to_original(X):
    """Inverse log10 for LOG10_COLS. X: (n, len(ALL_COLS)) in original-scale-after-log."""
    X = X.copy()
    for c in LOG10_COLS:
        j = ALL_COLS.index(c)
        X[:, j] = 10 ** X[:, j]
    return X
