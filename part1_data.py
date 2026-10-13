import numpy as np
import pandas as pd
from litho_setup import *


def load_raw_sheets(path=DATA_PATH):
    """data_raw.xlsx: one sheet per well. sheet_name=None reads all sheets
    (the default would silently read only the first)."""
    sheets = pd.read_excel(path, sheet_name=None)
    need = [LITHO_COL] + ALL_COLS
    for name, df in sheets.items():
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise KeyError(f"Sheet '{name}' lacks columns: {miss}")
    return sheets


def elbow_threshold(class_counts, step=ELBOW_STEP):
    """Classes sorted by total samples; threshold = midpoint of the largest relative
    drop between consecutive classes, rounded to `step`."""
    sizes = class_counts.sum(axis=1).sort_values(ascending=False).to_numpy()
    drop = (sizes[:-1] - sizes[1:]) / sizes[:-1]
    i = int(np.argmax(drop))
    return int(round((sizes[i] + sizes[i + 1]) / 2 / step) * step)


def preprocess(sheets):
    """Raw sheets -> listwise deletion per well -> keep classes with > elbow threshold
    samples AND present in >= MIN_WELLS wells. Returns one table with a WELL column."""
    need = [LITHO_COL] + ALL_COLS
    complete = {w: df.dropna(subset=need) for w, df in sheets.items()}
    counts = pd.DataFrame({w: df[LITHO_COL].value_counts() for w, df in complete.items()}).fillna(0)
    thr = elbow_threshold(counts)
    keep = counts.index[(counts.sum(axis=1) > thr) & ((counts > 0).sum(axis=1) >= MIN_WELLS)]
    df = pd.concat([d[d[LITHO_COL].isin(keep)].assign(**{WELL_COL: w}) for w, d in complete.items()],
                   ignore_index=True)
    print(f"[data] raw {sum(map(len, sheets.values()))} rows -> listwise "
          f"{sum(map(len, complete.values()))} -> class filter (>{thr} samples, >={MIN_WELLS} wells) "
          f"{len(df)} rows, {df[WELL_COL].nunique()} wells, {len(keep)} classes: {sorted(keep)}")
    return df


def load_data():
    df = preprocess(load_raw_sheets())
    if EXPECTED_SHAPE is not None:
        got = (len(df), df[WELL_COL].nunique(), df[LITHO_COL].nunique())
        assert got == tuple(EXPECTED_SHAPE), f"Filtered data {got} != EXPECTED_SHAPE {EXPECTED_SHAPE}"
    for c in LOG10_COLS:
        if (df[c] <= 0).any():
            raise ValueError(f"{c} has non-positive values; cannot log10")
        df[c] = np.log10(df[c])
    return df


def to_original(X):
    """Inverse log10 for LOG10_COLS. X: (n, len(ALL_COLS)) in original-scale-after-log."""
    X = X.copy()
    for c in LOG10_COLS:
        j = ALL_COLS.index(c)
        X[:, j] = 10 ** X[:, j]
    return X
