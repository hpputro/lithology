"""Config: multivariate MNAR imputation of well logs (LOWO validation)."""
from pathlib import Path

DATA_PATH = Path("data/data_raw.xlsx")    # raw NOPIMS logs, one sheet per well (sheet name = well)
OUT_DIR = Path("results")

# Column names in the data file (EDIT if different)
WELL_COL = "WELL"                          # added from the sheet name
LITHO_COL = "LITHOLOGY"
TARGETS = ["RS", "NPHI", "RHOB"]           # logs to mask + impute
PREDICTORS = ["GR", "DTCO", "DTSM", "RD"]  # always observed (assumption/limitation)
ALL_COLS = PREDICTORS + TARGETS            # column order used in all arrays
N_PRED = len(PREDICTORS)
LOG10_COLS = ["RS"]                        # log10 before MinMax, 10**x on inverse

# Preprocessing (same as RINENG part1): listwise deletion -> elbow class filter
MIN_WELLS = 2        # a lithology class must occur in at least this many wells
ELBOW_STEP = 10      # elbow threshold = midpoint of the largest drop, rounded to this step
EXPECTED_SHAPE = (6811, 6, 6)   # (rows, wells, classes) after filtering; None to skip check

# MNAR self-masking (componentwise, per lithology, per log)
MASK_Q = 0.20        # tail = value <= q-quantile or >= (1-q)-quantile
MASK_LEVEL = 0.30    # fraction of rows (per litho) masked, drawn from tails
MASK_SEEDS = [0, 1, 2]
TEST_SEED_OFFSET = 10_000   # test-well mask uses seed + offset (independent of train)

# LightGBM (EDIT to match RINENG config)
LGBM_PARAMS = dict(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                   n_jobs=-1, verbose=-1)
EARLY_STOP = 50
