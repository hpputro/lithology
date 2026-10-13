# Multivariate MNAR imputation of well logs (LOWO)

Targets (masked + imputed): RS, NPHI, RHOB. Predictors (always observed): GR, DTCO, DTSM, RD.
Methods: Mean, MICE, LightGBM (one regressor per target). Validation: leave-one-well-out.

## Setup
1. Put the raw logs in `data/data_raw.xlsx` (one sheet per well, sheet name = well).
2. Edit column names / LightGBM params in `litho_setup.py`.
3. `pip install -r requirements.txt && python run_experiment.py`

## Preprocessing (`part1_data.py`, same as RINENG part1)
Raw sheets -> listwise deletion per well (all logs + lithology) -> keep lithology classes with
more samples than the elbow threshold (midpoint of the largest drop in sorted class sizes,
rounded to 10; = 130) and present in >= 2 wells. Result: 6,811 rows, 6 wells, 6 classes
(checked against `EXPECTED_SHAPE`).

## Pipeline (per mask seed, per test well)
1. Split LOWO -> fit MinMax on train (before masking), RS is log10-transformed first.
2. MNAR self-masking, independent per log: per lithology, 30% of rows drawn from the
   tails (<= q or >= 1-q quantile, q=0.20). Train and test masks use independent RNG streams.
3. Fit imputers on masked train only; impute masked test.
4. Inverse MinMax, then `10**RS`; keep only masked test cells.

## Outputs (`results/`)
- `table_II_log_x_method.csv`: RMSE/MAE/nRMSE/R2 per log x method (mean, sd across seeds)
- `table_III_k_x_method.csv`: nRMSE by number of co-missing logs k (mean over logs)
- `per_well.csv`, `mask_stats.csv` (row share with k=0..3), `predictions_masked_cells.csv`
- `fig_parity_LGBM.png`

Metrics are pooled from raw residuals across folds (not averaged per fold); nRMSE = RMSE / dataset range (original units).
