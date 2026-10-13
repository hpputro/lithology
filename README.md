# Multivariate MNAR imputation of well logs (LOWO)

Targets (masked + imputed): RS, NPHI, RHOB. Predictors (always observed): GR, DTCO, DTSM, RD.
Methods: Mean, MICE, LightGBM (one regressor per target). Validation: leave-one-well-out.

## Setup
1. Put the data in `data/litho_data.csv` (or edit `DATA_PATH` in `litho_setup.py`).
2. Edit column names / LightGBM params in `litho_setup.py`.
3. `pip install -r requirements.txt && python run_experiment.py`

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
