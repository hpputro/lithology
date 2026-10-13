"""Run: LOWO x mask seeds x imputers -> tables + figure in results/."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from litho_setup import *
from part1_data import load_data, to_original
from part2_masking import mnar_mask, apply_mask, mask_stats
from part3_imputation import make_imputers
from part4_evaluation import make_tables, parity_plot


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    wells = sorted(df[WELL_COL].unique())
    print(f"{len(df)} rows, {len(wells)} wells: {wells}")

    full_orig = to_original(df[ALL_COLS].to_numpy())
    ranges = {t: np.ptp(full_orig[:, ALL_COLS.index(t)]) for t in TARGETS}

    records, mstats = [], []
    for seed in MASK_SEEDS:
        for w in wells:
            tr = df[df[WELL_COL] != w].reset_index(drop=True)
            te = df[df[WELL_COL] == w].reset_index(drop=True)

            sc = MinMaxScaler().fit(tr[ALL_COLS])               # fit on train, before masking
            Xtr, Xte = sc.transform(tr[ALL_COLS]), sc.transform(te[ALL_COLS])

            M_tr = mnar_mask(tr, seed)
            M_te = mnar_mask(te, seed + TEST_SEED_OFFSET)
            Xtr_m, Xte_m = apply_mask(Xtr, M_tr), apply_mask(Xte, M_te)
            mstats.append({"seed": seed, "test_well": w} | mask_stats(M_te))

            k = M_te.sum(1)
            rows, cols = np.where(M_te)                          # masked cells (row, target idx)
            y_true_all = to_original(sc.inverse_transform(Xte))

            for name, imp in make_imputers(seed).items():
                imp.fit(Xtr_m)                                   # train only
                Xhat = imp.transform(Xte_m)
                y_pred_all = to_original(sc.inverse_transform(Xhat))
                jj = N_PRED + cols
                records.append(pd.DataFrame({
                    "seed": seed, "method": name, "test_well": w,
                    "log": np.array(TARGETS)[cols], "litho": te[LITHO_COL].to_numpy()[rows],
                    "k": k[rows], "y_true": y_true_all[rows, jj], "y_pred": y_pred_all[rows, jj]}))
            print(f"seed={seed} test_well={w} done")

    pred = pd.concat(records, ignore_index=True)
    pred.to_csv(OUT_DIR / "predictions_masked_cells.csv", index=False)
    pd.DataFrame(mstats).to_csv(OUT_DIR / "mask_stats.csv", index=False)
    t2, t3, _ = make_tables(pred, ranges, OUT_DIR)
    parity_plot(pred, OUT_DIR)
    print(t2.round(4).to_string(index=False))
    print(t3.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
