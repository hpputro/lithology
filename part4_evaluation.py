import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from litho_setup import *


def metrics_by(pred, keys, ranges):
    """Pooled residual metrics grouped by keys (last key must be 'log')."""
    rows = []
    for name, g in pred.groupby(keys):
        name = name if isinstance(name, tuple) else (name,)
        e = g["y_pred"].to_numpy() - g["y_true"].to_numpy()
        rmse = np.sqrt((e ** 2).mean())
        ss = ((g["y_true"] - g["y_true"].mean()) ** 2).sum()
        rows.append(dict(zip(keys, name)) | {
            "RMSE": rmse, "MAE": np.abs(e).mean(),
            "nRMSE": rmse / ranges[g["log"].iloc[0]],
            "R2": 1 - (e ** 2).sum() / ss if ss > 0 else np.nan, "n": len(g)})
    return pd.DataFrame(rows)


def mean_sd(df, keys, cols):
    agg = df.groupby(keys)[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


def make_tables(pred, ranges, out):
    # Table II: log x method (pooled per seed, mean +- sd across seeds)
    per_seed = metrics_by(pred, ["seed", "method", "log"], ranges)
    t2 = mean_sd(per_seed, ["method", "log"], ["RMSE", "MAE", "nRMSE", "R2"])
    t2.to_csv(out / "table_II_log_x_method.csv", index=False)

    # Table III: k x method (nRMSE per log, averaged over logs, then mean +- sd across seeds)
    ks = metrics_by(pred, ["seed", "method", "k", "log"], ranges)
    ks = ks.groupby(["seed", "method", "k"], as_index=False)["nRMSE"].mean()
    t3 = mean_sd(ks, ["method", "k"], ["nRMSE"])
    t3.to_csv(out / "table_III_k_x_method.csv", index=False)

    # Per-well breakdown (optional)
    pw = metrics_by(pred, ["seed", "method", "test_well", "log"], ranges)
    pw = mean_sd(pw, ["method", "test_well", "log"], ["RMSE", "nRMSE", "R2"])
    pw.to_csv(out / "per_well.csv", index=False)
    return t2, t3, pw


def parity_plot(pred, out, method="LGBM-Direct", seed=FIG_SEED):
    d = pred[(pred.method == method) & (pred.seed == seed)]
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(11, 3.6))
    for ax, t in zip(axes, TARGETS):
        g = d[d["log"] == t]
        ax.scatter(g.y_true, g.y_pred, s=4, alpha=0.4)
        lo, hi = min(g.y_true.min(), g.y_pred.min()), max(g.y_true.max(), g.y_pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        if t in LOG10_COLS:
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(t); ax.set_xlabel("True"); ax.set_ylabel("Imputed")
    fig.tight_layout()
    fig.savefig(out / f"fig_parity_{method}.png", dpi=300)
    plt.close(fig)
