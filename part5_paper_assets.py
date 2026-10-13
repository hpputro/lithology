"""Paper tables (NOTES section 5) from the raw data and from results/ of run_experiment.py.
Run after run_experiment.py. Output: results/paper/*.csv + paper_assets.xlsx + fig_pipeline_draft.png + fig_well_log_track.png + fig_masking_track_seed*.png."""
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from litho_setup import *
from part1_data import load_raw_sheets, preprocess
from part2_masking import mnar_mask

PAPER_DIR = OUT_DIR / "paper"
DEPTH_COL = "DEPTH"
RAW_LOGS = PREDICTORS + TARGETS
METHOD_ORDER = ["Mean", "MICE", "LGBM"]
RMSE_DEC = {"RS": 1, "NPHI": 2, "RHOB": 3}   # decimals in original units


def pm(mean, sd, dec):
    return f"{mean:.{dec}f} ± {sd:.{dec}f}"


def short_well(w):
    return w.split("-", 1)[-1]   # "Well 1-Kronos1" -> "Kronos1"


# ---------- data section ----------
def table_I(df):
    """Per well: samples, depth range, samples per lithology (filtered data, n = 6,811)."""
    ct = pd.crosstab(df[WELL_COL], df[LITHO_COL])
    g = df.groupby(WELL_COL)[DEPTH_COL]
    out = pd.DataFrame({"Well": [short_well(w) for w in ct.index],
                        "Depth (m)": [f"{a:.1f}–{b:.1f}" for a, b in zip(g.min(), g.max())],
                        "N": ct.sum(axis=1).to_numpy()})
    out = pd.concat([out, ct.reset_index(drop=True)], axis=1)
    total = {"Well": "Total", "Depth (m)": f"{df[DEPTH_COL].min():.1f}–{df[DEPTH_COL].max():.1f}",
             "N": len(df), **ct.sum().to_dict()}
    return pd.concat([out, pd.DataFrame([total])], ignore_index=True)


def raw_missing(sheets):
    """Pooled missing rate per log in the raw data (all sheets)."""
    raw = pd.concat(sheets.values(), ignore_index=True)
    n = raw[RAW_LOGS].isna().sum()
    return pd.DataFrame({"Log": RAW_LOGS, "N missing": n.to_numpy(),
                         "Missing (%)": (n / len(raw) * 100).round(2).to_numpy(), "N total": len(raw)})


def raw_comissing(sheets):
    """P(column log missing | row log missing), %, pooled raw data. Read row by row."""
    raw = pd.concat(sheets.values(), ignore_index=True)
    m = raw[RAW_LOGS].isna().to_numpy().astype(int)
    both = m.T @ m
    out = pd.DataFrame(both / np.diag(both)[:, None] * 100, index=RAW_LOGS, columns=RAW_LOGS).round(2)
    out.index.name = "Missing log"
    return out.reset_index()


# ---------- results section ----------
def mask_summary(mstats, pred, n_per_well):
    """IV.A: share of rows with k co-missing targets (pooled over test wells, weighted by rows),
    mean ± sd across seeds; reference = independent masking at MASK_LEVEL; masked cells per log."""
    ks = [c for c in mstats.columns if c.startswith("k=")]
    m = mstats.assign(n=mstats["test_well"].map(n_per_well))
    per_seed = m.groupby("seed")[ks].apply(lambda g: g.mul(m.loc[g.index, "n"], axis=0).sum()
                                           / m.loc[g.index, "n"].sum()) * 100
    p, T = MASK_LEVEL, len(TARGETS)
    indep = [100 * math.comb(T, k) * p ** k * (1 - p) ** (T - k) for k in range(T + 1)]
    rows = [{"k": k[2:], "Rows (%)": pm(per_seed[k].mean(), per_seed[k].std(), 1),
             "Independent masking (%)": round(indep[i], 1)} for i, k in enumerate(ks)]
    k_tab = pd.DataFrame(rows)

    cells = pred[pred["method"] == pred["method"].iloc[0]].groupby(["seed", "log"]).size().unstack()
    cell_tab = pd.DataFrame({"Log": TARGETS,
                             "Masked cells per seed": [pm(cells[t].mean(), cells[t].std(), 0) for t in TARGETS],
                             "Share of rows (%)": [round(cells[t].mean() / sum(n_per_well.values()) * 100, 1)
                                                   for t in TARGETS]})
    return k_tab, cell_tab


def table_II_fmt(t2):
    rows = []
    for log in TARGETS:
        for meth in METHOD_ORDER:
            r = t2[(t2["log"] == log) & (t2["method"] == meth)].iloc[0]
            rows.append({"Log": log, "Method": meth,
                         "RMSE": pm(r.RMSE_mean, r.RMSE_std, RMSE_DEC[log]),
                         "MAE": pm(r.MAE_mean, r.MAE_std, RMSE_DEC[log]),
                         "nRMSE": pm(r.nRMSE_mean, r.nRMSE_std, 3),
                         "R2": pm(r.R2_mean, r.R2_std, 2)})
    return pd.DataFrame(rows)


def table_III_fmt(t3):
    t3 = t3.assign(cell=[pm(a, b, 3) for a, b in zip(t3.nRMSE_mean, t3.nRMSE_std)])
    out = t3.pivot(index="method", columns="k", values="cell").reindex(METHOD_ORDER)
    out.columns = [f"k={k}" for k in out.columns]
    return out.reset_index().rename(columns={"method": "Method"})


def stability(pw, t2):
    """IV.D: spread across test wells (of seed-averaged metrics) and across seeds (pooled)."""
    rows = []
    for log in TARGETS:
        for meth in METHOD_ORDER:
            g = pw[(pw["log"] == log) & (pw["method"] == meth)]
            r = t2[(t2["log"] == log) & (t2["method"] == meth)].iloc[0]
            worst = g.loc[g["R2_mean"].idxmin(), "test_well"]
            rows.append({"Log": log, "Method": meth,
                         "R2 wells min": round(g.R2_mean.min(), 2), "R2 wells max": round(g.R2_mean.max(), 2),
                         "R2 sd across wells": round(g.R2_mean.std(), 2),
                         "Worst well": short_well(worst),
                         "nRMSE sd across wells": round(g.nRMSE_mean.std(), 3),
                         "nRMSE sd across seeds": round(r.nRMSE_std, 4),
                         "max nRMSE sd across seeds (per well)": round(g.nRMSE_std.max(), 4)})
    return pd.DataFrame(rows)


def per_well_fmt(pw):
    out = pw.assign(Well=pw["test_well"].map(short_well),
                    nRMSE=[pm(a, b, 3) for a, b in zip(pw.nRMSE_mean, pw.nRMSE_std)],
                    R2=[pm(a, b, 2) for a, b in zip(pw.R2_mean, pw.R2_std)])
    return out[["method", "log", "Well", "nRMSE", "R2"]].rename(columns={"method": "Method", "log": "Log"})


# ---------- data figure (as RINENG part1) ----------
TRACK_LOGS = ["GR", "RHOB", "NPHI", "DTCO", "DTSM", "RS", "RD"]


def well_log_track(df, path, mask=None, title="Well-Log Track"):
    """All wells stacked top to bottom (each sorted by depth), one panel per log + lithology.
    Vertical axis = sample order (rows removed by listwise deletion are skipped), not depth.
    mask: optional bool DataFrame (columns TARGETS, same index as df); masked samples are drawn
    as red dots on the target panels and a k panel (number of co-missing targets) is added."""
    d = df.copy()
    if mask is not None:
        d[[f"_m_{t}" for t in TARGETS]] = mask[TARGETS].to_numpy()
    d = d.sort_values([WELL_COL, DEPTH_COL]).reset_index(drop=True)
    n = len(d)
    idx = np.arange(n)
    starts = d[WELL_COL].ne(d[WELL_COL].shift()).to_numpy().nonzero()[0]
    bounds = list(starts) + [n]
    mids = [(bounds[i] + bounds[i + 1]) / 2 for i in range(len(starts))]

    extra = ["k"] if mask is not None else []
    ratios = [1] * len(TRACK_LOGS) + [0.35] * len(extra) + [1]
    fig, axes = plt.subplots(1, len(ratios), figsize=(2.2 * sum(ratios), 14), sharey=True,
                             gridspec_kw={"width_ratios": ratios})
    for ax, col in zip(axes[:len(TRACK_LOGS)], TRACK_LOGS):
        for s, e in zip(bounds[:-1], bounds[1:]):          # one line per well, no joins
            ax.plot(d[col].iloc[s:e], idx[s:e], lw=0.5, color="black")
        if mask is not None and col in TARGETS:
            m = d[f"_m_{col}"].to_numpy()
            ax.scatter(d[col].to_numpy()[m], idx[m], s=2, color="red", zorder=3, linewidths=0)
            col = f"{col} (masked {m.mean() * 100:.0f}%)"
        for b in starts[1:]:
            ax.axhline(b, color="red", lw=0.6, linestyle="--")
        ax.set_title(col, fontsize=9)
        ax.grid(alpha=0.3)

    if mask is not None:
        ax_k = axes[len(TRACK_LOGS)]
        k = d[[f"_m_{t}" for t in TARGETS]].sum(axis=1).to_numpy()
        kmap = matplotlib.colors.ListedColormap(["white", "#fcbba1", "#fb6a4a", "#a50f15"])
        ax_k.imshow(k.reshape(-1, 1), aspect="auto", cmap=kmap, vmin=-0.5, vmax=3.5,
                    extent=[0, 1, n - 0.5, -0.5], interpolation="nearest")
        for b in starts[1:]:
            ax_k.axhline(b, color="black", lw=0.8)
        ax_k.set_title("k", fontsize=9)
        ax_k.set_xticks([])

    ax_l = axes[-1]
    code = d[LITHO_COL].map({name: i for i, name in enumerate(LITHO_ORDER)})
    assert code.notna().all(), "lithology not in LITHO_ORDER"
    cmap = plt.get_cmap("tab10", len(LITHO_ORDER))
    ax_l.imshow(code.to_numpy().reshape(-1, 1), aspect="auto", cmap=cmap, vmin=0,
                vmax=len(LITHO_ORDER) - 1, extent=[0, 1, n - 0.5, -0.5], interpolation="nearest")
    for b in starts[1:]:
        ax_l.axhline(b, color="white", lw=1.2)
    ax_l.set_title(LITHO_COL, fontsize=9)
    ax_l.set_xticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(len(LITHO_ORDER))]
    labels = [f"{i + 1}. {name}" for i, name in enumerate(LITHO_ORDER)]
    if mask is not None:
        handles += [plt.Line2D([], [], ls="", marker="o", ms=3, color="red")] + \
                   [plt.Rectangle((0, 0), 1, 1, fc=kmap(i), ec="grey", lw=0.3) for i in range(4)]
        labels += ["masked sample"] + [f"k = {i}" for i in range(4)]
    ax_l.legend(handles, labels, fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    axes[0].set_ylim(n - 0.5, -0.5)                         # first well on top, depth downward
    axes[0].set_yticks(mids)
    axes[0].set_yticklabels(d[WELL_COL].iloc[starts].tolist(), fontsize=7)
    axes[0].set_ylabel("Well (sorted by depth within well)")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def test_masks(df, seed):
    """Test-well masks exactly as in run_experiment.py (each well masked on its own with
    seed + TEST_SEED_OFFSET). Every well is the test well once, so one seed covers all rows."""
    parts = []
    for w in sorted(df[WELL_COL].unique()):
        te = df[df[WELL_COL] == w]
        M = mnar_mask(te.reset_index(drop=True), seed + TEST_SEED_OFFSET)
        parts.append(pd.DataFrame(M, columns=TARGETS, index=te.index))
    return pd.concat(parts).loc[df.index]


# ---------- Fig. 1 draft ----------
def pipeline_draft(path):
    steps = ["Raw logs\n7 wells, 65,190 rows",
             "Listwise deletion +\nelbow class filter\n6 wells, 6 classes, 6,811",
             "LOWO split\n(6 folds)",
             "log10(RS),\nMinMax fit on train",
             f"MNAR self-masking\nRS, NPHI, RHOB\nq={MASK_Q}, {int(MASK_LEVEL * 100)}%, {len(MASK_SEEDS)} seeds",
             "Fit imputer on\nmasked train\nMean / MICE / LGBM",
             "Impute masked test,\ninverse MinMax, 10^RS",
             "Metrics on masked\ntest cells\nRMSE, MAE, nRMSE, R²"]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_xlim(0, 4); ax.set_ylim(0, 2); ax.axis("off")
    pos = [(i % 4 + 0.5, 1.5 if i < 4 else 0.5) for i in range(len(steps))]
    for (x, y), s in zip(pos, steps):
        ax.text(x, y, s, ha="center", va="center", fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.35", fc="#eef3f8", ec="#3b6ea5", lw=0.8))
    for (x0, y0), (x1, y1) in zip(pos[:-1], pos[1:]):
        if y0 == y1:
            ax.annotate("", xy=(x1 - 0.42, y1), xytext=(x0 + 0.42, y0), arrowprops=dict(arrowstyle="->", lw=0.8))
        else:   # wrap from end of row 1 to start of row 2
            ax.plot([x0, x0, x1], [y0 - 0.3, 1.0, 1.0], color="black", lw=0.8)
            ax.annotate("", xy=(x1, y1 + 0.3), xytext=(x1, 1.0), arrowprops=dict(arrowstyle="->", lw=0.8))
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    sheets = load_raw_sheets()
    df = preprocess(sheets)
    n_per_well = df[WELL_COL].value_counts().to_dict()

    t2 = pd.read_csv(OUT_DIR / "table_II_log_x_method.csv")
    t3 = pd.read_csv(OUT_DIR / "table_III_k_x_method.csv")
    pw = pd.read_csv(OUT_DIR / "per_well.csv")
    mstats = pd.read_csv(OUT_DIR / "mask_stats.csv")
    pred = pd.read_csv(OUT_DIR / "predictions_masked_cells.csv", usecols=["seed", "method", "log"])

    k_tab, cell_tab = mask_summary(mstats, pred, n_per_well)
    tables = {
        "table_I_dataset": table_I(df),
        "raw_missing": raw_missing(sheets),
        "raw_comissing": raw_comissing(sheets),
        "mask_k_share": k_tab,
        "mask_cells": cell_tab,
        "table_II_formatted": table_II_fmt(t2),
        "table_III_formatted": table_III_fmt(t3),
        "stability": stability(pw, t2),
        "per_well_formatted": per_well_fmt(pw),
    }
    with pd.ExcelWriter(PAPER_DIR / "paper_assets.xlsx") as xw:
        for name, t in tables.items():
            t.to_csv(PAPER_DIR / f"{name}.csv", index=False)
            t.to_excel(xw, sheet_name=name[:31], index=False)
    pipeline_draft(PAPER_DIR / "fig_pipeline_draft.png")
    well_log_track(df, PAPER_DIR / "fig_well_log_track.png")
    for seed in MASK_SEEDS:
        M = test_masks(df, seed)
        n_mask = pred[(pred["seed"] == seed) & (pred["method"] == pred["method"].iloc[0])].groupby("log").size()
        assert (M.sum() == n_mask.reindex(TARGETS)).all(), "masks differ from run_experiment"
        well_log_track(df, PAPER_DIR / f"fig_masking_track_seed{seed}.png", mask=M,
                       title=f"MNAR masking of test wells (seed {seed}, q={MASK_Q}, {int(MASK_LEVEL * 100)}% per log)")

    cm = tables["raw_comissing"].set_index("Missing log")
    print(f"Co-missing P(DTSM missing | DTCO missing) = {cm.loc['DTCO', 'DTSM']:.2f}%")
    k0 = k_tab.loc[k_tab["k"] == "0", "Rows (%)"].iloc[0]
    print(f"Listwise deletion on masked data keeps rows with k=0: {k0}%")
    for name, t in tables.items():
        if name not in ("per_well_formatted",):
            print(f"\n== {name}\n{t.to_string(index=False)}")
    print(f"\nSaved to {PAPER_DIR}/")


if __name__ == "__main__":
    main()
