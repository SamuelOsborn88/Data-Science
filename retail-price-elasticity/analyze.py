"""Estimate price elasticity of demand for each brand from weekly retail data.

Two methods are compared:
  1. Naive week-over-week: %change in units / %change in price, averaged.
  2. Log-log regression:   ln(units) = brand intercept + week effect + elasticity * ln(price)
     The week effect absorbs seasonality and holidays shared by all brands, so the
     price coefficient isolates how volume responds to each brand's own price moves.

Because the data is synthetic with known true elasticities (see generate_data.py),
both methods can be scored against the truth.

Run:  python analyze.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "outputs"
FIG = HERE / "figures"
MIN_PRICE_MOVE = 0.01          # naive method: ignore weeks where price moved less than 1%
PRICE_TEST = 0.05              # revenue scenario: +5% price

# --- chart styling (fixed brand -> colour order, recessive axes) -----------------
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
GAIN, LOSS = "#2a78d6", "#e34948"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "legend.frameon": False,
})


def naive_elasticity(df):
    df = df.sort_values(["brand", "week"]).copy()
    g = df.groupby("brand")
    df["pct_price_change"] = g["price"].pct_change()
    df["pct_units_change"] = g["units_sold"].pct_change()
    moved = df["pct_price_change"].abs() >= MIN_PRICE_MOVE
    df["wow_elasticity"] = np.where(moved, df["pct_units_change"] / df["pct_price_change"], np.nan)
    return df


def regression_elasticity(df):
    """Pooled OLS with brand intercepts, week fixed effects and brand-specific price slopes."""
    brands = sorted(df["brand"].unique())
    weeks = sorted(df["week"].unique())
    y = np.log(df["units_sold"].to_numpy())
    cols, names = [], []
    for b in brands:
        is_b = (df["brand"] == b).to_numpy(float)
        cols += [is_b, is_b * np.log(df["price"].to_numpy())]
        names += [f"const:{b}", f"slope:{b}"]
    for w in weeks[1:]:                                   # first week is the baseline
        cols.append((df["week"] == w).to_numpy(float))
        names.append(f"week:{w}")
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    coef = dict(zip(names, beta))
    week_effect = {weeks[0]: 0.0, **{w: coef[f"week:{w}"] for w in weeks[1:]}}
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    rows = []
    for b in brands:
        i = names.index(f"slope:{b}")
        rows.append({"brand": b, "regression_elasticity": beta[i],
                     "ci_low": beta[i] - 1.96 * se[i], "ci_high": beta[i] + 1.96 * se[i],
                     "intercept": coef[f"const:{b}"]})
    return pd.DataFrame(rows), week_effect, r2


def classify(e):
    return "Elastic" if abs(e) > 1 else "Inelastic"


# --- figures ------------------------------------------------------------------------
def fig_weekly(df, brands):
    fig, (ax_p, ax_u) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for i, b in enumerate(brands):
        d = df[df["brand"] == b]
        ax_p.plot(d["week"], d["price"], color=SERIES[i], label=b, drawstyle="steps-post")
        ax_u.plot(d["week"], d["units_sold"], color=SERIES[i], label=b)
    ax_p.set_title("Shelf price by week", pad=30)
    ax_p.set_ylabel("Price ($)")
    ax_u.set_title("Units sold by week")
    ax_u.set_ylabel("Units")
    ax_p.legend(loc="lower left", ncol=4, fontsize=9, bbox_to_anchor=(0, 1.08), borderaxespad=0)
    ax_u.annotate("Promotions (price dips) line up with volume spikes;\n"
                  "the summer hump and July 4th peak are seasonal.",
                  xy=(0.01, 0.97), xycoords="axes fraction", va="top", fontsize=9, color=INK_2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "weekly_price_and_units.png", dpi=150)
    plt.close(fig)


def fig_loglog(df, summary, week_effect, brands):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, (ax, b) in enumerate(zip(axes.flat, brands)):
        d = df[df["brand"] == b]
        row = summary.set_index("brand").loc[b]
        adj_units = np.log(d["units_sold"]) - d["week"].map(week_effect)   # remove seasonality
        ax.scatter(np.log(d["price"]), adj_units, s=36, color=SERIES[i], marker=MARKERS[i],
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        xs = np.linspace(np.log(d["price"]).min(), np.log(d["price"]).max(), 20)
        ax.plot(xs, row["intercept"] + row["regression_elasticity"] * xs, color=INK_2, linewidth=1.5)
        ax.set_title(b, fontsize=11)
        ax.text(0.97, 0.95, f"elasticity {row['regression_elasticity']:.2f}\n(true {row['true_elasticity']:.1f})",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK_2)
        ax.set_xlabel("ln(price)")
        ax.set_ylabel("ln(units), seasonally adjusted")
    fig.suptitle("Log-log fit: the slope of each line is the brand's price elasticity",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "loglog_fit_by_brand.png", dpi=150)
    plt.close(fig)


def fig_compare(summary, brands):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    s = summary.set_index("brand").loc[brands[::-1]]
    y = np.arange(len(s))
    ax.axvline(-1, color=AXIS, linewidth=1)
    ax.text(-0.98, -0.55, "unit elastic (−1)", ha="left", va="center", fontsize=8, color=MUTED)
    ax.errorbar(s["regression_elasticity"], y + 0.12,
                xerr=[s["regression_elasticity"] - s["ci_low"], s["ci_high"] - s["regression_elasticity"]],
                fmt="o", color=SERIES[0], ecolor=SERIES[0], elinewidth=2, capsize=0, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label="Regression estimate (95% CI)", zorder=3)
    ax.scatter(s["naive_elasticity"], y - 0.12, marker="s", s=60, color=SERIES[1],
               edgecolor=SURFACE, linewidth=1.5, label="Naive week-over-week average", zorder=3)
    ax.scatter(s["true_elasticity"], y, marker="|", s=400, color=INK, linewidth=2, label="True value", zorder=4)
    ax.set_yticks(y, s.index)
    ax.set_ylim(-0.8, len(s) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Price elasticity (more negative = more price-sensitive)")
    ax.set_title("Estimated vs. true elasticity")
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(l) for l in ("True value", "Regression estimate (95% CI)", "Naive week-over-week average")]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="upper left", bbox_to_anchor=(0, -0.2), ncol=3, fontsize=9, borderaxespad=0)
    fig.tight_layout()
    fig.savefig(FIG / "estimates_vs_truth.png", dpi=150)
    plt.close(fig)


def fig_revenue(summary, brands):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    s = summary.set_index("brand").loc[brands[::-1]]
    vals = s["revenue_change_pct_if_price_up_5pct"]
    y = np.arange(len(s))
    ax.barh(y, vals, height=0.45, color=[GAIN if v >= 0 else LOSS for v in vals])
    ax.axvline(0, color=AXIS, linewidth=1)
    for yi, v in zip(y, vals):
        ax.text(v + (0.15 if v >= 0 else -0.15), yi, f"{v:+.1f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=INK)
    ax.set_yticks(y, s.index)
    ax.grid(axis="y", visible=False)
    lim = max(abs(vals).max() * 1.35, 1)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Change in weekly revenue (%)")
    ax.set_title("What a 5% price increase would do to revenue")
    fig.tight_layout()
    fig.savefig(FIG / "revenue_impact_5pct.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    df = pd.read_csv(DATA / "retail_price_volume.csv", parse_dates=["week"])
    truth = pd.read_csv(DATA / "true_elasticities.csv")
    brands = sorted(df["brand"].unique())

    weekly = naive_elasticity(df)
    naive = weekly.groupby("brand")["wow_elasticity"].agg(naive_elasticity="mean", naive_std="std",
                                                          weeks_used="count").reset_index()
    reg, week_effect, r2 = regression_elasticity(df)
    summary = reg.merge(naive, on="brand").merge(truth, on="brand")
    summary["elasticity_type"] = summary["regression_elasticity"].map(classify)
    summary["revenue_change_pct_if_price_up_5pct"] = 100 * ((1 + PRICE_TEST) ** (1 + summary["regression_elasticity"]) - 1)

    cols = ["brand", "regression_elasticity", "ci_low", "ci_high", "elasticity_type", "naive_elasticity",
            "naive_std", "weeks_used", "true_elasticity", "revenue_change_pct_if_price_up_5pct"]
    summary[cols].round(3).to_csv(OUT / "elasticity_summary.csv", index=False)
    weekly.assign(week=weekly["week"].dt.date).round(4).to_csv(OUT / "weekly_with_elasticity.csv", index=False)

    fig_weekly(df, brands)
    fig_loglog(df, summary, week_effect, brands)
    fig_compare(summary, brands)
    fig_revenue(summary, brands)

    pd.set_option("display.width", 140)
    print(f"Regression R² = {r2:.3f}\n")
    print(summary[cols].round(2).to_string(index=False))
    print(f"\nMean absolute error vs truth — regression: "
          f"{(summary['regression_elasticity'] - summary['true_elasticity']).abs().mean():.2f}, "
          f"naive: {(summary['naive_elasticity'] - summary['true_elasticity']).abs().mean():.2f}")
    print(f"\nOutputs written to {OUT.name}/ and {FIG.name}/")


if __name__ == "__main__":
    main()
