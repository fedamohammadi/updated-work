"""
Visualization with Seaborn:
- Seaborn vs. matplotlib: what the high-level API adds
- Distribution plots: histplot and kdeplot
- Box plots and violin plots: spread and shape across groups
- Categorical plots: barplot and countplot
- Relationship plots: scatterplot, regplot, and pairplot
- Heatmaps: correlation matrices and pivot tables
- Practical example: a multi-panel publication-ready figure
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared dataset: retail transactions (item level)
# ==============================================================
# 400 transaction rows with sales, units, discount, and rating across
# five product categories and four regions. The larger row count gives
# seaborn's statistical functions (KDE, confidence intervals) enough
# data to produce smooth, reliable estimates.

def make_transactions_df() -> pd.DataFrame:
    np.random.seed(13)
    n          = 400
    categories = ["Electronics", "Clothing", "Books", "Home", "Sports"]
    regions    = ["North", "South", "East", "West"]
    cat_arr    = np.random.choice(categories, n)
    base_sales = {"Electronics": 1800, "Clothing": 900, "Books": 250,
                  "Home": 650, "Sports": 500}
    sales = np.array([
        max(0.0, np.random.normal(base_sales[c], base_sales[c] * 0.25))
        for c in cat_arr
    ]).round(2)
    return pd.DataFrame({
        "category": cat_arr,
        "region":   np.random.choice(regions, n),
        "sales":    sales,
        "units":    np.random.randint(1, 30, n),
        "discount": np.random.uniform(0, 0.35, n).round(2),
        "rating":   np.clip(np.random.normal(3.7, 0.7, n), 1.0, 5.0).round(1),
    })


# ==============================================================
# 1. Seaborn vs. Matplotlib
# ==============================================================
# Seaborn is built on top of matplotlib and provides three key additions:
#   1. Statistical aggregation built in: barplot() computes mean + CI
#      automatically; you do not need to groupby first.
#   2. DataFrame-native API: pass column names as strings via x=, y=,
#      hue= — seaborn reads directly from a pandas DataFrame.
#   3. Polished defaults: colour palettes, grids, and font sizing that
#      require many matplotlib lines to replicate manually.
# matplotlib still controls figure saving, size, and low-level artists.

def demo_seaborn_vs_matplotlib() -> None:
    df = make_transactions_df()

    avg = df.groupby("category")["sales"].mean().sort_values(ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: matplotlib — requires manual groupby before plotting
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2"]
    axes[0].bar(avg.index, avg.values, color=colors, edgecolor="white")
    axes[0].set_title("matplotlib: manual groupby + bar()")
    axes[0].set_ylabel("Avg Sales ($)")
    for tick in axes[0].get_xticklabels():
        tick.set_rotation(20)

    # Right: seaborn — aggregation and CI bars are handled internally
    sns.barplot(data=df, x="category", y="sales", order=avg.index,
                errorbar="ci", capsize=0.1, ax=axes[1])
    axes[1].set_title("seaborn: barplot() with 95% CI")
    axes[1].set_ylabel("Avg Sales ($)")
    for tick in axes[1].get_xticklabels():
        tick.set_rotation(20)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "13_01_seaborn_vs_matplotlib.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"\n  Left panel  (matplotlib): bar() requires a pre-computed groupby series.")
    print(f"  Right panel (seaborn)   : barplot() aggregates internally and adds 95% CI.")
    print(f"  Global theme (whitegrid + muted palette) applies automatically.")
    print(f"  Saved -> {path}")


# ==============================================================
# 2. Distribution Plots: histplot and kdeplot
# ==============================================================
# histplot() is the modern replacement for distplot() (deprecated).
#   kde=True overlays a smooth kernel density estimate on the histogram.
#   hue= splits by a categorical variable and draws one curve per level.
# kdeplot() draws only the smoothed density — cleaner when comparing
# many groups because overlapping filled curves are easy to read.

def demo_distribution_plots() -> None:
    df = make_transactions_df()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.histplot(data=df, x="sales", kde=True, bins=30,
                 color="steelblue", ax=axes[0])
    axes[0].set_title("Sales Distribution (histogram + KDE)")
    axes[0].set_xlabel("Sales ($)")

    sns.kdeplot(data=df, x="sales", hue="category", ax=axes[1],
                fill=True, alpha=0.25, linewidth=1.5)
    axes[1].set_title("Sales KDE by Category")
    axes[1].set_xlabel("Sales ($)")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "13_02_distribution_plots.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"\n  {'Category':>14} | {'Mean':>8} | {'Std':>8} | {'Skew':>7}")
    print(f"  {'-'*14}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}")
    for cat in sorted(df["category"].unique()):
        vals = df[df["category"] == cat]["sales"]
        print(f"  {cat:>14} | {vals.mean():>8.0f} | {vals.std():>8.0f} | {vals.skew():>7.3f}")
    print(f"\n  Saved -> {path}")
