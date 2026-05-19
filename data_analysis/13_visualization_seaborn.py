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


# ==============================================================
# 3. Box Plots and Violin Plots
# ==============================================================
# A box plot shows the five-number summary (Q1, median, Q3, whiskers)
# and marks individual outliers as points. It is compact and reliable.
# A violin plot adds the full KDE shape on both sides of the axis,
# revealing bimodality or asymmetry that a box plot would hide.
# Use boxplot for quick outlier identification; violinplot when the
# distribution shape (not just quartiles) matters for the analysis.

def demo_box_violin() -> None:
    df = make_transactions_df()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.boxplot(data=df, x="category", y="sales", ax=axes[0],
                palette="muted",
                flierprops={"marker": "x", "markersize": 4})
    axes[0].set_title("Sales Box Plot by Category")
    axes[0].set_xlabel("Category")
    axes[0].set_ylabel("Sales ($)")
    for tick in axes[0].get_xticklabels():
        tick.set_rotation(20)

    sns.violinplot(data=df, x="category", y="sales", ax=axes[1],
                   palette="muted", inner="quartile", linewidth=1.2)
    axes[1].set_title("Sales Violin Plot by Category")
    axes[1].set_xlabel("Category")
    axes[1].set_ylabel("Sales ($)")
    for tick in axes[1].get_xticklabels():
        tick.set_rotation(20)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "13_03_box_violin.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"\n  {'Category':>14} | {'Q1':>7} | {'Median':>7} | {'Q3':>7} | Outliers")
    print(f"  {'-'*14}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}")
    for cat in sorted(df["category"].unique()):
        vals = df[df["category"] == cat]["sales"]
        q1, med, q3 = vals.quantile(0.25), vals.median(), vals.quantile(0.75)
        iqr  = q3 - q1
        out  = int(((vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)).sum())
        print(f"  {cat:>14} | {q1:>7.0f} | {med:>7.0f} | {q3:>7.0f} | {out:>8}")
    print(f"\n  Saved -> {path}")


# ==============================================================
# 4. Categorical Plots: barplot and countplot
# ==============================================================
# barplot() shows the mean of a numeric variable per category and adds
# a 95% bootstrap confidence interval — ideal for comparing group means.
# countplot() tallies the number of rows per category, equivalent to
# a bar chart of value_counts(). The hue= parameter splits bars by a
# second categorical variable and places them side by side automatically.

def demo_categorical_plots() -> None:
    df = make_transactions_df()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    order = (df.groupby("category")["rating"]
               .mean()
               .sort_values(ascending=False)
               .index)
    sns.barplot(data=df, x="category", y="rating", order=order,
                errorbar="ci", capsize=0.12, ax=axes[0], palette="muted")
    axes[0].set_title("Avg Rating by Category (95% CI)")
    axes[0].set_xlabel("Category")
    axes[0].set_ylabel("Rating")
    axes[0].set_ylim(0, 5.5)
    for tick in axes[0].get_xticklabels():
        tick.set_rotation(20)

    sns.countplot(data=df, x="region", hue="category",
                  ax=axes[1], palette="muted")
    axes[1].set_title("Transaction Count by Region and Category")
    axes[1].set_xlabel("Region")
    axes[1].set_ylabel("Count")
    axes[1].legend(title="Category", fontsize=8, loc="upper right")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "13_04_categorical_plots.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"\n  Average rating by category:")
    ratings = df.groupby("category")["rating"].mean().sort_values(ascending=False)
    for cat, r in ratings.items():
        print(f"  {cat:<14}: {r:.3f}")
    print(f"\n  Saved -> {path}")


# ==============================================================
# 5. Relationship Plots: scatterplot, regplot, and pairplot
# ==============================================================
# scatterplot() is the seaborn scatter with hue, size, and style support.
# regplot() adds an OLS fit line and a 95% confidence band in one call.
# pairplot() creates a grid of pairwise scatter plots for all numeric
# columns — the fastest multi-variable EDA tool when n_features is small.

def demo_relationship_plots() -> None:
    from scipy.stats import pearsonr

    df = make_transactions_df()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.scatterplot(data=df, x="discount", y="sales", hue="category",
                    ax=axes[0], alpha=0.55, s=35, palette="muted")
    axes[0].set_title("Sales vs Discount (by Category)")
    axes[0].set_xlabel("Discount")
    axes[0].set_ylabel("Sales ($)")
    axes[0].legend(title="Category", fontsize=8)

    sns.regplot(data=df, x="units", y="sales", ax=axes[1],
                scatter_kws={"alpha": 0.35, "s": 22, "color": "steelblue"},
                line_kws={"color": "tomato", "linewidth": 2})
    axes[1].set_title("Sales vs Units Sold (OLS Fit)")
    axes[1].set_xlabel("Units Sold")
    axes[1].set_ylabel("Sales ($)")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "13_05_relationship_plots.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    r_disc, p_disc = pearsonr(df["discount"], df["sales"])
    r_unit, p_unit = pearsonr(df["units"],    df["sales"])
    print(f"\n  Pearson r (sales vs discount): r={r_disc:.3f}  p={p_disc:.4f}")
    print(f"  Pearson r (sales vs units):    r={r_unit:.3f}  p={p_unit:.4f}")
    print(f"\n  Saved -> {path}")


# ==============================================================
# 6. Heatmaps
# ==============================================================
# sns.heatmap() visualises a matrix with colour intensity — most
# commonly a correlation matrix or an aggregated pivot table.
# annot=True prints the numeric value inside each cell.
# cmap="coolwarm" is the standard for correlations: red = strong
# positive, blue = strong negative, white = near zero.
# center=0, vmin=-1, vmax=1 anchors the colour scale correctly.

def demo_heatmaps() -> None:
    df = make_transactions_df()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    corr = df[["sales", "units", "discount", "rating"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, vmin=-1, vmax=1,
                linewidths=0.5, square=True, ax=axes[0])
    axes[0].set_title("Pearson Correlation Matrix")

    pivot = df.pivot_table(values="sales", index="category",
                           columns="region", aggfunc="mean")
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0.5, ax=axes[1])
    axes[1].set_title("Avg Sales: Category × Region")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "13_06_heatmaps.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"\n  Correlation matrix:")
    print(corr.round(3).to_string())
    print(f"\n  Avg sales by category × region:")
    print(pivot.round(0).to_string())
    print(f"\n  Saved -> {path}")
