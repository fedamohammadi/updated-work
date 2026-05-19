"""
Visualization with Matplotlib:
- The Figure and Axes API: creating and customising plot canvases
- Line plots: time series and multi-series comparisons
- Bar charts: single-category and grouped comparisons
- Scatter plots: visualising relationships between two variables
- Histograms and density curves: understanding distributions
- Subplots and multi-panel layout
- Practical example: a five-panel EDA dashboard
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend; saves to files
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import gaussian_kde, norm

PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared dataset: monthly sales by category
# ==============================================================
# 24 months of aggregated sales across four product categories.
# Each category has a realistic base level, an upward trend, and
# random noise. Used by all demo functions so each section can
# focus on chart technique rather than data wrangling.

def make_monthly_df() -> pd.DataFrame:
    np.random.seed(12)
    months     = pd.date_range("2022-01", periods=24, freq="ME")
    categories = ["Electronics", "Clothing", "Books", "Home"]
    bases      = {"Electronics": 4000, "Clothing": 2500, "Books": 800, "Home": 1500}
    rows = []
    for cat in categories:
        base  = bases[cat]
        trend = np.linspace(0, base * 0.3, 24)
        noise = np.random.normal(0, base * 0.08, 24)
        sales = np.maximum(base + trend + noise, 0)
        for month, s in zip(months, sales):
            rows.append({"month": month, "category": cat, "sales": round(s, 2)})
    return pd.DataFrame(rows)


# ==============================================================
# 1. The Figure and Axes API
# ==============================================================
# matplotlib organises output around two objects:
#   Figure: the full canvas (size, resolution, background colour).
#   Axes:   a single plot area inside the figure with its own
#           x/y axes, labels, title, and artists (lines, bars, etc.).
#
# plt.subplots() returns both at once. Calling methods on the Axes
# object (ax.plot(), ax.set_title()) is the recommended style — it
# avoids the global state issues of pyplot's plt.plot() interface.
# fig.savefig() writes the figure to disk; plt.close(fig) frees memory.

def demo_figure_axes() -> None:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    x = np.linspace(0, 2 * np.pi, 120)
    axes[0].plot(x, np.sin(x), color="steelblue", linewidth=2, label="sin(x)")
    axes[0].plot(x, np.cos(x), color="tomato",    linewidth=2, linestyle="--", label="cos(x)")
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle=":")
    axes[0].set_title("Sine and Cosine")
    axes[0].set_xlabel("x (radians)")
    axes[0].set_ylabel("value")
    axes[0].legend()

    x2 = np.linspace(-3, 3, 80)
    axes[1].plot(x2, x2 ** 2, color="seagreen", linewidth=2)
    axes[1].annotate("minimum", xy=(0, 0), xytext=(1.0, 3.5),
                     arrowprops={"arrowstyle": "->", "color": "black"},
                     fontsize=10)
    axes[1].set_title("Parabola with Annotation")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("x²")

    fig.suptitle("Figure / Axes API demo", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "12_01_figure_axes.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"\n  Figure attributes: size={fig.get_size_inches()} inches")
    print(f"  Number of Axes objects: 2")
    print(f"  Left panel : sin(x) and cos(x) with legend and zero line.")
    print(f"  Right panel: parabola with an annotation arrow at the minimum.")
    print(f"  Saved -> {path}")


# ==============================================================
# 2. Line Plots and Time Series
# ==============================================================
# Line plots connect ordered data points and are the natural choice
# for time series. Each category gets a distinct colour and marker.
# ax.yaxis.set_major_formatter() formats tick labels (e.g., $1,000)
# without touching the underlying data. plt.xticks(rotation=30)
# prevents date labels from overlapping on the x-axis.

def demo_line_plots() -> None:
    df     = make_monthly_df()
    cats   = ["Electronics", "Clothing", "Books", "Home"]
    colors = ["steelblue", "tomato", "seagreen", "darkorange"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for cat, col in zip(cats, colors):
        sub = df[df["category"] == cat].sort_values("month")
        ax.plot(sub["month"], sub["sales"],
                label=cat, color=col, linewidth=2, marker="o", markersize=3)

    ax.set_title("Monthly Sales by Category (2022–2023)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales ($)")
    ax.legend(title="Category", loc="upper left")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, "12_02_line_plots.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"\n  Category trend (first -> last month):")
    for cat in cats:
        sub    = df[df["category"] == cat].sort_values("month")
        first  = sub["sales"].iloc[0]
        last   = sub["sales"].iloc[-1]
        change = (last - first) / first * 100
        print(f"  {cat:<14}: ${first:>7,.0f} -> ${last:>7,.0f}  ({change:>+.1f}%)")
    print(f"\n  Saved -> {path}")


# ==============================================================
# 3. Bar Charts
# ==============================================================
# Bar charts compare magnitudes across discrete categories.
# A grouped bar chart places bars for multiple series side by side
# using a manual x-offset computed from np.arange(). bar_label()
# adds value labels directly on each bar without manual positioning.
# Horizontal bars (barh) work better when category names are long.

def demo_bar_charts() -> None:
    df = make_monthly_df()

    avg_sales = (df.groupby("category")["sales"]
                   .mean()
                   .sort_values(ascending=False))
    df["half"] = df["month"].dt.month.apply(lambda m: "H1" if m <= 6 else "H2")
    pivot      = df.groupby(["category", "half"])["sales"].mean().unstack()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: single-category bar with data labels
    bars = axes[0].bar(avg_sales.index, avg_sales.values,
                       color=["steelblue", "tomato", "seagreen", "darkorange"],
                       edgecolor="white", linewidth=0.8)
    axes[0].bar_label(bars, fmt="$%.0f", padding=4, fontsize=9)
    axes[0].set_title("Average Monthly Sales by Category")
    axes[0].set_ylabel("Avg Sales ($)")
    axes[0].set_ylim(0, avg_sales.max() * 1.22)
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    # Right: grouped bar chart — H1 vs H2
    x     = np.arange(len(pivot))
    width = 0.35
    axes[1].bar(x - width / 2, pivot["H1"], width, label="H1", color="steelblue")
    axes[1].bar(x + width / 2, pivot["H2"], width, label="H2", color="tomato")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pivot.index, rotation=15, ha="right")
    axes[1].set_title("H1 vs H2 Average Sales")
    axes[1].set_ylabel("Avg Sales ($)")
    axes[1].legend()
    axes[1].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "12_03_bar_charts.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"\n  Average monthly sales by category:")
    print(avg_sales.round(0).to_string())
    print(f"\n  H1 vs H2 averages:")
    print(pivot.round(0).to_string())
    print(f"\n  Saved -> {path}")


# ==============================================================
# 4. Scatter Plots
# ==============================================================
# A scatter plot reveals the relationship between two numeric variables.
# Colour-coding a third variable adds an extra dimension without a
# second axes. np.polyfit() fits a linear trend; plotting the resulting
# line over the scatter shows whether the trend is strong or weak.

def demo_scatter_plots() -> None:
    df     = make_monthly_df()
    cats   = sorted(df["category"].unique())
    colors = ["steelblue", "tomato", "seagreen", "darkorange"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Electronics scatter + OLS trend line
    elec  = df[df["category"] == "Electronics"].sort_values("month")
    x_num = np.arange(len(elec))
    y     = elec["sales"].values
    m, b  = np.polyfit(x_num, y, 1)
    axes[0].scatter(x_num, y, color="steelblue", s=55, zorder=3, label="Monthly sales")
    axes[0].plot(x_num, m * x_num + b, color="tomato", linewidth=2,
                 linestyle="--", label=f"Trend (+${m:.0f}/mo)")
    axes[0].set_title("Electronics Sales: Scatter + Trend")
    axes[0].set_xlabel("Month index")
    axes[0].set_ylabel("Sales ($)")
    axes[0].legend()
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    # Right: all categories, colour-coded
    for cat, col in zip(cats, colors):
        sub   = df[df["category"] == cat].sort_values("month")
        x_idx = np.arange(len(sub))
        axes[1].scatter(x_idx, sub["sales"].values,
                        color=col, s=40, alpha=0.75, label=cat)
    axes[1].set_title("Sales by Category (Scatter)")
    axes[1].set_xlabel("Month index")
    axes[1].set_ylabel("Sales ($)")
    axes[1].legend(title="Category", fontsize=9)
    axes[1].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "12_04_scatter_plots.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"\n  Electronics OLS trend: slope={m:+.1f} $/month  intercept=${b:.0f}")
    print(f"\n  Saved -> {path}")


# ==============================================================
# 5. Histograms and Density Curves
# ==============================================================
# A histogram bins continuous values to show distribution shape.
# density=True normalises the y-axis to a probability density (area
# integrates to 1), enabling direct overlay with a KDE or normal PDF.
# gaussian_kde from scipy provides a non-parametric smooth estimate;
# norm.pdf plots the normal reference curve.

def demo_histograms() -> None:
    df = make_monthly_df()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: single histogram + KDE + normal PDF for all sales
    sales = df["sales"].values
    axes[0].hist(sales, bins=25, density=True,
                 color="steelblue", alpha=0.55, edgecolor="white", label="Histogram")
    xs  = np.linspace(sales.min(), sales.max(), 300)
    kde = gaussian_kde(sales)
    axes[0].plot(xs, kde(xs), color="tomato",   linewidth=2.0, label="KDE")
    mu, sigma = sales.mean(), sales.std()
    axes[0].plot(xs, norm.pdf(xs, mu, sigma), color="seagreen", linewidth=2.0,
                 linestyle="--", label=f"Normal(μ={mu:.0f})")
    axes[0].set_title("Distribution of All Monthly Sales")
    axes[0].set_xlabel("Sales ($)")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Right: overlapping histograms for two categories
    for cat, col in [("Electronics", "steelblue"), ("Books", "tomato")]:
        vals = df[df["category"] == cat]["sales"].values
        axes[1].hist(vals, bins=16, density=True,
                     color=col, alpha=0.45, edgecolor="white", label=cat)
    axes[1].set_title("Electronics vs Books Distribution")
    axes[1].set_xlabel("Sales ($)")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "12_05_histograms.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"\n  Overall sales: mean=${mu:.0f}  std=${sigma:.0f}  skew={df['sales'].skew():.3f}")
    for cat in ["Electronics", "Books"]:
        vals = df[df["category"] == cat]["sales"]
        print(f"  {cat:<14}: mean=${vals.mean():.0f}  std=${vals.std():.0f}")
    print(f"\n  Saved -> {path}")


# ==============================================================
# 6. Subplots and Multi-Panel Layout
# ==============================================================
# plt.subplots(nrows, ncols) creates a grid of Axes objects.
# sharex=True links x-axes so zooming or panning one panel affects
# all others in the column — useful for time series comparisons.
# fill_between() adds a shaded area under the line for visual emphasis.
# plt.setp() applies shared formatting (rotation) to a list of artists.

def demo_subplots() -> None:
    df     = make_monthly_df()
    cats   = ["Electronics", "Clothing", "Books", "Home"]
    colors = ["steelblue", "tomato", "seagreen", "darkorange"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes_flat = axes.ravel()

    for ax, cat, col in zip(axes_flat, cats, colors):
        sub = df[df["category"] == cat].sort_values("month")
        ax.plot(sub["month"], sub["sales"], color=col, linewidth=1.8)
        ax.fill_between(sub["month"], sub["sales"], alpha=0.15, color=col)
        ax.set_title(f"{cat}", fontweight="bold")
        ax.set_ylabel("Sales ($)")
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))

    # Rotate x-tick labels only on the bottom two panels
    for ax in axes_flat[2:]:
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.suptitle("Monthly Sales by Category (2022–2023)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path = os.path.join(PLOT_DIR, "12_06_subplots.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"\n  2×2 subplot grid: one panel per category.")
    print(f"  sharex=True: all panels share the same date range on the x-axis.")
    print(f"  fill_between: shaded area highlights volume under each series.")
    print(f"  Saved -> {path}")
