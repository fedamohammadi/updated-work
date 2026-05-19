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
