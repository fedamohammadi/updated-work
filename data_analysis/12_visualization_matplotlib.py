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
