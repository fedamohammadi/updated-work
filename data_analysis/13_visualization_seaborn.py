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
