"""
Summary Statistics and Exploratory Data Analysis:
- Descriptive statistics: central tendency, spread, and range
- Distribution shape: skewness, kurtosis, and coefficient of variation
- Missing data: counts, percentages, and group-level patterns
- Correlation analysis: Pearson, Spearman, and notable pairs
- Value counts and frequency tables for categorical columns
- Grouped statistics: comparing means and spreads across categories
- A practical EDA pipeline for a new dataset
"""

import numpy as np
import pandas as pd
from scipy import stats


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared dataset: retail sales transactions
# ==============================================================
# 200 rows of daily sales records across product categories and regions.
# Ten rating values and five discount values are set to NaN so the
# missing-data section has clear targets to detect and handle.

def make_sales_df() -> pd.DataFrame:
    np.random.seed(11)
    n          = 200
    categories = ["Electronics", "Clothing", "Books", "Home", "Sports"]
    regions    = ["North", "South", "East", "West"]
    df = pd.DataFrame({
        "date":     pd.date_range("2023-01-01", periods=n, freq="D"),
        "category": np.random.choice(categories, n),
        "region":   np.random.choice(regions, n),
        "sales":    np.abs(np.random.normal(1200, 400, n)).round(2),
        "units":    np.random.randint(1, 40, n),
        "discount": np.random.uniform(0, 0.30, n).round(2),
        "rating":   np.clip(np.random.normal(3.8, 0.6, n), 1.0, 5.0).round(1),
    })
    rng = np.random.default_rng(11)
    df.loc[rng.choice(n, 10, replace=False), "rating"]   = np.nan
    df.loc[rng.choice(n, 5,  replace=False), "discount"] = np.nan
    return df
