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


# ==============================================================
# 1. Descriptive Statistics
# ==============================================================
# The five-number summary (min, Q1, median, Q3, max) plus mean and
# std give a compact picture of each numeric column. pandas describe()
# computes all of these in one call. Use it on any new dataset first.
# Comparing mean vs. median reveals skew: a large positive gap means
# a right tail is pulling the mean up.

def demo_descriptive_stats() -> None:
    df      = make_sales_df()
    numeric = ["sales", "units", "discount", "rating"]

    print(f"\n  Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Date range:    {df['date'].min().date()} to {df['date'].max().date()}")
    print()
    print("  describe() output (numeric columns):")
    print(df[numeric].describe().round(3).to_string())

    print()
    print(f"  {'Column':>10} | {'Mean':>8} | {'Median':>8} | {'Gap':>8}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for col in numeric:
        col_data = df[col].dropna()
        mean     = col_data.mean()
        median   = col_data.median()
        print(f"  {col:>10} | {mean:>8.3f} | {median:>8.3f} | {mean - median:>+8.3f}")


# ==============================================================
# 2. Distribution Shape: Skewness and Kurtosis
# ==============================================================
# Skewness measures asymmetry: > 0 = right tail, < 0 = left tail.
# |skew| > 1 is usually considered substantial.
# Excess kurtosis measures tail weight relative to a normal distribution;
# > 0 means heavier tails and more likely extreme values.
# Coefficient of variation (CV = std / mean) compares spread across
# columns that live on different scales.

def demo_distribution_shape() -> None:
    df      = make_sales_df()
    numeric = ["sales", "units", "discount", "rating"]

    print(f"\n  {'Column':>10} | {'Skewness':>9} | {'Ex. Kurt':>9} | {'CV (%)':>8} | Shape")
    print(f"  {'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}-+-{'-'*20}")

    for col in numeric:
        vals = df[col].dropna()
        skew = float(vals.skew())
        kurt = float(vals.kurtosis())
        cv   = vals.std() / vals.mean() * 100 if vals.mean() != 0 else float("nan")
        if abs(skew) < 0.5:
            shape = "roughly symmetric"
        elif skew > 0:
            shape = "right-skewed"
        else:
            shape = "left-skewed"
        print(f"  {col:>10} | {skew:>9.4f} | {kurt:>9.4f} | {cv:>8.2f} | {shape}")

    stat, p = stats.shapiro(df["sales"].dropna())
    print()
    print(f"  Shapiro-Wilk test on 'sales': W={stat:.4f}  p={p:.4f}")
    print(f"  {'Normally distributed' if p > 0.05 else 'Deviates from normal'} at the 5% level.")


# ==============================================================
# 3. Missing Data Summary
# ==============================================================
# Before modelling, count and locate missing values column by column.
# A column with > 30 % missing is often dropped; < 5 % can be imputed.
# Checking whether missingness concentrates in particular groups
# (e.g., a single category always missing ratings) reveals non-random
# patterns that may signal a data collection problem.

def demo_missing_data() -> None:
    df = make_sales_df()

    miss_count = df.isnull().sum()
    miss_pct   = df.isnull().mean() * 100

    print(f"\n  Missing value summary:")
    print(f"  {'Column':>12} | {'Count':>7} | {'%':>7} | Suggested action")
    print(f"  {'-'*12}-+-{'-'*7}-+-{'-'*7}-+-{'-'*22}")

    for col in df.columns:
        count = int(miss_count[col])
        pct   = miss_pct[col]
        if pct == 0:
            action = "no action needed"
        elif pct < 5:
            action = "impute (low rate)"
        elif pct < 30:
            action = "impute carefully"
        else:
            action = "consider dropping"
        print(f"  {col:>12} | {count:>7} | {pct:>7.1f} | {action}")

    # Are missing ratings concentrated in any single category?
    miss_by_cat = (df.groupby("category")["rating"]
                   .apply(lambda s: int(s.isnull().sum()))
                   .rename("n_missing"))
    print(f"\n  Missing ratings by category:")
    print(miss_by_cat.to_string())

    # Median imputation example
    df["rating_filled"] = df["rating"].fillna(df["rating"].median())
    still_missing = int(df["rating_filled"].isnull().sum())
    print(f"\n  After median imputation: {still_missing} missing values remain in 'rating'.")


# ==============================================================
# 4. Correlation Analysis
# ==============================================================
# Pearson r measures linear association; sensitive to outliers.
# Spearman rho is rank-based: robust to outliers and non-linearity.
# Both range from -1 (perfect negative) to +1 (perfect positive).
# Correlations > 0.7 between features often cause multicollinearity
# in regression — flag these before fitting any model.

def demo_correlation() -> None:
    df      = make_sales_df()
    numeric = ["sales", "units", "discount", "rating"]

    pearson  = df[numeric].corr(method="pearson")
    spearman = df[numeric].corr(method="spearman")

    print(f"\n  Pearson correlation matrix:")
    print(pearson.round(3).to_string())
    print(f"\n  Spearman correlation matrix:")
    print(spearman.round(3).to_string())

    print(f"\n  Notable Pearson correlations (|r| > 0.25):")
    found = False
    for i in range(len(numeric)):
        for j in range(i + 1, len(numeric)):
            r = pearson.loc[numeric[i], numeric[j]]
            if abs(r) > 0.25:
                direction = "positive" if r > 0 else "negative"
                print(f"  {numeric[i]} vs {numeric[j]}: r={r:.3f}  ({direction})")
                found = True
    if not found:
        print("  (none above threshold — features are weakly correlated)")


# ==============================================================
# 5. Value Counts and Frequency Tables
# ==============================================================
# For categorical columns, value_counts() replaces describe().
# normalize=True converts counts to proportions for easy comparison.
# pd.crosstab() shows how two categorical variables co-occur — it is
# the exploratory version of a contingency table and highlights
# imbalanced category × region combinations.

def demo_value_counts() -> None:
    df = make_sales_df()

    for col in ["category", "region"]:
        counts = df[col].value_counts()
        pcts   = df[col].value_counts(normalize=True) * 100
        print(f"\n  '{col}' distribution:")
        print(f"  {'Level':>14} | {'Count':>6} | {'%':>6}")
        print(f"  {'-'*14}-+-{'-'*6}-+-{'-'*6}")
        for level in counts.index:
            print(f"  {level:>14} | {int(counts[level]):>6} | {pcts[level]:>6.1f}")

    ct = pd.crosstab(df["category"], df["region"])
    print(f"\n  Cross-tabulation (category × region):")
    print(ct.to_string())
