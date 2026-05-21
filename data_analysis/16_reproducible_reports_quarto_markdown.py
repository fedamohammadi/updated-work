"""
Reproducible Reports with Python, Markdown, and Quarto:
- Markdown basics: headings, lists, code blocks, and tables
- Generating .md files programmatically from Python data
- string.Template: injecting computed values into report templates
- pandas HTML styling: colour-coded tables for HTML reports
- Quarto concepts: YAML frontmatter, Python chunks, and rendering
- Generating a .qmd file from Python: parameterised report source
- Practical example: auto-generating a complete analysis report
"""

import os
import string
import textwrap
import numpy as np
import pandas as pd

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ==============================================================
# Shared dataset: quarterly product sales
# ==============================================================
# Three months of sales across five products with revenue, units,
# and gross margin per row. Every demo section uses this data so
# the report content stays consistent across all output formats.

def make_sales_df() -> pd.DataFrame:
    np.random.seed(16)
    products = ["Laptop", "Headphones", "Keyboard", "Monitor", "Webcam"]
    months   = ["2024-01", "2024-02", "2024-03"]
    bases    = {"Laptop": 1200, "Headphones": 90, "Keyboard": 50,
                "Monitor": 350, "Webcam": 60}
    rows = []
    for month in months:
        for product in products:
            units   = int(np.random.randint(5, 40))
            price   = bases[product] * np.random.uniform(0.95, 1.05)
            revenue = round(units * price, 2)
            margin  = round(np.random.uniform(0.20, 0.45), 3)
            rows.append({"month": month, "product": product,
                         "units": units, "revenue": revenue, "margin": margin})
    return pd.DataFrame(rows)


def _df_to_md(df: pd.DataFrame, fmt: dict = None) -> str:
    """Convert a DataFrame to a GitHub-flavoured Markdown pipe table."""
    fmt     = fmt or {}
    display = df.copy()
    for col, fn in fmt.items():
        if col in display.columns:
            display[col] = display[col].map(fn)
    cols   = list(display.columns)
    widths = {c: max(len(str(c)), display[c].astype(str).str.len().max()) for c in cols}
    header = "| " + " | ".join(f"{c:<{widths[c]}}" for c in cols) + " |"
    sep    = "| " + " | ".join("-" * widths[c] for c in cols) + " |"
    rows   = ["| " + " | ".join(f"{str(display.iat[i, j]):<{widths[c]}}"
                                 for j, c in enumerate(cols)) + " |"
              for i in range(len(display))]
    return "\n".join([header, sep] + rows)
