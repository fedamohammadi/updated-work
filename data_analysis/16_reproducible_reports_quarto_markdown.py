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


# ==============================================================
# 1. Markdown Basics
# ==============================================================
# Markdown is a lightweight markup language that renders to HTML.
# # = h1, ## = h2, **text** = bold, *text* = italic, - = bullet.
# Fenced code blocks use triple backticks with an optional language
# tag for syntax highlighting. Generating .md files from Python
# embeds computed values directly in prose — the core idea behind
# reproducible reports: run the script, get an updated document.

def demo_markdown_basics() -> None:
    md = textwrap.dedent("""\
        # Sales Analysis Report

        Generated automatically by Python. Re-run the script to refresh.

        ## Key Findings

        - Total Q1 revenue exceeded **$50,000** across all products.
        - The **Laptop** category led all products by revenue share.
        - Average gross margin was *32%* across the product mix.

        ## Code Example

        ```python
        df = pd.read_sql_query("SELECT * FROM sales", engine)
        summary = df.groupby("product")["revenue"].sum().sort_values()
        ```

        > **Tip:** Pair this script with Quarto to render HTML or PDF output.
    """)

    path = os.path.join(REPORT_DIR, "16_01_basics.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n  Markdown elements written to: {path}")
    print(f"    # / ##     - h1 and h2 headings")
    print(f"    **bold**   - strong emphasis")
    print(f"    *italic*   - regular emphasis")
    print(f"    - item     - unordered list bullet")
    print(f"    ``` block  - fenced code block with language tag")
    print(f"    > quote    - blockquote for callouts")
    print(f"\n  First 5 lines of output:")
    for line in md.splitlines()[:5]:
        print(f"    {line}")


# ==============================================================
# 2. Markdown Tables from DataFrames
# ==============================================================
# A Markdown pipe table uses | to delimit columns and :--- for
# alignment. Building it manually avoids the optional tabulate
# dependency. The helper _df_to_md() measures column widths from
# the data and header, then formats each row to match. These tables
# render correctly in GitHub, VS Code preview, and Quarto HTML.

def demo_markdown_tables() -> None:
    df = make_sales_df()
    pivot = (df.groupby("product")
               .agg(Revenue=("revenue", "sum"),
                    Units=("units", "sum"),
                    Margin=("margin", "mean"))
               .round({"Revenue": 2, "Margin": 3})
               .sort_values("Revenue", ascending=False)
               .reset_index())

    md_table = _df_to_md(
        pivot,
        {"Revenue": lambda v: f"${v:,.2f}", "Margin": lambda v: f"{v:.1%}"},
    )

    content = f"# Q1 2024 Product Sales Summary\n\n{md_table}\n"
    path    = os.path.join(REPORT_DIR, "16_02_tables.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n  Markdown pipe table saved to: {path}")
    print(f"\n  Table preview:")
    for line in md_table.splitlines():
        print(f"  {line}")


# ==============================================================
# 3. string.Template: Dynamic Report Sections
# ==============================================================
# string.Template uses $variable or ${variable} placeholders.
# safe_substitute() fills known keys and leaves unknown ones intact
# — safer than substitute() which raises KeyError for missing keys.
# A $$ in the template produces a literal $ in the output, which
# is needed for currency values inside a Markdown template string.

def demo_string_template() -> None:
    df      = make_sales_df()
    total   = df["revenue"].sum()
    top     = df.groupby("product")["revenue"].sum().idxmax()
    avg_mgn = df["margin"].mean() * 100
    months  = df["month"].nunique()

    tmpl = string.Template(textwrap.dedent("""\
        ## Executive Summary -- $period

        | Metric           | Value         |
        |------------------|---------------|
        | Total Revenue    | $$$total      |
        | Top Product      | $top_product  |
        | Avg Margin       | $avg_margin%  |
        | Months Covered   | $months       |

        Sales were driven by strong demand for $top_product throughout
        $period. Gross margins averaged $avg_margin% across the product mix.
    """))

    filled = tmpl.safe_substitute(
        period="Q1 2024",
        total=f"{total:,.2f}",
        top_product=top,
        avg_margin=f"{avg_mgn:.1f}",
        months=months,
    )

    path = os.path.join(REPORT_DIR, "16_03_template.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(filled)

    print(f"\n  Template placeholders: period, total, top_product, avg_margin, months")
    print(f"\n  Rendered output:")
    for line in filled.splitlines():
        print(f"  {line}")


# ==============================================================
# 4. pandas HTML Styling
# ==============================================================
# df.style returns a Styler that attaches CSS to a DataFrame.
# .format() sets number display per column (%, $, commas).
# .highlight_max() and .highlight_min() colour extreme values.
# .background_gradient() applies a continuous colour scale.
# .to_html() renders the styled table as an HTML string that
# can be embedded directly in a report or opened in a browser.

def demo_html_styling() -> None:
    df = make_sales_df()
    pivot = (df.groupby("product")
               .agg(Revenue=("revenue", "sum"),
                    Units=("units", "sum"),
                    Margin=("margin", "mean"))
               .round({"Revenue": 2, "Margin": 3})
               .sort_values("Revenue", ascending=False))

    styled = (pivot.style
              .format({"Revenue": "${:,.2f}", "Units": "{:,}", "Margin": "{:.1%}"})
              .highlight_max(subset=["Revenue", "Units"], color="#c8f7c5")
              .highlight_min(subset=["Margin"],           color="#f7c8c8")
              .background_gradient(subset=["Revenue"],    cmap="Blues", vmin=0)
              .set_caption("Q1 2024 Product Sales -- styled HTML table"))

    html = f"<html><body>\n{styled.to_html()}\n</body></html>"
    path = os.path.join(REPORT_DIR, "16_04_styled.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n  Styled HTML saved to: {path}  (open in a browser to view)")
    print(f"\n  Styler methods applied:")
    print(f"    .format()              - currency, commas, and percentage formatting")
    print(f"    .highlight_max()       - green for highest revenue and units")
    print(f"    .highlight_min()       - red for lowest margin")
    print(f"    .background_gradient() - blue gradient scaled to revenue")
    print(f"\n  Plain pivot table:")
    print(pivot.to_string())
