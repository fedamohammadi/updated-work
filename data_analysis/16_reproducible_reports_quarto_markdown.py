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


# ==============================================================
# 5. Quarto Concepts: YAML and Python Chunks
# ==============================================================
# Quarto (.qmd) is a markdown-based publishing system. A .qmd file
# has three parts: (1) YAML frontmatter between --- that sets the
# title, author, and output format; (2) Markdown prose; (3) fenced
# Python chunks (```{python}) that execute on render. Running
# "quarto render report.qmd" converts the file to HTML, PDF, or Word
# with plots and DataFrame outputs embedded automatically.

def demo_quarto_concepts() -> None:
    example = textwrap.dedent("""\
        ---
        title: "Sales Analysis Q1 2024"
        author: "Analytics Team"
        date: "2024-03-31"
        format:
          html:
            toc: true
            code-fold: true
          pdf:
            documentclass: article
        ---

        ## Overview

        This report summarises Q1 sales across five product categories.

        ```{python}
        #| echo: false
        #| fig-cap: "Revenue by product"
        import pandas as pd, matplotlib.pyplot as plt
        df = pd.read_csv("sales.csv")
        df.groupby("product")["revenue"].sum().sort_values().plot(kind="barh")
        plt.xlabel("Revenue ($)")
        plt.tight_layout()
        plt.show()
        ```

        The chart confirms that Laptop generated the highest revenue.
    """)

    print(f"\n  .qmd file structure:")
    for line in example.splitlines():
        print(f"  {line}")
    print(f"\n  Chunk options (# | prefix inside the chunk):")
    print(f"    #| echo: false     - hide code, show output only")
    print(f"    #| eval: false     - show code, skip execution")
    print(f"    #| fig-cap: '...' - add a caption below the figure")
    print(f"\n  Render commands:")
    print(f"    quarto render report.qmd")
    print(f"    quarto render report.qmd --to pdf")
    print(f"    quarto preview report.qmd   # live-reload in browser")


# ==============================================================
# 6. Generating a .qmd File from Python
# ==============================================================
# Writing the .qmd source from a Python script lets you parameterise
# the report -- swap date ranges, data sources, or filters without
# manually editing the document each time. string.Template fills
# computed summary values into the frontmatter and prose; the code
# chunks are left as static Python that quarto executes on render.

def demo_generate_qmd() -> None:
    df      = make_sales_df()
    total   = df["revenue"].sum()
    top     = df.groupby("product")["revenue"].sum().idxmax()
    months  = sorted(df["month"].unique())
    period  = f"{months[0]} to {months[-1]}"

    tmpl = string.Template(textwrap.dedent("""\
        ---
        title: "Sales Report: $period"
        author: "Automated Analytics"
        date: "$date"
        format: html
        ---

        ## Executive Summary

        Total revenue for $period was **$$$total**.
        The top-performing product was **$top_product**.

        ## Revenue by Product

        ```{python}
        #| echo: false
        import pandas as pd
        df = pd.read_csv("sales.csv")
        df.groupby("product").agg({"revenue": "sum", "units": "sum"})
        ```

        ## Monthly Trend

        ```{python}
        #| echo: false
        #| fig-cap: "Monthly revenue trend"
        import matplotlib.pyplot as plt
        df.groupby("month")["revenue"].sum().plot(marker="o")
        plt.ylabel("Revenue ($)")
        plt.tight_layout()
        plt.show()
        ```
    """))

    qmd = tmpl.safe_substitute(
        period=period,
        date="2024-03-31",
        total=f"{total:,.2f}",
        top_product=top,
    )

    path = os.path.join(REPORT_DIR, "16_06_report.qmd")
    with open(path, "w", encoding="utf-8") as f:
        f.write(qmd)

    print(f"\n  Generated .qmd file: {path}")
    print(f"  To render: quarto render {os.path.basename(path)}")
    print(f"\n  File content:")
    for line in qmd.splitlines():
        print(f"  {line}")


# ==============================================================
# 7. Practical Example: Auto-Generated Analysis Report
# ==============================================================

def demo_full_report() -> None:
    df = make_sales_df()

    pivot   = (df.groupby("product")
                 .agg(revenue=("revenue", "sum"),
                      units=("units", "sum"),
                      margin=("margin", "mean"))
                 .sort_values("revenue", ascending=False)
                 .round({"revenue": 2, "margin": 3})
                 .reset_index())
    monthly = (df.groupby("month")["revenue"]
                 .sum()
                 .reset_index()
                 .rename(columns={"revenue": "revenue"}))

    total   = pivot["revenue"].sum()
    top     = pivot.iloc[0]["product"]
    mom_pct = monthly["revenue"].pct_change().dropna().mean() * 100

    product_table = _df_to_md(
        pivot,
        {"revenue": lambda v: f"${v:,.2f}", "margin": lambda v: f"{v:.1%}"},
    )
    monthly_table = _df_to_md(
        monthly,
        {"revenue": lambda v: f"${v:,.2f}"},
    )

    report = textwrap.dedent(f"""\
        # Q1 2024 Sales Analysis Report

        *Generated automatically -- re-run the script to refresh all figures.*

        ## Executive Summary

        | Metric           | Value            |
        |------------------|------------------|
        | Total Revenue    | ${total:,.2f}    |
        | Top Product      | {top}            |
        | Avg MoM Growth   | {mom_pct:+.1f}%  |
        | Products         | {pivot.shape[0]} |
        | Months           | {monthly.shape[0]} |

        ## Revenue by Product

        {product_table}

        ## Monthly Revenue Trend

        {monthly_table}

        ## Notes

        - Data covers {df['month'].nunique()} months and {df['product'].nunique()} products.
        - Margin values are gross margin estimates.
        - Render to HTML with: `quarto render 16_06_report.qmd`
    """)

    path = os.path.join(REPORT_DIR, "16_07_full_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n  Full report saved to: {path}")
    print(f"\n  Report summary:")
    print(f"    Total revenue : ${total:,.2f}")
    print(f"    Top product   : {top}")
    print(f"    Avg MoM growth: {mom_pct:+.1f}%")
    print(f"\n  Product revenue table:")
    for line in product_table.splitlines():
        print(f"  {line}")
    print(f"\n  Monthly revenue:")
    for line in monthly_table.splitlines():
        print(f"  {line}")


# ==============================================================
# main
# ==============================================================

def main() -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)

    section("1. Markdown Basics")
    demo_markdown_basics()

    section("2. Markdown Tables from DataFrames")
    demo_markdown_tables()

    section("3. string.Template: Dynamic Report Sections")
    demo_string_template()

    section("4. pandas HTML Styling")
    demo_html_styling()

    section("5. Quarto Concepts: YAML and Python Chunks")
    demo_quarto_concepts()

    section("6. Generating a .qmd File from Python")
    demo_generate_qmd()

    section("7. Practical Example: Auto-Generated Analysis Report")
    demo_full_report()

    print(f"\n  All reports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
