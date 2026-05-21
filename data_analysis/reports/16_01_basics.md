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
