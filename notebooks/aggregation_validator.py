import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl

    return Path, mo, pl


@app.cell
def _(mo):
    mo.md(
        r"""
    # Aggregation Validation Tool

    This notebook validates if data is properly aggregated by checking that all non-quantity columns form unique combinations.
    """
    )
    return


@app.cell
def _(Path, pl):
    data_path = Path("data/processed/processed_optimized_082025_salvador.parquet")
    df = pl.read_parquet(data_path)

    return (df,)


@app.cell
def _(df, mo):
    mo.md(
        f"""
    ## Dataset Overview

    - **Total records**: {len(df):,}
    - **Columns**: {len(df.columns)}
    - **Column names**: {", ".join(df.columns)}
    """
    )
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(mo):
    mo.md(r"""## Aggregation Validation""")
    return


@app.cell
def _(df, mo):
    quantity_col = "quantity"

    if quantity_col not in df.columns:
        mo.md("❌ **ERROR**: 'quantity' column not found in dataset!")
        mo.stop(True)

    non_quantity_cols = [col for col in df.columns if col != quantity_col]

    mo.md(f"""
    **Validation Setup:**

    - Quantity column: `{quantity_col}`
    - Non-quantity columns ({len(non_quantity_cols)}): {", ".join(f"`{col}`" for col in non_quantity_cols)}
    """)
    return non_quantity_cols, quantity_col


@app.cell
def _(df, non_quantity_cols, pl):
    grouped_df = df.group_by(non_quantity_cols).len().rename({"len": "row_count"})

    duplicates = grouped_df.filter(pl.col("row_count") > 1)
    total_combinations = len(grouped_df)
    duplicate_combinations = len(duplicates)

    return duplicate_combinations, duplicates, total_combinations


@app.cell
def _(df, duplicates, mo, non_quantity_cols):
    if len(duplicates) > 0:
        sample_duplicate = duplicates.head(1)

        duplicate_example = df.join(
            sample_duplicate.select(non_quantity_cols), on=non_quantity_cols, how="inner"
        )

        mo.md(f"""
        ❌ **VALIDATION FAILED**

        Found {len(duplicates)} combinations with multiple rows.

        **Example of duplicate combination:**
        """)

        duplicate_example
    else:
        mo.md("✅ **VALIDATION PASSED** - No duplicate combinations found!")
    return


@app.cell
def _(df, duplicate_combinations, mo, pl, quantity_col, total_combinations):
    total_rows = len(df)
    expected_rows_if_aggregated = total_combinations
    aggregation_ratio = (
        total_rows / expected_rows_if_aggregated if expected_rows_if_aggregated > 0 else 0
    )

    quantity_stats = df.select(
        pl.col(quantity_col).sum().alias("total_quantity"),
        pl.col(quantity_col).mean().alias("avg_quantity"),
        pl.col(quantity_col).min().alias("min_quantity"),
        pl.col(quantity_col).max().alias("max_quantity"),
    ).to_dicts()[0]

    validation_status = "✅ PASS" if duplicate_combinations == 0 else "❌ FAIL"

    mo.md(f"""
    ## Validation Summary

    **Overall Status**: {validation_status}

    **Aggregation Metrics:**

    - Total rows in dataset: {total_rows:,}
    - Unique combinations of non-quantity columns: {total_combinations:,}
    - Duplicate combinations: {duplicate_combinations:,}
    - Aggregation ratio: {aggregation_ratio:.2f}x

    **Quantity Column Statistics:**

    - Total quantity: {quantity_stats["total_quantity"]:,}
    - Average quantity per row: {quantity_stats["avg_quantity"]:.2f}
    - Min quantity: {quantity_stats["min_quantity"]}
    - Max quantity: {quantity_stats["max_quantity"]:,}

    **Interpretation:**

    - If aggregation ratio = 1.0: ✅ Data is properly aggregated (no duplicates)
    - If aggregation ratio > 1.0: ❌ Data has duplicate combinations that should be aggregated
    - If aggregation ratio < 1.0: ⚠️ Unexpected scenario (fewer rows than unique combinations)
    """)
    return


@app.cell
def _(duplicates, mo):
    if len(duplicates) > 0:
        mo.md(f"""
        ## Duplicate Combinations Details

        The following {len(duplicates)} combinations have multiple rows and should be investigated:
        """)
        duplicates.sort("row_count", descending=True)
    return


if __name__ == "__main__":
    app.run()
