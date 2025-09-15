import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from vir.config import PROCESSED_DATA_DIR

    return PROCESSED_DATA_DIR, mo, pl


@app.cell
def _(F, PROCESSED_DATA_DIR, mo):
    file_browser = mo.ui.file_browser(initial_path=PROCESSED_DATA_DIR, multiple=F)
    file_browser
    return (file_browser,)


@app.cell
def _(file_browser, pl):
    df = pl.read_parquet(file_browser.path(index=0))
    df = df.with_columns(
        pl.concat_str(
            [
                pl.col("year").cast(pl.Utf8),
                pl.lit("-W"),
                pl.col("week").cast(pl.Utf8).str.zfill(2),
                pl.lit("-1"),  # Monday (ISO 8601)
            ]
        )
        .str.to_date("%G-W%V-%u")  # ISO 8601 week format
        .alias("date")
    )
    df.head()
    return (df,)


@app.cell
def _(df):
    df.columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Temporal Clustering

    Using [tslearn](https://tslearn.readthedocs.io).

    A timeseries is a two-dimensional numpy array with its first dimension corresponding to the time axis and the second one being the feature dimensionality.

    Expected format:

    * `n_series`: number of temporal series (one by diagnosis and its quantity)
    * `n_timesteps`: number of points in time (in our case, number of weeks: 53)
    * `n_features`: number of variables that are being measured for each point in time.

    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
