import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Temporal View ⌛

    Similarity Network Fusion
    """)
    return


@app.cell
def _():
    import os

    import marimo as mo
    import numpy as np
    import polars as pl
    from aeon.distances import dtw_pairwise_distance
    from dotenv import load_dotenv

    return dtw_pairwise_distance, load_dotenv, mo, np, os, pl


@app.cell
def _():
    from vir.config import INTERIM_DATA_DIR
    from vir.db import DEFAULT_MODEL, fetch_terminologies
    from vir.helpers import get_n_jobs

    return DEFAULT_MODEL, INTERIM_DATA_DIR, fetch_terminologies, get_n_jobs


@app.cell
def _(load_dotenv):
    _ = load_dotenv()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Data loading
    """)
    return


@app.cell
def _(DEFAULT_MODEL, fetch_terminologies):
    terminology_labels = fetch_terminologies(DEFAULT_MODEL)
    return (terminology_labels,)


@app.cell
def _(terminology_labels):
    terminology_labels
    return


@app.cell
def _(INTERIM_DATA_DIR, mo):
    _all_options = [
        str(_file.resolve()) for _file in INTERIM_DATA_DIR.glob("**/cleaned_*.parquet")
    ]
    default = [
        _option
        for _option in _all_options
        if "cleaned_aesop_raw_dados_2017_20250223_20municipios_salvador.parquet" in _option
    ]

    files_dropdown = mo.ui.dropdown(
        options=_all_options,
        label="Choose dataset",
        searchable=True,
        value=default[0] if default else None,
    )
    files_dropdown
    return (files_dropdown,)


@app.cell
def _(files_dropdown, pl, terminology_labels):
    selected_filename = files_dropdown.value
    df = pl.read_parquet(selected_filename)
    df = df.drop(["unique_id", "state", "city", "region", "population_range"])  # make it per city
    df = df.group_by(["year", "week", "sex", "age_group", "code_type", "code"]).agg(
        pl.col("quantity").sum()
    )
    df = df.join(terminology_labels, on=["code", "code_type"]).rename(
        {"id": "terminology_id", "label": "terminology_label"}
    )
    df = df.filter(pl.col("year").ge(2022))
    df.head()
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Temporal dimension ⌛

    * Features: week, year
    """)
    return


@app.cell
def _(df, pl):
    df_year_week = (
        df.with_columns(
            (pl.col("year").cast(pl.Utf8) + "-" + pl.col("week").cast(pl.Utf8)).alias("year_week")
        )
        .group_by(["terminology_id", "year", "week", "year_week"])
        .agg(pl.col("quantity").sum().alias("total"))
    )
    df_year_week
    return (df_year_week,)


@app.cell
def _(df_year_week):
    ordered_year_week = (
        df_year_week.select("year_week", "year", "week")
        .unique()
        .sort(["year", "week"])  # chronological order
        .get_column("year_week")
        .to_list()
    )
    return (ordered_year_week,)


@app.cell
def _(df_year_week, ordered_year_week):
    temporal_pivot = (
        df_year_week.pivot(
            index="terminology_id",  # rows
            on="year_week",  # new columns
            values="total",  # cell value
        )
        .fill_null(0)
        .select(["terminology_id"] + ordered_year_week)
    )
    temporal_pivot
    return (temporal_pivot,)


@app.cell
def _(dtw_pairwise_distance, get_n_jobs, ordered_year_week, temporal_pivot):
    dtw_results = dtw_pairwise_distance(
        temporal_pivot.select(ordered_year_week).to_numpy(),
        n_jobs=get_n_jobs(),
    )
    dtw_results
    return (dtw_results,)


@app.cell
def _():
    # dtw took 14 min with n jobs = -1
    # 25 min with 16 threads
    return


@app.cell
def _(dtw_results, np):
    def load_temporal_dimension():
        temporal_matrix = dtw_results.copy()

        np.fill_diagonal(temporal_matrix, 0)  # enforce 0
        return temporal_matrix

    return (load_temporal_dimension,)


@app.cell
def _(load_temporal_dimension):
    temporal_matrix = load_temporal_dimension()
    return (temporal_matrix,)


@app.cell
def _(temporal_matrix):
    temporal_matrix
    return


@app.cell
def _():
    # 19098 J10 influenza
    # 16966 E11 diabetes
    return


@app.cell
def _(np):
    from aeon.clustering import TimeSeriesKMeans

    X = np.random.random(size=(10, 2, 20))
    clst = TimeSeriesKMeans(distance="dtw", n_clusters=2)
    clst.fit(X)

    preds = clst.predict(X)
    return TimeSeriesKMeans, preds


@app.cell
def _(preds):
    preds
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Time Series Clustering
    """)
    return


@app.cell
def _(np):
    seed = 0
    np.random.seed(seed)
    return


@app.cell
def _(ordered_year_week, temporal_pivot):
    time_series = temporal_pivot.select(ordered_year_week).to_numpy()
    time_series
    return (time_series,)


@app.cell
def _(time_series):
    # Scaler for time series. Scales time series so that their mean (resp. standard deviation) in each dimension is mu (resp. std).
    # mu=0.0, std=1.0
    from aeon.transformations.collection import Normalizer

    normalizer = Normalizer()
    normalized_time_series = normalizer.fit_transform(time_series)
    normalized_time_series
    return (normalized_time_series,)


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _(normalized_time_series, plt):
    for _yi in range(5):
        plt.subplot(4, 3, _yi + 1)
        plt.plot(normalized_time_series[_yi].ravel(), "k-", alpha=0.2)
        plt.show()

    return


@app.cell
def _(TimeSeriesKMeans, normalized_time_series):
    sum_of_squared_distances = []

    for _k in range(2, 5):  # one cluster = 79 min
        print(_k)
        _k_means = TimeSeriesKMeans(distance="dtw", n_clusters=_k)
        _fitted = _k_means.fit(normalized_time_series)
        sum_of_squared_distances.append(_k_means.inertia_)

    return (sum_of_squared_distances,)


@app.cell
def _(sum_of_squared_distances):
    sum_of_squared_distances
    return


@app.cell
def _(plt, sum_of_squared_distances):
    plt.plot(range(2, 5), sum_of_squared_distances, "bx-")
    plt.xlabel("k")
    plt.ylabel("Sum of squared distances")
    plt.title("Elbow Method For Optimal k")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
