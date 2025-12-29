import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os

    import plotly.express as px
    import polars as pl
    from dotenv import load_dotenv

    return load_dotenv, os, pl, px


@app.cell
def _():
    from vir.config import DATA_DIR, INTERIM_DATA_DIR
    from vir.db import (
        fetch_terminologies,
    )

    return DATA_DIR, INTERIM_DATA_DIR, fetch_terminologies


@app.cell
def _(load_dotenv):
    _ = load_dotenv()
    return


@app.cell
def _(os):
    n_jobs = int(os.getenv("NUMBER_OF_THREADS", -1))
    return (n_jobs,)


@app.cell
def _(mo):
    mo.md(r"""
    # Temporal dimension ⌛

    How the diagnosis behave through time and how similar they are.

    * Features: week, year, diagnosis code
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Preparation

    ### Fetch diagnosis terminology
    """)
    return


@app.cell
def _(fetch_terminologies):
    embeddings_model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    terminology_labels = fetch_terminologies(embeddings_model_name)
    return (terminology_labels,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Select codes that are indicative
    """)
    return


@app.cell
def _():
    indicative_codes_enabled = True
    return (indicative_codes_enabled,)


@app.cell
def _(DATA_DIR, indicative_codes_enabled, pl):
    include_codes = []
    if indicative_codes_enabled:
        indicative_codes = pl.read_csv(
            f"{DATA_DIR}/indicative_probabilities_sentence-transformers_all-MiniLM-L6-v2_codes.csv"
        )
        for _data in indicative_codes.iter_rows(named=True):
            if _data["indicative_probability"] > 0.50:
                include_codes.append(f"{_data['code']}:{_data['code_type']}")
    return (include_codes,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Prepare dataframe
    """)
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
def _(
    files_dropdown,
    include_codes,
    indicative_codes_enabled,
    pl,
    terminology_labels,
):
    selected_filename = files_dropdown.value
    df = pl.read_parquet(selected_filename)
    df = df.drop(["unique_id", "state", "city", "region", "population_range"])  # make it per city
    df = df.group_by(["year", "week", "sex", "age_group", "code_type", "code"]).agg(
        pl.col("quantity").sum()
    )
    df = df.join(terminology_labels, on=["code", "code_type"]).rename(
        {"id": "terminology_id", "label": "terminology_label"}
    )
    df = df.filter(pl.col("year").is_between(2022, 2024))
    if indicative_codes_enabled:
        df = df.filter(pl.col("terminology_label").is_in(include_codes))
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""

    """)
    return


@app.cell
def _(df, pl):
    df_week = df.group_by(["terminology_id", "week"]).agg(pl.col("quantity").sum().alias("total"))
    df_week
    return (df_week,)


@app.cell
def _(df_week):
    df_week
    return


@app.cell
def _():
    ordered_week = [str(_week) for _week in range(1, 53)]
    return (ordered_week,)


@app.cell
def _(df_week, ordered_week):
    temporal_pivot = (
        df_week.pivot(
            index="terminology_id",  # rows
            on="week",  # new columns
            values="total",  # cell value
        )
        .fill_null(0)
        .select(["terminology_id"] + ordered_week)
    )
    temporal_pivot
    return (temporal_pivot,)


@app.cell
def _(mo):
    mo.md(r"""
    Dynamic Time Warping is used to calculate the distance between each time series in a collection of time series.
    Our collection of time series here is composed by diagnosis and the frequency of cases in given week/year.
    """)
    return


@app.cell
def _(temporal_pivot):
    temporal_sample = temporal_pivot.sample(500)
    return (temporal_sample,)


@app.cell
def _(ordered_week, temporal_sample):
    only_weeks = temporal_sample.select(ordered_week).to_numpy()
    return (only_weeks,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Normalization

    The time series were aggregated by week, so we can capture cyclic behaviour.
    Now it is time to normalize, so less frequent diagnosis can be part of the representation.
    """)
    return


@app.cell
def _(only_weeks):
    only_weeks
    return


@app.cell
def _(only_weeks):
    from aeon.utils.numba.general import z_normalise_series_2d

    X_norm = z_normalise_series_2d(only_weeks)
    X_norm
    return (X_norm,)


@app.cell
def _():
    # dtw took 14 min with n jobs = -1
    # 25 min with 16 threads
    # dtw_results = dtw_pairwise_distance(
    #     only_weeks,
    #     n_jobs=int(os.getenv("NUMBER_OF_THREADS", -1)),
    # )
    # dtw_results
    return


@app.cell
def _(dtw_results, np):
    def load_temporal_dimension():
        temporal_matrix = dtw_results.copy()

        np.fill_diagonal(temporal_matrix, 0)  # enforce 0
        return temporal_matrix

    return


@app.cell
def _():
    # temporal_matrix = load_temporal_dimension()
    return


@app.cell
def _():
    from aeon.visualisation import plot_series_collection

    return (plot_series_collection,)


@app.cell
def _(X_norm, plot_series_collection):
    plot_series_collection(X_norm)
    return


@app.cell
def _(X_norm, n_jobs):
    from aeon.clustering import TimeSeriesKMeans

    clst = TimeSeriesKMeans(distance="dtw", averaging_method="ba", n_jobs=n_jobs)
    clst.fit(X_norm)
    return TimeSeriesKMeans, clst


@app.cell
def _(clst, only_weeks):
    preds = clst.predict(only_weeks)
    return (preds,)


@app.cell
def _(preds):
    preds
    return


@app.cell
def _(clst):
    clst.inertia_
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Deciding the ideal number of clusters

    Elbow curve
    """)
    return


@app.cell
def _(TimeSeriesKMeans, X_norm, n_jobs):
    sum_of_squared_distances = []

    for n_clusters in range(2, 11):
        km = TimeSeriesKMeans(
            n_clusters=n_clusters,
            distance="msm",
            averaging_method="ba",
            n_jobs=n_jobs,
            random_state=42,
        )
        km = km.fit(X_norm)
        sum_of_squared_distances.append(km.inertia_)

    return (sum_of_squared_distances,)


@app.cell
def _(sum_of_squared_distances):
    sum_of_squared_distances
    return


@app.cell
def _(px, sum_of_squared_distances):
    px.line(x=list(range(2, 11)), y=sum_of_squared_distances)
    return


@app.cell
def _():
    return


@app.cell
def _(preds):
    number_of_clusters = len(set(preds))
    number_of_clusters
    return (number_of_clusters,)


@app.cell
def _(clst, number_of_clusters, only_weeks):
    from aeon.visualisation import plot_cluster_algorithm

    plot_cluster_algorithm(clst, only_weeks, number_of_clusters)
    return


if __name__ == "__main__":
    app.run()
