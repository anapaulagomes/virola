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
    import plotly.graph_objects as go
    import polars as pl
    from aeon.clustering import TimeSeriesKMedoids
    from aeon.distances import pairwise_distance
    from aeon.transformations.collection import MinMaxScaler
    from aeon.visualisation import plot_cluster_algorithm, plot_series_collection
    from dotenv import load_dotenv
    from sklearn.metrics import silhouette_score
    from sklearn.pipeline import make_pipeline

    return (
        MinMaxScaler,
        TimeSeriesKMedoids,
        go,
        load_dotenv,
        make_pipeline,
        os,
        pairwise_distance,
        pl,
        plot_cluster_algorithm,
        plot_series_collection,
        px,
        silhouette_score,
    )


@app.cell
def _():
    from vir.config import INDICATIVE_MODEL_RESULTS, INTERIM_DATA_DIR
    from vir.data_filters import filter_most_representative_rows
    from vir.db import DEFAULT_MODEL, fetch_terminologies

    return (
        DEFAULT_MODEL,
        INDICATIVE_MODEL_RESULTS,
        INTERIM_DATA_DIR,
        fetch_terminologies,
        filter_most_representative_rows,
    )


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
def _(DEFAULT_MODEL, fetch_terminologies):
    terminology_labels = fetch_terminologies(DEFAULT_MODEL)
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
def _(INDICATIVE_MODEL_RESULTS, indicative_codes_enabled, pl):
    include_codes = []
    if indicative_codes_enabled:
        indicative_codes = pl.read_csv(INDICATIVE_MODEL_RESULTS)
        for _data in indicative_codes.iter_rows(named=True):
            if _data["indicative_prob"] > 0.50:
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


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _(df, filter_most_representative_rows):
    top_10 = filter_most_representative_rows(df).top_k(k=10, by="quantity")
    return (top_10,)


@app.cell
def _(df, pl, px, top_10):
    _fig = px.line(
        df.filter(pl.col("terminology_label").is_in(top_10["terminology_label"]))
        .group_by(["terminology_label", "year", "week"])
        .agg(pl.col("quantity").sum())
        .sort(["year", "week"]),
        x="week",
        y="quantity",
        color="terminology_label",
        markers=True,
        facet_row="year",
        # log_y=True,
    )
    _fig
    return


@app.cell
def _(df, pl):
    more_than_12_weeks = (
        df.group_by("terminology_id")
        .agg(pl.col("week").count().alias("week_count"))
        .filter(pl.col("week_count").ge(12))
    )

    df_week = (
        df.filter(pl.col("terminology_id").is_in(more_than_12_weeks["terminology_id"]))
        .group_by(["terminology_id", "week"])
        .agg(pl.col("quantity").sum().alias("total"))
    )

    df_week
    return (df_week,)


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
def _(ordered_week, temporal_pivot):
    X = temporal_pivot.select(ordered_week).to_numpy()
    return (X,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Clustering

    The time series were aggregated by week, so we can capture cyclic behaviour.
    Now it is time to normalize, so less frequent diagnosis can be part of the representation.
    """)
    return


@app.cell
def _(X, plot_series_collection):
    plot_series_collection(X)
    return


@app.cell
def _(X, n_jobs, pairwise_distance):
    distance_matrix = pairwise_distance(X, method="msm", n_jobs=n_jobs, c=1)
    return (distance_matrix,)


@app.cell
def _(MinMaxScaler, TimeSeriesKMedoids, X, make_pipeline):
    pipeline = make_pipeline(
        MinMaxScaler(), TimeSeriesKMedoids(distance="msm", random_state=42, n_clusters=3)
    )

    pipeline.fit(X)
    return (pipeline,)


@app.cell
def _(X, pipeline):
    preds = pipeline.named_steps["timeserieskmedoids"].predict(X)
    return (preds,)


@app.cell
def _(preds):
    preds
    return


@app.cell
def _(pipeline):
    pipeline.named_steps["timeserieskmedoids"].inertia_
    return


@app.cell
def _(X, number_of_clusters, pipeline, plot_cluster_algorithm):
    _fig, _ = plot_cluster_algorithm(
        pipeline.named_steps["timeserieskmedoids"], X, number_of_clusters
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Deciding the ideal number of clusters

    * Elbow curve
    * Silhouette score
    """)
    return


@app.cell
def _(go):
    def plot_silhouette_vs_inertia(cluster_range, silhouette_scores, inertia_scores):
        _fig = go.Figure()

        _fig.add_trace(
            go.Scatter(
                x=list(cluster_range),
                y=silhouette_scores,
                name="Silhouette Score",
                mode="lines+markers",
                marker=dict(size=10, color="firebrick"),
                line=dict(width=3),
            )
        )

        _fig.add_trace(
            go.Scatter(
                x=list(cluster_range),
                y=inertia_scores,
                name="Inertia (distances sum)",
                mode="lines+markers",
                marker=dict(size=8, color="royalblue", symbol="diamond"),
                line=dict(width=2, dash="dot"),
                yaxis="y2",
            )
        )

        _fig.update_layout(
            title="Silhouette vs Inertia (MSM)",
            xaxis_title="Nr. of clusters (k)",
            yaxis=dict(title="Silhouette Score"),
            yaxis2=dict(title="Inertia", overlaying="y", side="right", showgrid=False),
            template="plotly_white",
            hovermode="x unified",
            legend=dict(x=0.5, y=1.1, orientation="h", xanchor="center"),
        )
        return _fig

    return (plot_silhouette_vs_inertia,)


@app.cell
def _(
    MinMaxScaler,
    TimeSeriesKMedoids,
    distance_matrix,
    make_pipeline,
    silhouette_score,
):
    silhouette_scores = []
    inertia_scores = []
    cluster_range = range(2, 11)

    print("Beginning clustering analysis...")

    for _n_clusters in cluster_range:
        _pipeline = make_pipeline(
            MinMaxScaler(),
            TimeSeriesKMedoids(
                n_clusters=_n_clusters, distance="msm", random_state=42, distance_params={"c": 0.1}
            ),
        )

        _pipeline.fit(distance_matrix)
        _labels = _pipeline.named_steps["timeserieskmedoids"].labels_
        _inertia = _pipeline.named_steps["timeserieskmedoids"].inertia_

        # SILHOUETTE: Calculamos usando a MESMA matriz de distâncias MSM
        # Isso garante consistência matemática (não usar Euclidiana para validar MSM)
        score = silhouette_score(distance_matrix, _labels, metric="precomputed")
        silhouette_scores.append(score)
        inertia_scores.append(_inertia)

        print(f"k={_n_clusters} | Silhouette MSM: {score:.4f}")
    return cluster_range, inertia_scores, silhouette_scores


@app.cell
def _(
    cluster_range,
    inertia_scores,
    plot_silhouette_vs_inertia,
    silhouette_scores,
):
    plot_silhouette_vs_inertia(cluster_range, silhouette_scores, inertia_scores)
    return


if __name__ == "__main__":
    app.run()
