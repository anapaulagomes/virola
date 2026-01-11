import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os

    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    from aeon.clustering import TimeSeriesKMedoids
    from aeon.distances import pairwise_distance
    from aeon.transformations.collection import MinMaxScaler
    from aeon.visualisation import plot_cluster_algorithm, plot_series_collection
    from dotenv import load_dotenv
    from sklearn.metrics import silhouette_samples, silhouette_score
    from sklearn.pipeline import make_pipeline

    return (
        MinMaxScaler,
        TimeSeriesKMedoids,
        go,
        load_dotenv,
        make_pipeline,
        np,
        os,
        pairwise_distance,
        pl,
        plot_cluster_algorithm,
        plot_series_collection,
        px,
        silhouette_samples,
        silhouette_score,
    )


@app.cell
def _():
    from vir.data_filters import filter_most_representative_rows
    from vir.notebook_widgets import (
        create_file_dropdown,
        extract_dataset_name,
        load_and_prepare_dataframe,
        load_indicative_codes,
        load_terminology_labels,
    )
    from vir.views import TemporalView

    return (
        TemporalView,
        create_file_dropdown,
        extract_dataset_name,
        filter_most_representative_rows,
        load_and_prepare_dataframe,
        load_indicative_codes,
        load_terminology_labels,
    )


@app.cell
def _(load_dotenv):
    _ = load_dotenv()
    return


@app.cell
def _(mo):
    args = mo.cli_args() if mo.app_meta().mode == "script" else {}
    return (args,)


@app.cell
def _(args):
    year_min = args.get("year_min", 2022)
    year_max = args.get("year_max", 2024)
    min_weeks = args.get("min_weeks", 12)
    return min_weeks, year_max, year_min


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
def _(load_terminology_labels):
    terminology_labels = load_terminology_labels()
    return (terminology_labels,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Select codes that are indicative
    """)
    return


@app.cell
def _(load_indicative_codes):
    include_codes, indicative_codes_enabled = load_indicative_codes()
    return (include_codes,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Prepare dataframe
    """)
    return


@app.cell
def _(args, create_file_dropdown):
    default_dataset = args.get("dataset", "salvador")
    files_dropdown = create_file_dropdown(default_pattern=default_dataset)
    return (files_dropdown,)


@app.cell
def _(files_dropdown):
    file_path = files_dropdown.value
    return (file_path,)


@app.cell
def _(
    file_path,
    include_codes,
    load_and_prepare_dataframe,
    mo,
    terminology_labels,
    year_max,
    year_min,
):
    df = load_and_prepare_dataframe(
        file_path=file_path,
        terminology_labels=terminology_labels,
        include_codes=include_codes,
        year_min=year_min,
        year_max=year_max,
    )
    if mo.app_meta().mode != "script":
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
def _(TemporalView, df, extract_dataset_name, file_path, min_weeks, n_jobs):
    # Create temporal view with MSM parameters
    temporal_view = TemporalView(
        df=df,
        dataset_name=extract_dataset_name(file_path),
        params={"min_weeks": min_weeks, "c": 1.0, "n_jobs": n_jobs},
    )

    # Get time series matrix and pivot table
    X = temporal_view.get_time_series()
    temporal_pivot = temporal_view.get_temporal_pivot()

    temporal_pivot
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
def _(mo):
    mo.md(r"""
    **Note**: Distance matrix will be computed after Grid Search with optimal c parameter.
    """)
    return


@app.cell
def _(MinMaxScaler, TimeSeriesKMedoids, X, make_pipeline):
    pipeline = make_pipeline(
        MinMaxScaler(), TimeSeriesKMedoids(distance="msm", random_state=42, n_clusters=3)
    )

    pipeline.fit(X)
    return (pipeline,)


@app.cell
def _(X, pipeline):
    pipeline.named_steps["timeserieskmedoids"].predict(X)
    return


@app.cell
def _(pipeline):
    pipeline.named_steps["timeserieskmedoids"].inertia_
    return


@app.cell
def _(MinMaxScaler, X):
    Xt = MinMaxScaler().fit_transform(X)
    return (Xt,)


@app.cell
def _(Xt, pipeline, plot_cluster_algorithm):
    _fig, _ = plot_cluster_algorithm(pipeline.named_steps["timeserieskmedoids"], Xt, 3)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Grid Search for optimal c parameter

    The `c` parameter in MSM controls the cost of split/merge operations.
    Low values make these operations cheaper (more flexibility),
    high values make them more expensive (more rigidity).

    Find the `c` that maximizes Silhouette Score for k=3 or k=4, as these are the natural number of clusters identified in the data.
    """)
    return


@app.cell
def _(mo):
    k_for_grid_search = mo.ui.slider(start=2, stop=5, value=3, label="k for c Grid Search")
    k_for_grid_search
    return (k_for_grid_search,)


@app.cell
def _(
    MinMaxScaler,
    TimeSeriesKMedoids,
    X,
    k_for_grid_search,
    make_pipeline,
    n_jobs,
    pairwise_distance,
    pl,
    silhouette_score,
):
    c_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    grid_search_results = []

    print(f"Grid Search for k={k_for_grid_search.value}...")
    print("-" * 60)

    for _c in c_values:
        _dist_matrix = pairwise_distance(X, method="msm", n_jobs=n_jobs, c=_c)
        _pipeline = make_pipeline(
            MinMaxScaler(),
            TimeSeriesKMedoids(
                n_clusters=k_for_grid_search.value,
                distance="msm",
                random_state=42,
                distance_params={"c": _c},
            ),
        )

        _pipeline.fit(_dist_matrix)
        _labels = _pipeline.named_steps["timeserieskmedoids"].labels_
        _silhouette = silhouette_score(_dist_matrix, _labels, metric="precomputed")

        grid_search_results.append(
            {"c": _c, "silhouette": _silhouette, "k": k_for_grid_search.value}
        )

        print(f"c={_c:6.2f} | Silhouette={_silhouette:.4f}")

    grid_df = pl.DataFrame(grid_search_results)
    best_c_row = grid_df.sort("silhouette", descending=True)[0]
    best_c = best_c_row["c"][0]
    best_silhouette = best_c_row["silhouette"][0]

    print("-" * 60)
    print(f"🏆 BEST c = {best_c} (Silhouette = {best_silhouette:.4f})")
    return best_c, grid_df


@app.cell
def _(grid_df, px):
    _fig = px.line(
        grid_df,
        x="c",
        y="silhouette",
        markers=True,
        title="Grid Search: Silhouette Score vs c parameter (MSM)",
        labels={"c": "Parameter c", "silhouette": "Silhouette Score"},
        color_discrete_sequence=["#E15759"],  # Tableau Red
    )
    _fig.update_traces(marker=dict(size=10))
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Distance matrix with optimal c
    """)
    return


@app.cell
def _(TemporalView, df, extract_dataset_name, files_dropdown):
    temporal_view_optimal = TemporalView(
        df=df,
        dataset_name=extract_dataset_name(files_dropdown.value),
    )

    # get optimal distance matrix (cached for reuse)
    optimal_distance_matrix = temporal_view_optimal.get_matrix()
    return (optimal_distance_matrix,)


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
    best_c,
    make_pipeline,
    optimal_distance_matrix,
    silhouette_score,
):
    silhouette_scores = []
    inertia_scores = []
    cluster_range = range(2, 5)

    print("Beginning clustering analysis with optimal distance matrix...")

    for _n_clusters in cluster_range:
        _pipeline = make_pipeline(
            MinMaxScaler(),
            TimeSeriesKMedoids(
                n_clusters=_n_clusters,
                distance="msm",
                random_state=42,
                distance_params={"c": best_c},
            ),
        )

        _pipeline.fit(optimal_distance_matrix)
        _labels = _pipeline.named_steps["timeserieskmedoids"].labels_

        _inertia = _pipeline.named_steps["timeserieskmedoids"].inertia_

        # SILHOUETTE: Compute using the SAME MSM distance matrix
        # This ensures mathematical consistency (don't use Euclidean to validate MSM)
        score = silhouette_score(optimal_distance_matrix, _labels, metric="precomputed")
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


@app.cell
def _(mo):
    mo.md(r"""
    ### Detailed Silhouette Score per Cluster

    This plot shows the **individual** silhouette score of each sample,
    grouped by cluster. Useful to identify:
    - Well-formed clusters (high and uniform scores)
    - Problematic clusters (low or negative scores)
    - Misclassified samples (negative scores = closer to another cluster)

    **How to interpret**:
    - Score > 0.5: Well-classified sample
    - Score 0.3-0.5: Reasonable classification
    - Score < 0.3: Weak classification
    - Score < 0: Sample probably in wrong cluster
    """)
    return


@app.cell
def _(go, np, silhouette_samples, TABLEAU_COLORS):
    def plot_silhouette_per_cluster(distance_matrix, labels, n_clusters):
        sample_silhouette_values = silhouette_samples(
            distance_matrix, labels, metric="precomputed"
        )

        # global average silhouette
        avg_silhouette = np.mean(sample_silhouette_values)

        _fig = go.Figure()

        y_lower = 10

        for i in range(n_clusters):
            cluster_silhouette_values = sample_silhouette_values[labels == i]
            cluster_silhouette_values = np.sort(cluster_silhouette_values)

            cluster_size = cluster_silhouette_values.shape[0]
            y_upper = y_lower + cluster_size

            _fig.add_trace(
                go.Bar(
                    x=cluster_silhouette_values,
                    y=list(range(y_lower, y_upper)),
                    orientation="h",
                    name=f"Cluster {i}",
                    marker=dict(color=TABLEAU_COLORS[i % len(TABLEAU_COLORS)]),
                    hovertemplate="Cluster %{fullData.name}<br>Silhouette: %{x:.3f}<extra></extra>",
                )
            )

            y_lower = y_upper + 10  # Space between clusters

        # add vertical line for average
        _fig.add_vline(
            x=avg_silhouette,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Average: {avg_silhouette:.3f}",
            annotation_position="top",
        )

        _fig.update_layout(
            title=f"Silhouette Score per Sample (k={n_clusters})",
            xaxis_title="Silhouette Score",
            yaxis_title="Samples (grouped by cluster)",
            template="plotly_white",
            showlegend=True,
            height=600,
            bargap=0,
        )
        return _fig

    return (plot_silhouette_per_cluster,)


@app.cell
def _(mo):
    k_for_silhouette_detail = mo.ui.slider(
        start=2, stop=10, value=3, label="k for detailed Silhouette plot"
    )
    k_for_silhouette_detail
    return (k_for_silhouette_detail,)


@app.cell
def _(
    MinMaxScaler,
    TimeSeriesKMedoids,
    best_c,
    k_for_silhouette_detail,
    make_pipeline,
    optimal_distance_matrix,
    plot_silhouette_per_cluster,
):
    _pipeline_detail = make_pipeline(
        MinMaxScaler(),
        TimeSeriesKMedoids(
            n_clusters=k_for_silhouette_detail.value,
            distance="msm",
            random_state=42,
            distance_params={"c": best_c},
        ),
    )

    _pipeline_detail.fit(optimal_distance_matrix)
    _labels_detail = _pipeline_detail.named_steps["timeserieskmedoids"].labels_

    _fig = plot_silhouette_per_cluster(
        optimal_distance_matrix, _labels_detail, k_for_silhouette_detail.value
    )
    _fig
    return


@app.cell
def _(mo):
    if mo.app_meta().mode == "script":
        print("✓ Temporal view analysis completed")
    return


if __name__ == "__main__":
    app.run()
