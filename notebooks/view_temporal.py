import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    from aeon.visualisation import plot_series_collection
    from dotenv import load_dotenv
    from sklearn.metrics import silhouette_samples, silhouette_score

    return (
        go,
        load_dotenv,
        np,
        pl,
        plot_series_collection,
        px,
        silhouette_samples,
        silhouette_score,
    )


@app.cell
def _():
    from vir.data_filters import filter_most_representative_rows
    from vir.helpers import extract_dataset_name, get_n_jobs
    from vir.notebook_widgets import (
        create_file_dropdown,
        load_and_prepare_dataframe,
    )
    from vir.views import TemporalView
    from vir.viz import TABLEAU_COLORS

    return (
        TABLEAU_COLORS,
        TemporalView,
        create_file_dropdown,
        extract_dataset_name,
        filter_most_representative_rows,
        get_n_jobs,
        load_and_prepare_dataframe,
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
    year_min = args.get("year-min", 2022)
    year_max = args.get("year-max", 2025)
    min_weeks = args.get("min-weeks", 12)
    return min_weeks, year_max, year_min


@app.cell
def _(get_n_jobs):
    n_jobs = get_n_jobs()
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
    files_dropdown
    return


@app.cell
def _(files_dropdown):
    file_path = files_dropdown.value
    return (file_path,)


@app.cell
def _(file_path, load_and_prepare_dataframe, mo, year_max, year_min):
    df = load_and_prepare_dataframe(
        file_path=file_path,
        year_min=year_min,
        year_max=year_max,
    )
    if mo.app_meta().mode != "script":
        df.head()
    return (df,)


@app.cell
def _(df):
    df.shape
    return


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
        df.filter(pl.col("terminology_label").is_in(top_10["terminology_label"].implode()))
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
    # Columns are sequential ints (chronological year_week order)
    # Use get_col_labels() for human-readable YYYY-WW labels
    X = temporal_view.get_time_series()
    temporal_pivot = temporal_view.get_temporal_pivot()
    col_labels = temporal_view.get_col_labels()

    print(f"Series length: {X.shape[1]} year_weeks ({col_labels[0]} → {col_labels[-1]})")
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
def _(TemporalView, df, extract_dataset_name, files_dropdown, min_weeks, n_jobs):
    temporal_view_optimal = TemporalView(
        df=df,
        dataset_name=extract_dataset_name(files_dropdown.value),
        params={"min_weeks": min_weeks, "c": 1.0, "n_jobs": n_jobs},
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
def _(np):
    def kmedoids_precomputed(D, n_clusters, random_state=42, n_init=10, max_iter=100):
        """K-medoids on a precomputed N×N distance matrix. O(N × k × n_init × max_iter)."""
        rng = np.random.RandomState(random_state)
        n = len(D)
        best_labels, best_inertia = None, np.inf
        for _ in range(n_init):
            medoids = rng.choice(n, n_clusters, replace=False)
            for _ in range(max_iter):
                labels = D[:, medoids].argmin(axis=1)
                new_medoids = medoids.copy()
                for k in range(n_clusters):
                    idx = np.where(labels == k)[0]
                    if len(idx) > 0:
                        new_medoids[k] = idx[D[np.ix_(idx, idx)].sum(0).argmin()]
                if np.array_equal(new_medoids, medoids):
                    break
                medoids = new_medoids
            inertia = D[:, medoids].min(axis=1).sum()
            if inertia < best_inertia:
                best_inertia, best_labels = inertia, labels
        return best_labels, best_inertia

    return (kmedoids_precomputed,)


@app.cell
def _(kmedoids_precomputed, optimal_distance_matrix, silhouette_score):
    silhouette_scores = []
    inertia_scores = []
    cluster_range = range(2, 5)

    print("Beginning clustering analysis with optimal distance matrix...")

    for _n_clusters in cluster_range:
        _labels, _inertia = kmedoids_precomputed(optimal_distance_matrix, _n_clusters)
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
def _(TABLEAU_COLORS, go, np, silhouette_samples):
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
    k_for_silhouette_detail,
    kmedoids_precomputed,
    optimal_distance_matrix,
    plot_silhouette_per_cluster,
):
    _labels_detail, _ = kmedoids_precomputed(
        optimal_distance_matrix, k_for_silhouette_detail.value
    )
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
