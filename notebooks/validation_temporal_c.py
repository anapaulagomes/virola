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
    from aeon.distances import pairwise_distance
    from aeon.transformations.collection import MinMaxScaler
    from dotenv import load_dotenv
    from sklearn.metrics import silhouette_score

    return (
        MinMaxScaler,
        go,
        load_dotenv,
        np,
        pairwise_distance,
        pl,
        px,
        silhouette_score,
    )


@app.cell
def _():
    from vir.config import VALIDATION_DIR
    from vir.helpers import extract_dataset_name, get_n_jobs
    from vir.notebook_widgets import (
        create_file_dropdown,
        load_and_prepare_dataframe,
    )
    from vir.views.temporal import TemporalView

    return (
        TemporalView,
        VALIDATION_DIR,
        create_file_dropdown,
        extract_dataset_name,
        get_n_jobs,
        load_and_prepare_dataframe,
    )


@app.cell
def _(load_dotenv):
    _ = load_dotenv()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Validation: MSM cost parameter $c$

    The Move-Split-Merge (MSM) distance has a single tunable parameter $c$
    that controls the cost of **split** and **merge** operations relative to
    **move** operations.

    - **Low $c$** (e.g. 0.01): splits and merges are cheap, allowing aggressive
      warping that can absorb large amplitude differences.  Risk: unrelated
      series look similar because structural differences are ignored.
    - **High $c$** (e.g. 10.0): splits and merges are expensive, making the
      metric behave more like a rigid point-wise comparison.  Risk: similar
      patterns with minor timing shifts appear distant.
    - **$c = 1.0$**: the default value proposed in the original MSM paper, which
      balances flexibility and rigidity.

    ## Literature

    1. **Stefan, Athitsos & Das (2013)** — *"The Move-Split-Merge metric for
       time series"*, IEEE TKDE 25(6).  Introduced MSM and recommended $c = 1.0$
       as the default cost, showing competitive accuracy on the UCR archive
       without per-dataset tuning.

    2. **Shifaz et al. (2023)** — *"Elastic Similarity and Distance Measures for
       Multivariate Time Series"*, Machine Learning 112.  Benchmarked elastic
       distances (including MSM) across 128 UCR datasets in the aeon/tsml
       framework and confirmed that $c = 1.0$ is a robust default: tuning $c$
       produced marginal gains on average while adding significant computational
       cost.

    3. **Herrmann & Bagnall (2023)** — *"Proximity Forest 2.0"*, Data Mining and
       Knowledge Discovery 37.  Used MSM with $c = 1.0$ as one of the core
       distance measures in the state-of-the-art proximity forest classifier,
       further validating the default.

    ## Goal

    Run a grid search over $c \in \{0.01, 0.1, 0.5, 1.0, 5.0, 10.0\}$ for
    several values of $k$ (number of clusters) and verify that $c = 1.0$
    maximises (or is within margin of) the silhouette score for our
    epidemiological time series.
    """)
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
def _(file_path, load_and_prepare_dataframe, year_max, year_min):
    df = load_and_prepare_dataframe(
        file_path=file_path,
        year_min=year_min,
        year_max=year_max,
    )
    return (df,)


@app.cell
def _(TemporalView, df, extract_dataset_name, file_path, min_weeks):
    temporal_view = TemporalView(
        df=df,
        dataset_name=extract_dataset_name(file_path),
        params={"min_weeks": min_weeks},
    )
    time_series_matrix = temporal_view.get_time_series()
    return (time_series_matrix,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Grid search: distance matrices per $c$

    Pre-compute the pairwise MSM distance matrix for each candidate $c$ value.
    This is the expensive step — it runs once regardless of how many $k$ values
    are explored afterwards.
    """)
    return


@app.cell
def _(MinMaxScaler, n_jobs, np, pairwise_distance, time_series_matrix):
    _normalized_matrix = MinMaxScaler().fit_transform(time_series_matrix)

    c_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    distance_matrices_by_c = {}

    for _c_value in c_values:
        _distance_matrix = pairwise_distance(
            _normalized_matrix, method="msm", n_jobs=n_jobs, c=_c_value
        ).astype(np.float32)
        distance_matrices_by_c[_c_value] = _distance_matrix
    return c_values, distance_matrices_by_c


@app.cell
def _(mo):
    mo.md(r"""
    ## Clustering evaluation

    For each $(c, k)$ pair, run k-medoids on the precomputed distance matrix
    and record the silhouette score.
    """)
    return


@app.cell
def _(np):
    def kmedoids_precomputed(
        distance_matrix, n_clusters, random_state=42, n_init=10, max_iter=100
    ):
        rng = np.random.RandomState(random_state)
        n = len(distance_matrix)
        best_labels, best_inertia = None, np.inf
        for _ in range(n_init):
            medoids = rng.choice(n, n_clusters, replace=False)
            for _ in range(max_iter):
                labels = distance_matrix[:, medoids].argmin(axis=1)
                new_medoids = medoids.copy()
                for k in range(n_clusters):
                    idx = np.where(labels == k)[0]
                    if len(idx) > 0:
                        new_medoids[k] = idx[distance_matrix[np.ix_(idx, idx)].sum(0).argmin()]
                if np.array_equal(new_medoids, medoids):
                    break
                medoids = new_medoids
            inertia = distance_matrix[:, medoids].min(axis=1).sum()
            if inertia < best_inertia:
                best_inertia, best_labels = inertia, labels
        return best_labels, best_inertia

    return (kmedoids_precomputed,)


@app.cell
def _(
    c_values,
    distance_matrices_by_c,
    kmedoids_precomputed,
    pl,
    silhouette_score,
):
    k_range = list(range(2, 6))
    _grid_search_results = []

    for _c_value in c_values:
        _distance_matrix = distance_matrices_by_c[_c_value]
        for _k in k_range:
            _labels, _ = kmedoids_precomputed(_distance_matrix, _k)
            _score = silhouette_score(_distance_matrix, _labels, metric="precomputed")
            _grid_search_results.append({"c": _c_value, "k": _k, "silhouette": _score})

    grid_search_df = pl.DataFrame(_grid_search_results)
    grid_search_df
    return grid_search_df, k_range


@app.cell
def _(mo):
    mo.md(r"""
    ## Results
    """)
    return


@app.cell
def _(grid_search_df, px):
    _fig = px.line(
        grid_search_df,
        x="c",
        y="silhouette",
        color="k",
        markers=True,
        log_x=True,
        title="Silhouette Score vs MSM cost parameter c",
        labels={"c": "Parameter c (log scale)", "silhouette": "Silhouette Score", "k": "k"},
    )
    _fig.update_traces(marker=dict(size=10))
    _fig.update_layout(template="plotly_white")
    _fig
    return


@app.cell
def _(go, grid_search_df, k_range, np, pl):
    _pivot = grid_search_df.pivot(index="c", on="k", values="silhouette").sort("c")
    _c_labels = [str(c) for c in _pivot["c"].to_list()]
    _k_labels = [str(k) for k in k_range]
    _z_values = _pivot.select(pl.exclude("c")).to_numpy()

    _best_row = grid_search_df.sort("silhouette", descending=True)[0]
    best_c = _best_row["c"][0]
    best_k = _best_row["k"][0]
    best_silhouette = _best_row["silhouette"][0]

    _c1_scores = grid_search_df.filter(pl.col("c") == 1.0)
    _c1_best = _c1_scores.sort("silhouette", descending=True)[0]["silhouette"][0]
    delta = best_silhouette - _c1_best

    _fig = go.Figure(
        data=go.Heatmap(
            z=_z_values,
            x=_k_labels,
            y=_c_labels,
            colorscale="RdYlGn",
            text=np.round(_z_values, 3),
            texttemplate="%{text}",
            colorbar=dict(title="Silhouette"),
        )
    )
    _fig.update_layout(
        title=f"Silhouette heatmap (c × k) — best: c={best_c}, k={best_k} (Δ from c=1.0: {delta:+.4f})",
        xaxis_title="Number of clusters (k)",
        yaxis_title="MSM cost parameter c",
        template="plotly_white",
    )
    _fig
    return best_c, best_k, best_silhouette, delta


@app.cell
def _(VALIDATION_DIR, extract_dataset_name, file_path, grid_search_df):
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    _dataset_name = extract_dataset_name(file_path)
    output_file = VALIDATION_DIR / f"temporal_grid_search_c_{_dataset_name}.json"
    grid_search_df.write_json(output_file)
    return (output_file,)


@app.cell
def _(best_c, best_k, best_silhouette, delta, mo, output_file):
    mo.md(rf"""
    ## Conclusion

    | Metric | Value |
    |--------|-------|
    | Best $c$ | {best_c} |
    | Best $k$ | {best_k} |
    | Silhouette | {best_silhouette:.4f} |
    | $\Delta$ vs $c = 1.0$ | {delta:+.4f} |

    Results saved to `{output_file}`.

    {"**$c = 1.0$ is optimal or within negligible margin**, consistent with the literature recommendation (Stefan et al. 2013, Shifaz et al. 2023)." if abs(delta) < 0.02 else f"**$c = {best_c}$ outperforms $c = 1.0$ by {delta:.4f}** — consider updating the default."}
    """)
    return


@app.cell
def _(mo):
    if mo.app_meta().mode == "script":
        mo.md("Grid search completed")
    return


if __name__ == "__main__":
    app.run()
