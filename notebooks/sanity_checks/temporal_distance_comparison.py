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
    from scipy.stats import spearmanr
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
        spearmanr,
    )


@app.cell
def _():
    from vir.db import get_connection
    from vir.helpers import extract_dataset_name, get_n_jobs
    from vir.notebook_widgets import (
        create_file_dropdown,
        load_and_prepare_dataframe,
    )
    from vir.views.temporal import TemporalView

    return (
        TemporalView,
        create_file_dropdown,
        extract_dataset_name,
        get_connection,
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
    # Distance Metric Comparison for Temporal View

    Compares four elastic distance measures on the same temporal data to
    understand how each captures epidemiological similarity:

    | Method | Key property |
    |--------|-------------|
    | **MSM** | Penalises split/merge of peaks — current default |
    | **DTW** | Classic elastic alignment — over-warps amplitude changes |
    | **WDTW** | DTW with weight bias towards diagonal — reduces pathological warping |
    | **ShapeDTW** | DTW on shape descriptors — captures local shape rather than raw values |

    All distances are computed on **minmax-normalised** series (same as the
    production TemporalView) so results are directly comparable.
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
    _default = args.get("dataset", "salvador")
    files_dropdown = create_file_dropdown(default_pattern=_default)
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
    terminology_ids = temporal_view.get_terminology_ids()
    col_labels = temporal_view.get_col_labels()
    return col_labels, temporal_view, terminology_ids, time_series_matrix


@app.cell
def _(MinMaxScaler, time_series_matrix):
    normalized_matrix = MinMaxScaler().fit_transform(time_series_matrix)
    return (normalized_matrix,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Computing pairwise distances

    Each method runs on the same minmax-normalised matrix. This takes a while
    for large datasets — results are displayed as they complete.
    """)
    return


@app.cell
def _(n_jobs, normalized_matrix, np, pairwise_distance):
    DISTANCE_CONFIGS = {
        "MSM": ("msm", {"c": 1.0}),
        "DTW": ("dtw", {}),
        "WDTW": ("wdtw", {}),
        "ShapeDTW": ("shape_dtw", {}),
    }

    distance_matrices = {}
    for _name, (_method, _kwargs) in DISTANCE_CONFIGS.items():
        _matrix = pairwise_distance(
            normalized_matrix, method=_method, n_jobs=n_jobs, **_kwargs
        ).astype(np.float32)
        distance_matrices[_name] = _matrix
    return (distance_matrices,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Distance distributions
    """)
    return


@app.cell
def _(distance_matrices, go, mo, np):
    _dist_figs = []
    for _name, _matrix in distance_matrices.items():
        _upper = _matrix[np.triu_indices_from(_matrix, k=1)]
        _counts, _bin_edges = np.histogram(_upper, bins=80)
        _bin_centers = (_bin_edges[:-1] + _bin_edges[1:]) / 2
        _median = float(np.median(_upper))

        _fig = go.Figure(
            data=go.Bar(x=_bin_centers, y=_counts, width=_bin_edges[1] - _bin_edges[0])
        )
        _fig.add_vline(
            x=_median,
            line_dash="dash",
            annotation_text=f"median={_median:.2f}",
        )
        _fig.update_layout(
            title=f"{_name} distance distribution",
            xaxis_title="Distance",
            yaxis_title="Count",
            template="plotly_white",
            height=280,
        )
        _dist_figs.append(_fig)

    mo.vstack([mo.hstack(_dist_figs[:2]), mo.hstack(_dist_figs[2:])])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Correlation between methods

    Spearman rank correlation on the upper-triangle distances — measures whether
    methods agree on the **ordering** of pairs (which pairs are more/less similar),
    regardless of absolute scale.
    """)
    return


@app.cell
def _(distance_matrices, np, px, spearmanr):
    _method_names = list(distance_matrices.keys())
    _uppers = {
        _name: _matrix[np.triu_indices_from(_matrix, k=1)]
        for _name, _matrix in distance_matrices.items()
    }

    _n = len(_method_names)
    _corr_matrix = np.ones((_n, _n))
    for _i in range(_n):
        for _j in range(_i + 1, _n):
            _rho, _ = spearmanr(_uppers[_method_names[_i]], _uppers[_method_names[_j]])
            _corr_matrix[_i, _j] = _rho
            _corr_matrix[_j, _i] = _rho

    _fig = px.imshow(
        _corr_matrix,
        x=_method_names,
        y=_method_names,
        color_continuous_scale="RdYlGn",
        zmin=0.5,
        zmax=1.0,
        text_auto=".3f",
        title="Spearman rank correlation between distance methods",
        labels={"color": "ρ"},
    )
    _fig.update_layout(height=400, template="plotly_white")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Silhouette comparison (k-medoids)

    For each method, run k-medoids on the precomputed distance matrix across
    a range of k values and compare silhouette scores.
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
def _(distance_matrices, kmedoids_precomputed, pl, px, silhouette_score):
    _k_range = list(range(2, 6))
    _sil_rows = []

    for _name, _matrix in distance_matrices.items():
        for _k in _k_range:
            _labels, _ = kmedoids_precomputed(_matrix, _k)
            _score = silhouette_score(_matrix, _labels, metric="precomputed")
            _sil_rows.append({"method": _name, "k": _k, "silhouette": round(_score, 4)})

    silhouette_df = pl.DataFrame(_sil_rows)

    _fig = px.line(
        silhouette_df,
        x="k",
        y="silhouette",
        color="method",
        markers=True,
        title="Silhouette Score by method and k",
        labels={"k": "Number of clusters", "silhouette": "Silhouette Score", "method": "Method"},
    )
    _fig.update_layout(template="plotly_white")
    _fig
    return (silhouette_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Interactive comparison: pick codes and compare

    Select diagnosis codes and see their time series side by side, plus the
    pairwise distance that each method assigns to every selected pair.
    """)
    return


@app.cell
def _(get_connection):
    _connection = get_connection()
    code_lookup = _connection.execute(
        "SELECT id, code, code_type, description FROM medical_terminologies ORDER BY id"
    ).pl()
    return (code_lookup,)


@app.cell
def _(code_lookup, df, mo, pl):
    _available_ids = df["terminology_id"].unique()
    _available_codes = code_lookup.filter(pl.col("id").is_in(_available_ids))

    _opts = {
        f"{r['code']} [{r['code_type']}] {r['description'][:50]}": r["id"]
        for r in _available_codes.iter_rows(named=True)
    }
    code_selector = mo.ui.multiselect(
        options=_opts,
        label="Select codes to compare (2-8)",
        max_selections=8,
    )
    code_selector
    return (code_selector,)


@app.cell
def _(
    code_lookup,
    code_selector,
    col_labels,
    distance_matrices,
    mo,
    np,
    pl,
    px,
    temporal_view,
    terminology_ids,
):
    mo.stop(len(code_selector.value) < 2, mo.md("*Select at least 2 codes above.*"))

    _selected_ids = list(code_selector.value)
    _id_to_idx = {tid: idx for idx, tid in enumerate(terminology_ids)}
    _ts_matrix = temporal_view.get_time_series()

    _plot_rows = []
    for _tid in _selected_ids:
        _idx = _id_to_idx.get(_tid)
        if _idx is None:
            continue
        _meta = code_lookup.filter(pl.col("id") == _tid)
        if _meta.is_empty():
            continue
        _label = f"{_meta['code'][0]} [{_meta['code_type'][0]}] {_meta['description'][0][:35]}"
        for _t, _val in enumerate(_ts_matrix[_idx]):
            _plot_rows.append(
                {
                    "code": _label,
                    "year_week": col_labels[_t],
                    "value": float(_val),
                }
            )

    _ts_fig = px.line(
        pl.DataFrame(_plot_rows),
        x="year_week",
        y="value",
        color="code",
        title="Time series of selected codes",
        labels={"year_week": "Year-Week", "value": "Raw count", "code": ""},
    )
    _ts_fig.update_layout(height=400, xaxis_tickangle=45, template="plotly_white")

    _dist_rows = []
    _valid_ids = [_tid for _tid in _selected_ids if _tid in _id_to_idx]
    for _i in range(len(_valid_ids)):
        for _j in range(_i + 1, len(_valid_ids)):
            _idx_a = _id_to_idx[_valid_ids[_i]]
            _idx_b = _id_to_idx[_valid_ids[_j]]
            _meta_a = code_lookup.filter(pl.col("id") == _valid_ids[_i])
            _meta_b = code_lookup.filter(pl.col("id") == _valid_ids[_j])
            _label_a = _meta_a["code"][0] if not _meta_a.is_empty() else str(_valid_ids[_i])
            _label_b = _meta_b["code"][0] if not _meta_b.is_empty() else str(_valid_ids[_j])
            _row = {"pair": f"{_label_a} ↔ {_label_b}"}
            for _name, _matrix in distance_matrices.items():
                _row[_name] = round(float(_matrix[_idx_a, _idx_b]), 4)
            _dist_rows.append(_row)

    _dist_df = pl.DataFrame(_dist_rows)

    _rank_rows = []
    for _row in _dist_rows:
        _rank_row = {"pair": _row["pair"]}
        _values = [(k, v) for k, v in _row.items() if k != "pair"]
        _sorted_methods = sorted(_values, key=lambda x: x[1])
        for _rank, (_method, _val) in enumerate(_sorted_methods, 1):
            _rank_row[_method] = f"#{_rank} ({_val:.3f})"
        _rank_rows.append(_rank_row)
    _rank_df = pl.DataFrame(_rank_rows)

    _bar_rows = []
    for _row in _dist_rows:
        for _method in distance_matrices:
            _bar_rows.append(
                {
                    "pair": _row["pair"],
                    "method": _method,
                    "distance": _row[_method],
                }
            )

    _bar_fig = px.bar(
        pl.DataFrame(_bar_rows),
        x="distance",
        y="pair",
        color="method",
        barmode="group",
        orientation="h",
        title="Distance by method for each pair",
        labels={"distance": "Distance", "pair": "", "method": "Method"},
    )
    _bar_fig.update_layout(height=max(300, len(_dist_rows) * 80), template="plotly_white")

    mo.vstack(
        [
            _ts_fig,
            mo.md("### Pairwise distances"),
            _dist_df,
            mo.md("### Rankings (which method sees each pair as most/least similar)"),
            _rank_df,
            _bar_fig,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
