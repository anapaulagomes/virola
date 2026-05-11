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
    from vir.config import VALIDATION_DIR
    from vir.db import get_connection
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
    year_max = args.get("year-max", 2024)
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
def _(get_connection):
    _connection = get_connection()
    code_lookup = _connection.execute(
        "SELECT id, code, code_type, description FROM medical_terminologies ORDER BY id"
    ).pl()
    return (code_lookup,)


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
def _(distance_matrices, np):
    p90_per_method = {}
    for _name, _matrix in distance_matrices.items():
        _upper = _matrix[np.triu_indices_from(_matrix, k=1)]
        p90_per_method[_name] = float(np.percentile(_upper, 90))
    return (p90_per_method,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Distance distributions (P90-normalized)

    All distances are divided by their method's 90th percentile, placing
    every method on a comparable [0, ~1] scale. Values above 1.0 are the
    10% most distant pairs.
    """)
    return


@app.cell
def _(distance_matrices, go, mo, np, p90_per_method):
    _dist_figs = []
    for _name, _matrix in distance_matrices.items():
        _upper = _matrix[np.triu_indices_from(_matrix, k=1)]
        _p90 = p90_per_method[_name]
        _normalized = _upper / _p90 if _p90 > 0 else _upper
        _counts, _bin_edges = np.histogram(_normalized, bins=80)
        _bin_centers = (_bin_edges[:-1] + _bin_edges[1:]) / 2
        _median = float(np.median(_normalized))

        _fig = go.Figure(
            data=go.Bar(x=_bin_centers, y=_counts, width=_bin_edges[1] - _bin_edges[0])
        )
        _fig.add_vline(
            x=_median,
            line_dash="dash",
            annotation_text=f"median={_median:.2f}",
        )
        _fig.add_vline(x=1.0, line_dash="dot", line_color="red", annotation_text="P90")
        _fig.update_layout(
            title=f"{_name} (P90={_p90:.2f})",
            xaxis_title="Distance / P90",
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
    ## Structural separation test

    Does a method assign **higher** distances to pairs with opposite temporal
    structure than to pairs with similar structure?

    Pairs are loaded from `data/validation/sanity_pairs.csv`.
    All distances are P90-normalized so values are comparable across methods.
    The **separation ratio** = mean(opposite) / mean(similar). Higher = better.
    """)
    return


@app.cell
def _(VALIDATION_DIR, code_lookup, pl):
    _pairs_df = pl.read_csv(VALIDATION_DIR / "sanity_pairs.csv")

    _resolved = []
    for _row in _pairs_df.iter_rows(named=True):
        _meta_a = code_lookup.filter(
            (pl.col("code") == _row["code_a"]) & (pl.col("code_type") == _row["code_type_a"])
        )
        _meta_b = code_lookup.filter(
            (pl.col("code") == _row["code_b"]) & (pl.col("code_type") == _row["code_type_b"])
        )
        if _meta_a.is_empty() or _meta_b.is_empty():
            continue
        _resolved.append(
            {
                "id_a": _meta_a["id"][0],
                "id_b": _meta_b["id"][0],
                "code_a": _row["code_a"],
                "code_b": _row["code_b"],
                "expected": _row["expected"],
                "rationale": _row["rationale"],
            }
        )
    structural_resolved = _resolved
    return (structural_resolved,)


@app.cell
def _(distance_matrices, mo, np, p90_per_method, pl, px, structural_resolved, terminology_ids):
    mo.stop(len(structural_resolved) < 2)

    _id_to_idx = {tid: idx for idx, tid in enumerate(terminology_ids)}
    _method_names = list(distance_matrices.keys())

    _rows = []
    for _pair in structural_resolved:
        _idx_a = _id_to_idx.get(_pair["id_a"])
        _idx_b = _id_to_idx.get(_pair["id_b"])
        if _idx_a is None or _idx_b is None:
            continue
        _row = {
            "pair": f"{_pair['code_a']} vs {_pair['code_b']}",
            "expected": _pair["expected"],
            "rationale": _pair["rationale"],
        }
        for _name, _matrix in distance_matrices.items():
            _raw = float(_matrix[_idx_a, _idx_b])
            _p90 = p90_per_method[_name]
            _row[_name] = round(_raw / _p90, 4) if _p90 > 0 else 0.0
        _rows.append(_row)

    _results_df = pl.DataFrame(_rows)

    _summary_rows = []
    for _method in _method_names:
        _similar = _results_df.filter(pl.col("expected") == "similar")[_method].to_numpy()
        _opposite = _results_df.filter(pl.col("expected") == "opposite")[_method].to_numpy()
        _sim_mean = float(np.mean(_similar)) if len(_similar) > 0 else 0.0
        _opp_mean = float(np.mean(_opposite)) if len(_opposite) > 0 else 0.0
        _ratio = round(_opp_mean / _sim_mean, 2) if _sim_mean > 0 else None
        _summary_rows.append(
            {
                "method": _method,
                "similar_mean_p90": round(_sim_mean, 4),
                "opposite_mean_p90": round(_opp_mean, 4),
                "separation_ratio": _ratio,
            }
        )

    _summary_df = pl.DataFrame(_summary_rows).sort("separation_ratio", descending=True)

    _bar_rows = []
    for _row in _rows:
        for _method in _method_names:
            _bar_rows.append(
                {
                    "pair": _row["pair"],
                    "expected": _row["expected"],
                    "method": _method,
                    "distance_p90": _row[_method],
                }
            )

    _bar_fig = px.bar(
        pl.DataFrame(_bar_rows),
        x="distance_p90",
        y="pair",
        color="method",
        facet_row="expected",
        barmode="group",
        orientation="h",
        title="P90-normalized distances (comparable across methods)",
        labels={"distance_p90": "Distance / P90", "pair": ""},
    )
    _bar_fig.update_layout(
        height=max(400, len(_rows) * 40),
        template="plotly_white",
    )
    _bar_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    mo.vstack(
        [
            mo.md("### Separation ratio by method"),
            mo.md("Higher = better discrimination. All values are P90-normalized."),
            _summary_df,
            mo.md("### Per-pair P90-normalized distances"),
            _bar_fig,
            mo.md("### Full table"),
            _results_df,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interactive comparison: pick codes and compare

    Select diagnosis codes and see their time series side by side, plus the
    pairwise distance (P90-normalized) that each method assigns to every pair.
    """)
    return


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
    p90_per_method,
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
                _raw = float(_matrix[_idx_a, _idx_b])
                _p90 = p90_per_method[_name]
                _row[_name] = round(_raw / _p90, 4) if _p90 > 0 else 0.0
            _dist_rows.append(_row)

    _dist_df = pl.DataFrame(_dist_rows)

    _bar_rows = []
    for _row in _dist_rows:
        for _method in distance_matrices:
            _bar_rows.append(
                {
                    "pair": _row["pair"],
                    "method": _method,
                    "distance_p90": _row[_method],
                }
            )

    _bar_fig = px.bar(
        pl.DataFrame(_bar_rows),
        x="distance_p90",
        y="pair",
        color="method",
        barmode="group",
        orientation="h",
        title="P90-normalized distance by method",
        labels={"distance_p90": "Distance / P90", "pair": "", "method": "Method"},
    )
    _bar_fig.update_layout(height=max(300, len(_dist_rows) * 80), template="plotly_white")

    mo.vstack(
        [
            _ts_fig,
            mo.md("### Pairwise distances (P90-normalized)"),
            _dist_df,
            _bar_fig,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
