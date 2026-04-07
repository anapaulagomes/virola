import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    from dotenv import load_dotenv
    from sklearn import cluster
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
    )
    from sklearn.metrics.pairwise import cosine_similarity
    from snf import compute, metrics

    from vir.config import PROCESSED_DATA_DIR
    from vir.db import fetch_cluster_assignments, fetch_runs
    from vir.notebook_widgets import (
        distance_to_affinity,
        load_and_prepare_dataframe,
    )
    from vir.views import ClinicalView, DemographicView, TemporalView

    return (
        ClinicalView,
        DemographicView,
        PROCESSED_DATA_DIR,
        TemporalView,
        adjusted_rand_score,
        cluster,
        compute,
        cosine_similarity,
        distance_to_affinity,
        fetch_cluster_assignments,
        fetch_runs,
        go,
        load_and_prepare_dataframe,
        load_dotenv,
        metrics,
        mo,
        normalized_mutual_info_score,
        np,
        pl,
    )


@app.cell
def _(load_dotenv):
    _ = load_dotenv()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Ablation Study — Multi-city

    An **ablation study** systematically removes components of a model to measure each
    component's individual contribution. In multi-view learning, this answers the question:
    does fusing all three views genuinely outperform any single view or two-view combination?
    Without this evidence, the use of SNF over a simpler single-view baseline cannot be
    scientifically justified.

    This notebook runs the eight ablation conditions independently on a user-selected
    set of cities (1 to N). Each city is treated as a replication of the experiment;
    aggregated results report the mean ± standard deviation of every metric across the
    selected cities. This is the form expected by the paper's Section 6.4 results table.

    ## Eight conditions

    | # | Condition | Views fused | Purpose |
    |---|-----------|-------------|---------|
    | 1 | Full model | Clinical + Demographic + Temporal | Reference — the proposed method |
    | 2 | Drop Clinical | Demographic + Temporal | Measure clinical view's unique contribution |
    | 3 | Drop Demographic | Clinical + Temporal | Measure demographic view's unique contribution |
    | 4 | Drop Temporal | Clinical + Demographic | Measure temporal view's unique contribution |
    | 5 | Clinical only | Clinical | Semantic similarity as sole signal |
    | 6 | Demographic only | Demographic | Sex/age distribution as sole signal |
    | 7 | Temporal only | Temporal | Epidemic curves as sole signal |
    | 8 | Naive concatenation | All three (no SNF diffusion) | Does SNF add value beyond concatenation? |

    All conditions for a given city use the **same distance matrices** and the same
    $n_\text{clusters}$ as that city's reference run. Only the fusion step changes.
    """)
    return


@app.cell
def _(fetch_runs, mo):
    _runs = fetch_runs()
    mo.stop(
        _runs.is_empty(),
        mo.callout(mo.md("No saved runs found. Run the SNF pipeline first."), kind="warn"),
    )

    runs_latest_per_city = _runs.unique(subset=["city"], keep="first")
    runs_by_city = {
        row["city"]: row for row in runs_latest_per_city.iter_rows(named=True) if row["city"]
    }
    return (runs_by_city,)


@app.cell
def _(mo, runs_by_city):
    city_selector = mo.ui.multiselect(
        options=sorted(runs_by_city.keys()),
        value=sorted(runs_by_city.keys()),
        label="Cities to include in the ablation (defaults to all)",
    )
    city_selector
    return (city_selector,)


@app.cell
def _(PROCESSED_DATA_DIR, city_selector, mo, runs_by_city):
    mo.stop(
        not city_selector.value,
        mo.callout(mo.md("Select at least one city above."), kind="warn"),
    )

    _all_files = [str(file.resolve()) for file in PROCESSED_DATA_DIR.glob("**/prepared_*.parquet")]

    def _resolve_file_for_dataset(dataset_name):
        _matches = [file for file in _all_files if dataset_name in file]
        return _matches[0] if _matches else None

    selected_runs = []
    missing_files = []
    for _city in city_selector.value:
        _run = runs_by_city[_city]
        _file = _resolve_file_for_dataset(_run["dataset_name"])
        if _file is None:
            missing_files.append((_city, _run["dataset_name"]))
            continue
        selected_runs.append({**_run, "file_path": _file})

    mo.stop(
        not selected_runs,
        mo.callout(
            mo.md(
                "No prepared parquet files matched any selected city.\n\n"
                f"Missing: `{missing_files}`"
            ),
            kind="danger",
        ),
    )

    _missing_note = f"\n\n_Missing files for: `{missing_files}`_" if missing_files else ""
    mo.md(
        f"**Selected {len(selected_runs)} cities for ablation:** "
        + ", ".join(f"`{run['city']}`" for run in selected_runs)
        + _missing_note
    )
    return (selected_runs,)


@app.cell
def _(mo):
    mo.md(r"""
    ## View Matrices

    For each selected city, the same three distance matrices — one per view — are
    aligned to that city's reference run code list. Distance matrices are normalised
    to $[0, 1]$ before conversion to affinity using SNF's adaptive Gaussian kernel
    with $\mu = 0.5$ and $K = 20$. Each affinity matrix is normalised to $[0, 1]$
    so all views enter SNF on equal footing.

    ## Running the Eight Conditions

    For conditions 1–7, the selected affinity matrices are passed to SNF's iterative
    non-linear diffusion; spectral clustering on the fused matrix yields the final
    partition. For condition 8 (naive concatenation), affinity matrix rows are
    concatenated horizontally and pairwise cosine similarity replaces the fused
    matrix — no iterative diffusion is applied. All spectral clustering calls use
    `random_state=42` for reproducibility.

    The cell below loops over the selected cities, computes the eight conditions
    for each, and collects a long-format result row per (city, condition) pair.
    """)
    return


@app.cell
def _(
    ClinicalView,
    DemographicView,
    TemporalView,
    adjusted_rand_score,
    cluster,
    compute,
    cosine_similarity,
    distance_to_affinity,
    fetch_cluster_assignments,
    load_and_prepare_dataframe,
    metrics,
    mo,
    normalized_mutual_info_score,
    np,
    pl,
    selected_runs,
):
    AFFINITY_ZSCORE_PERMUTATIONS = 200

    _CONDITIONS = [
        {
            "name": "Full model",
            "group": "full",
            "views": ["clinical", "demographic", "temporal"],
            "naive": False,
        },
        {
            "name": "Drop Clinical",
            "group": "drop_one",
            "views": ["demographic", "temporal"],
            "naive": False,
        },
        {
            "name": "Drop Demographic",
            "group": "drop_one",
            "views": ["clinical", "temporal"],
            "naive": False,
        },
        {
            "name": "Drop Temporal",
            "group": "drop_one",
            "views": ["clinical", "demographic"],
            "naive": False,
        },
        {
            "name": "Clinical only",
            "group": "single",
            "views": ["clinical"],
            "naive": False,
        },
        {
            "name": "Demographic only",
            "group": "single",
            "views": ["demographic"],
            "naive": False,
        },
        {
            "name": "Temporal only",
            "group": "single",
            "views": ["temporal"],
            "naive": False,
        },
        {
            "name": "Naive concat",
            "group": "baseline",
            "views": ["clinical", "demographic", "temporal"],
            "naive": True,
        },
    ]

    def _run_condition(condition, view_affinities, n_clusters, snf_k):
        if condition["naive"]:
            _concatenated = np.hstack([view_affinities[view] for view in condition["views"]])
            _fused = cosine_similarity(_concatenated)
            np.fill_diagonal(_fused, 0.0)
        elif len(condition["views"]) == 1:
            _fused = view_affinities[condition["views"][0]].copy()
        else:
            _affinities = [view_affinities[view] for view in condition["views"]]
            _fused = compute.snf(_affinities, K=snf_k)
            _fused = np.nan_to_num(_fused, nan=0.0, posinf=1.0, neginf=0.0)
        _labels = cluster.spectral_clustering(_fused, n_clusters=n_clusters, random_state=42)
        return _fused, _labels

    _result_rows = []
    for _run in selected_runs:
        _city = _run["city"]
        _dataset_name = _run["dataset_name"]

        _dataframe = load_and_prepare_dataframe(
            file_path=_run["file_path"],
            year_min=_run["year_min"],
            year_max=_run["year_max"],
        )

        _assignments = fetch_cluster_assignments(_run["run_id"]).sort("terminology_id")
        _common_ids = _assignments["terminology_id"].to_list()
        _reference_labels = _assignments["cluster_id"].to_numpy()

        _clinical_view = ClinicalView(df=_dataframe, dataset_name=_dataset_name)
        _demographic_view = DemographicView(df=_dataframe, dataset_name=_dataset_name)
        _temporal_view = TemporalView(df=_dataframe, dataset_name=_dataset_name)

        _view_affinities = {
            "clinical": distance_to_affinity(_clinical_view.align_to_common_ids(_common_ids)),
            "demographic": distance_to_affinity(
                _demographic_view.align_to_common_ids(_common_ids)
            ),
            "temporal": distance_to_affinity(_temporal_view.align_to_common_ids(_common_ids)),
        }

        for _condition in _CONDITIONS:
            _fused_matrix, _labels = _run_condition(
                _condition,
                _view_affinities,
                n_clusters=_run["n_clusters"],
                snf_k=_run["snf_k"],
            )
            _result_rows.append(
                {
                    "city": _city,
                    "run_id": _run["run_id"],
                    "n_codes": len(_common_ids),
                    "condition": _condition["name"],
                    "group": _condition["group"],
                    "ari": round(adjusted_rand_score(_reference_labels, _labels), 4),
                    "nmi": round(normalized_mutual_info_score(_reference_labels, _labels), 4),
                    "affinity_zscore": round(
                        float(
                            metrics.affinity_zscore(
                                _fused_matrix,
                                _labels,
                                n_perms=AFFINITY_ZSCORE_PERMUTATIONS,
                            )
                        ),
                        4,
                    ),
                    "silhouette": round(
                        float(metrics.silhouette_score(_fused_matrix, _labels)), 4
                    ),
                }
            )

    results_long_df = pl.DataFrame(_result_rows)

    mo.md(
        f"Ablation complete — {len(_CONDITIONS)} conditions × "
        f"{len(selected_runs)} cities = **{len(_result_rows)} clusterings** evaluated."
    )
    return (results_long_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Aggregated results across cities

    The summary table reports, for each ablation condition, the mean and standard
    deviation across the selected cities for every metric. This is the form used
    to populate the paper's ablation table — single-row-per-condition with
    city-level variability collapsed into the standard deviation.

    ### How to read the metrics

    - **ARI** (Adjusted Rand Index, $[-1, 1]$): agreement between a condition's partition
      and the city's full-model partition. ARI = 1 means identical; ARI ≈ 0 means random.
      The _Full model_ row should show ARI ≈ 1 in every city (sanity check).
    - **NMI** (Normalised Mutual Information, $[0, 1]$): shared information between two
      partitions. Complements ARI for distributions with very uneven cluster sizes.
    - **Affinity Z-score**: SNF-native internal validity (Wang et al. 2014). Compares
      observed intra-cluster affinity in the fused matrix against a permutation null
      distribution over cluster labels (200 permutations). This is the canonical
      quality metric for SNF clusterings — the same metric stored in the `runs` table
      for the reference runs. Higher is better; values around 0 indicate clusters no
      better than random label assignment.
    - **Silhouette** ($[-1, 1]$): kept as a secondary sanity check using `snfpy`'s
      affinity-aware implementation. Higher is better.

    ### Key patterns to look for

    1. **Full model dominates the affinity Z-score** across cities → fusing all three
       views yields more cohesive clusters in the fused-affinity geometry than any
       subset.
    2. **The drop-one condition with the largest mean ARI/NMI drop** reveals the view
       that contributes most uniquely to the partition.
    3. **Ordering of single-view conditions** on affinity Z-score → which view carries
       the most intrinsic clustering signal when used alone.
    4. **Full model > Naive concat** on all metrics → SNF's non-linear diffusion adds
       value beyond simple feature concatenation.
    5. **Low standard deviations** across cities → the ablation conclusions generalise
       beyond a single dataset.
    """)
    return


@app.cell
def _(pl, results_long_df):
    results_summary_df = (
        results_long_df.group_by(["condition", "group"])
        .agg(
            [
                pl.col("ari").mean().round(4).alias("ari_mean"),
                pl.col("ari").std().round(4).alias("ari_std"),
                pl.col("nmi").mean().round(4).alias("nmi_mean"),
                pl.col("nmi").std().round(4).alias("nmi_std"),
                pl.col("affinity_zscore").mean().round(4).alias("affinity_zscore_mean"),
                pl.col("affinity_zscore").std().round(4).alias("affinity_zscore_std"),
                pl.col("silhouette").mean().round(4).alias("silhouette_mean"),
                pl.col("silhouette").std().round(4).alias("silhouette_std"),
                pl.len().alias("n_cities"),
            ]
        )
        .with_columns(
            [
                pl.col("ari_std").fill_null(0.0),
                pl.col("nmi_std").fill_null(0.0),
                pl.col("affinity_zscore_std").fill_null(0.0),
                pl.col("silhouette_std").fill_null(0.0),
            ]
        )
    )
    return (results_summary_df,)


@app.cell
def _(mo, results_summary_df):
    mo.ui.table(results_summary_df)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Per-city detail

    Long-format table with one row per (city, condition). Useful for inspecting
    outliers and confirming that aggregated values are not driven by a single city.
    """)
    return


@app.cell
def _(mo, results_long_df):
    mo.ui.table(results_long_df)
    return


@app.cell
def _(go, mo, results_summary_df):
    _CONDITION_ORDER = [
        "Full model",
        "Drop Clinical",
        "Drop Demographic",
        "Drop Temporal",
        "Clinical only",
        "Demographic only",
        "Temporal only",
        "Naive concat",
    ]
    _GROUP_COLORS = {
        "full": "#2E86AB",
        "drop_one": "#A23B72",
        "single": "#F18F01",
        "baseline": "#C73E1D",
    }
    _metric_specs = [
        ("ari", "ARI vs. Full Model (mean ± std across cities)", "Adjusted Rand Index"),
        ("nmi", "NMI vs. Full Model (mean ± std across cities)", "Normalised Mutual Information"),
        (
            "affinity_zscore",
            "Affinity Z-score (mean ± std across cities)",
            "Affinity Z-score (200 permutations)",
        ),
        ("silhouette", "Silhouette (mean ± std across cities)", "Silhouette (sanity check)"),
    ]

    _rows_by_condition = {row["condition"]: row for row in results_summary_df.to_dicts()}
    _ordered_rows = [
        _rows_by_condition[name] for name in _CONDITION_ORDER if name in _rows_by_condition
    ]
    _ordered_reversed = list(reversed(_ordered_rows))
    _conditions = [row["condition"] for row in _ordered_reversed]
    _colors = [_GROUP_COLORS[row["group"]] for row in _ordered_reversed]

    _full_row = _rows_by_condition.get("Full model")

    _figs = []
    for _column, _title, _xaxis_title in _metric_specs:
        _means = [row[f"{_column}_mean"] for row in _ordered_reversed]
        _stds = [row[f"{_column}_std"] for row in _ordered_reversed]
        _x_max = max(mean + std for mean, std in zip(_means, _stds))
        _x_min = min(mean - std for mean, std in zip(_means, _stds))
        _padding = (_x_max - _x_min) * 0.25 + 0.01
        _figure = go.Figure(
            go.Bar(
                y=_conditions,
                x=_means,
                error_x=dict(type="data", array=_stds, visible=True),
                orientation="h",
                marker_color=_colors,
                text=[f"{mean:.3f} ± {std:.3f}" for mean, std in zip(_means, _stds)],
                textposition="outside",
            )
        )
        if _column in ("ari", "nmi") and _full_row is not None:
            _figure.add_vline(
                x=_full_row[f"{_column}_mean"],
                line_dash="dot",
                line_color="grey",
                annotation_text="Full model",
                annotation_position="top right",
            )
        _figure.update_layout(
            title=_title,
            xaxis_title=_xaxis_title,
            xaxis=dict(
                range=[min(_x_min, 0) - 0.02, _x_max + _padding],
                gridcolor="#eeeeee",
            ),
            height=320,
            margin=dict(t=50, b=50, l=160, r=160),
            showlegend=False,
            plot_bgcolor="white",
        )
        _figs.append(_figure)

    mo.vstack([mo.ui.plotly(figure) for figure in _figs])
    return


if __name__ == "__main__":
    app.run()
