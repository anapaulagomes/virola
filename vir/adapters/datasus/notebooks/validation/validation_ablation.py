import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    from dotenv import load_dotenv
    from loguru import logger
    from sklearn import cluster
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
    )
    from sklearn.metrics.pairwise import cosine_similarity
    from snf import compute, metrics

    from vir.adapters.datasus.features import CITY_NAMES
    from vir.config import (
        AFFINITY_ZSCORE_PERMS,
        AFFINITY_ZSCORE_SEED,
        PROCESSED_DATA_DIR,
        VALIDATION_DIR,
    )
    from vir.db import fetch_cluster_assignments, fetch_runs
    from vir.helpers import slugify
    from vir.notebook_widgets import (
        distance_to_affinity,
        load_and_prepare_dataframe,
    )
    from vir.views import ClinicalView, DemographicView, TemporalView

    return (
        CITY_NAMES,
        ClinicalView,
        DemographicView,
        Path,
        PROCESSED_DATA_DIR,
        TemporalView,
        VALIDATION_DIR,
        AFFINITY_ZSCORE_PERMS,
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
        logger,
        metrics,
        mo,
        normalized_mutual_info_score,
        np,
        pl,
        slugify,
        AFFINITY_ZSCORE_SEED,
    )


@app.cell
def _():
    from vir.adapters.datasus import DatasusAdapter

    return (DatasusAdapter,)


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
def _(
    CITY_NAMES,
    Path,
    VALIDATION_DIR,
    AFFINITY_ZSCORE_PERMS,
    logger,
    mo,
    runs_by_city,
    slugify,
):
    if mo.app_meta().mode == "script":
        _cli_args = mo.cli_args()
        _city_arg = _cli_args.get("city", "")
        _slug_to_canonical = {slugify(key): key for key in runs_by_city}
        _matched_city = _slug_to_canonical.get(slugify(_city_arg)) if _city_arg else None
        if _city_arg and _matched_city is None:
            logger.warning(
                f"--city {_city_arg!r} did not match any saved run; running all cities. "
                f"Available: {sorted(runs_by_city)}"
            )
        _selected_cities = (
            [_matched_city]
            if _matched_city
            else sorted(k for k in runs_by_city if k in CITY_NAMES)
        )
        n_permutations = int(_cli_args.get("permutations", AFFINITY_ZSCORE_PERMS))
        output_dir = Path(_cli_args["output"]) if _cli_args.get("output") else VALIDATION_DIR
    else:
        n_permutations = AFFINITY_ZSCORE_PERMS
        _selected_cities = sorted(k for k in runs_by_city if k in CITY_NAMES)
        output_dir = VALIDATION_DIR

    city_selector = mo.ui.multiselect(
        options=_selected_cities,
        value=_selected_cities,
        label="Cities to include in the ablation",
    )
    city_selector
    return city_selector, n_permutations, output_dir


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
def _(
    ClinicalView,
    DatasusAdapter,
    DemographicView,
    TemporalView,
    adjusted_rand_score,
    cluster,
    compute,
    distance_to_affinity,
    fetch_cluster_assignments,
    load_and_prepare_dataframe,
    logger,
    mo,
    np,
    selected_runs,
):
    _scan_run = selected_runs[0]
    _scan_assignments = fetch_cluster_assignments(_scan_run["run_id"]).sort("terminology_id")
    _scan_stored_labels = _scan_assignments["cluster_id"].to_numpy()
    _scan_common_ids = _scan_assignments["terminology_id"].to_list()

    logger.info(f"=== Assumption scan: {_scan_run['city']} ===")
    logger.info(
        f"[1] n_clusters stored vs unique labels: "
        f"{_scan_run['n_clusters']} vs {len(set(_scan_stored_labels))}"
    )
    logger.info(f"[2] SNF K used in ablation: {_scan_run['snf_k']}")

    _scan_df = load_and_prepare_dataframe(
        file_path=_scan_run["file_path"],
        year_min=_scan_run["year_min"],
        year_max=_scan_run["year_max"],
    )
    _scan_clinical = ClinicalView(
        df=_scan_df,
        dataset_name=_scan_run["dataset_name"],
        params={"model": DatasusAdapter.default_model},
    )
    _scan_demographic = DemographicView(df=_scan_df, dataset_name=_scan_run["dataset_name"])
    _scan_temporal = TemporalView(df=_scan_df, dataset_name=_scan_run["dataset_name"])

    _scan_affinities = [
        distance_to_affinity(_scan_clinical.align_to_common_ids(_scan_common_ids)),
        distance_to_affinity(_scan_demographic.align_to_common_ids(_scan_common_ids)),
        distance_to_affinity(_scan_temporal.align_to_common_ids(_scan_common_ids)),
    ]
    _scan_fused = compute.snf(_scan_affinities, K=_scan_run["snf_k"])
    _scan_fused = np.nan_to_num(_scan_fused, nan=0.0, posinf=1.0, neginf=0.0)
    _scan_rerun_labels = cluster.spectral_clustering(
        _scan_fused, n_clusters=_scan_run["n_clusters"], random_state=42
    )
    _scan_ari = adjusted_rand_score(_scan_stored_labels, _scan_rerun_labels)
    logger.info(
        f"[3] ARI Full model re-run vs stored labels: {_scan_ari:.4f} "
        f"(measures pipeline determinism; ablation uses re-run as reference, not stored)"
    )
    logger.info(
        "[4] Cache source: matrices loaded from disk, not recomputed, "
        "if you see 'Loading from cache' above"
    )
    mo.md(
        f"**Assumption scan ({_scan_run['city']}):** "
        f"n_clusters {_scan_run['n_clusters']} = {len(set(_scan_stored_labels))} unique labels · "
        f"SNF K={_scan_run['snf_k']} · "
        f"ARI full re-run vs stored = **{_scan_ari:.4f}** (should be ≈ 1.0)"
    )
    return


@app.cell
def _(
    ClinicalView,
    DatasusAdapter,
    DemographicView,
    TemporalView,
    adjusted_rand_score,
    cluster,
    compute,
    cosine_similarity,
    distance_to_affinity,
    fetch_cluster_assignments,
    load_and_prepare_dataframe,
    logger,
    metrics,
    mo,
    n_permutations,
    normalized_mutual_info_score,
    np,
    pl,
    selected_runs,
    AFFINITY_ZSCORE_SEED,
):
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

    assert _CONDITIONS[0]["name"] == "Full model", "Full model must be first in _CONDITIONS"

    def _build_view_affinities(run, dataframe, common_ids):
        clinical = ClinicalView(
            df=dataframe,
            dataset_name=run["dataset_name"],
            params={"model": DatasusAdapter.default_model},
        )
        demographic = DemographicView(df=dataframe, dataset_name=run["dataset_name"])
        temporal = TemporalView(df=dataframe, dataset_name=run["dataset_name"])
        return {
            "clinical": distance_to_affinity(clinical.align_to_common_ids(common_ids)),
            "demographic": distance_to_affinity(demographic.align_to_common_ids(common_ids)),
            "temporal": distance_to_affinity(temporal.align_to_common_ids(common_ids)),
        }

    def _run_condition(condition, view_affinities, n_clusters, snf_k):
        if condition["naive"]:
            concatenated = np.hstack([view_affinities[view] for view in condition["views"]])
            fused = cosine_similarity(concatenated)
            np.fill_diagonal(fused, 0.0)
        elif len(condition["views"]) == 1:
            fused = view_affinities[condition["views"][0]].copy()
        else:
            affinities = [view_affinities[view] for view in condition["views"]]
            fused = compute.snf(affinities, K=snf_k)
            fused = np.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=0.0)
        labels = cluster.spectral_clustering(fused, n_clusters=n_clusters, random_state=42)
        return fused, labels

    def _run_ablation_for_city(run, conditions, permutation_count):
        dataframe = load_and_prepare_dataframe(
            file_path=run["file_path"],
            year_min=run["year_min"],
            year_max=run["year_max"],
        )
        assignments = fetch_cluster_assignments(run["run_id"]).sort("terminology_id")
        common_ids = assignments["terminology_id"].to_list()
        view_affinities = _build_view_affinities(run, dataframe, common_ids)

        rows = []
        full_model_labels = None
        for condition in conditions:
            fused_matrix, labels = _run_condition(
                condition, view_affinities, run["n_clusters"], run["snf_k"]
            )
            if condition["name"] == "Full model":
                full_model_labels = labels
            reference_labels = full_model_labels if full_model_labels is not None else labels
            rows.append(
                {
                    "city": run["city"],
                    "run_id": run["run_id"],
                    "n_codes": len(common_ids),
                    "condition": condition["name"],
                    "group": condition["group"],
                    "ari": float(adjusted_rand_score(reference_labels, labels)),
                    "nmi": float(normalized_mutual_info_score(reference_labels, labels)),
                    "affinity_zscore": float(
                        metrics.affinity_zscore(
                            fused_matrix,
                            labels,
                            n_perms=permutation_count,
                            seed=AFFINITY_ZSCORE_SEED,
                        )
                    ),
                    "silhouette": float(metrics.silhouette_score(fused_matrix, labels)),
                }
            )
        return rows

    _result_rows = []
    for _run in selected_runs:
        logger.info(f"Running ablation for {_run['city']}...")
        _result_rows.extend(_run_ablation_for_city(_run, _CONDITIONS, n_permutations))

    results_long_df = pl.DataFrame(_result_rows)

    mo.md(
        f"Ablation complete — {len(_CONDITIONS)} conditions × "
        f"{len(selected_runs)} cities = **{len(_result_rows)} clusterings** evaluated."
    )
    return (results_long_df,)


@app.cell
def _(logger, mo, output_dir, results_long_df, slugify):
    output_dir.mkdir(parents=True, exist_ok=True)

    if mo.app_meta().mode == "script":
        for _city, _city_df in results_long_df.group_by("city"):
            _slug = slugify(_city[0])
            _path = output_dir / f"ablation_{_slug}.csv"
            _city_df.write_csv(_path)
            logger.info(f"Saved {_path}")
    else:
        _path = output_dir / "ablation_results.csv"
        results_long_df.write_csv(_path)
        mo.md(f"Results saved to `{_path}`")
    return


@app.cell
def _(mo):
    mo.stop(mo.app_meta().mode == "script")
    return


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
      distribution over cluster labels (a configurable number of permutations, default: 1000).
      This is the canonical quality metric for SNF clusterings — the same metric stored in the `runs` table
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
def _(go, mo, n_permutations, results_summary_df):
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
            f"Affinity Z-score ({n_permutations:,} permutations)",
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
