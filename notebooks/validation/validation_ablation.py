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
        calinski_harabasz_score,
        normalized_mutual_info_score,
    )
    from sklearn.metrics.pairwise import cosine_similarity
    from snf import compute, metrics

    from vir.db import fetch_cluster_assignments, fetch_runs
    from vir.helpers import extract_dataset_name
    from vir.notebook_widgets import (
        create_file_dropdown,
        distance_to_affinity,
        load_and_prepare_dataframe,
    )
    from vir.views import ClinicalView, DemographicView, TemporalView

    return (
        ClinicalView,
        DemographicView,
        TemporalView,
        adjusted_rand_score,
        calinski_harabasz_score,
        cluster,
        compute,
        cosine_similarity,
        create_file_dropdown,
        distance_to_affinity,
        extract_dataset_name,
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
    # Ablation Study

    An **ablation study** systematically removes components of a model to measure each
    component's individual contribution. In multi-view learning, this answers the question:
    does fusing all three views genuinely outperform any single view or two-view combination?
    Without this evidence, the use of SNF over a simpler single-view baseline cannot be
    scientifically justified.

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

    All conditions use the **same distance matrices** and the same $n_\text{clusters}$ as the
    selected reference run. Only the fusion step changes.
    """)
    return


@app.cell
def _(fetch_runs, mo):
    _runs = fetch_runs()
    mo.stop(
        _runs.is_empty(),
        mo.callout(mo.md("No saved runs found. Run the SNF pipeline first."), kind="warn"),
    )
    _opts = {
        f"[{r['run_id']}] {r['dataset_name']} {r['year_min']}–{r['year_max']} ({r['n_clusters']} clusters)": r[
            "run_id"
        ]
        for r in _runs.iter_rows(named=True)
    }
    run_dropdown = mo.ui.dropdown(
        options=_opts, label="Select reference SNF run", value=list(_opts.keys())[0]
    )
    run_dropdown
    return (run_dropdown,)


@app.cell
def _(fetch_cluster_assignments, fetch_runs, mo, pl, run_dropdown):
    mo.stop(
        run_dropdown.value is None,
        mo.callout(mo.md("Select a run above."), kind="warn"),
    )

    _run_id = run_dropdown.value
    run_meta = fetch_runs().filter(pl.col("run_id") == _run_id).row(0, named=True)

    # Sort by terminology_id for a stable, reproducible ordering that all
    # view alignment calls will match.
    _assignments = fetch_cluster_assignments(_run_id).sort("terminology_id")
    common_ids = _assignments["terminology_id"].to_list()
    reference_labels = _assignments["cluster_id"].to_numpy()

    mo.md(f"""
    **Reference run**

    | Parameter | Value |
    |-----------|-------|
    | Dataset | `{run_meta["dataset_name"]}` |
    | Period | {run_meta["year_min"]}–{run_meta["year_max"]} |
    | Clusters ($K$) | {run_meta["n_clusters"]} |
    | SNF $K$ | {run_meta["snf_k"]} |
    | Codes | {len(common_ids)} |
    | Silhouette (saved) | {run_meta["silhouette_score"]:.4f} |
    """)
    return common_ids, reference_labels, run_meta


@app.cell
def _(create_file_dropdown, run_meta):
    file_dropdown = create_file_dropdown(default_pattern=run_meta["dataset_name"])
    file_dropdown
    return (file_dropdown,)


@app.cell
def _(mo):
    mo.md(r"""
    ## View Matrices

    Each condition re-uses the **same three distance matrices** — one per view — aligned
    to the reference run's code list (`common_ids`). This guarantees that any difference
    in clustering quality between conditions is caused exclusively by the fusion strategy,
    not by differences in the underlying data.

    Distance matrices are normalised to $[0, 1]$ before conversion to affinity. Affinities
    are computed with an adaptive Gaussian kernel:

    $$W_{ij} = \exp\!\left(-\frac{d_{ij}^2}{2\,\sigma_i\,\sigma_j}\right), \quad
    \sigma_i = \mu \cdot \frac{1}{k}\sum_{l \in \mathcal{N}_k(i)} d_{il}$$

    where $\mu = 0.5$ and $k = 20$. Each affinity matrix is subsequently normalised to
    $[0, 1]$ so that all views enter SNF on equal footing.

    If view matrices are cached from a prior run they are loaded directly; otherwise
    they are computed from the selected data file.
    """)
    return


@app.cell
def _(
    ClinicalView,
    DemographicView,
    TemporalView,
    common_ids,
    extract_dataset_name,
    file_dropdown,
    load_and_prepare_dataframe,
    mo,
    run_meta,
):
    mo.stop(
        file_dropdown.value is None,
        mo.callout(mo.md("Select a data file above."), kind="warn"),
    )

    _dataset_name = extract_dataset_name(file_dropdown.value)

    _df = load_and_prepare_dataframe(
        file_path=file_dropdown.value,
        year_min=run_meta["year_min"],
        year_max=run_meta["year_max"],
    )

    _clinical_view = ClinicalView(df=_df, dataset_name=_dataset_name)
    _demographic_view = DemographicView(df=_df, dataset_name=_dataset_name)
    _temporal_view = TemporalView(df=_df, dataset_name=_dataset_name)

    clinical_dist = _clinical_view.align_to_common_ids(common_ids)
    demographic_dist = _demographic_view.align_to_common_ids(common_ids)
    temporal_dist = _temporal_view.align_to_common_ids(common_ids)
    return clinical_dist, demographic_dist, temporal_dist


@app.cell
def _(clinical_dist, demographic_dist, distance_to_affinity, temporal_dist):
    W_clin = distance_to_affinity(clinical_dist)
    W_demo = distance_to_affinity(demographic_dist)
    W_temp = distance_to_affinity(temporal_dist)
    return W_clin, W_demo, W_temp


@app.cell
def _(mo):
    mo.md(r"""
    ## Running the Eight Conditions

    For conditions 1–7, the selected affinity matrices are passed to SNF's iterative
    non-linear diffusion:

    $$\mathbf{P}^{(v)} \leftarrow \mathbf{S}^{(v)} \cdot
    \left(\sum_{u \neq v} \mathbf{P}^{(u)}\right) / (m - 1) \cdot (\mathbf{S}^{(v)})^\top$$

    where $\mathbf{S}^{(v)}$ is the normalised sparse affinity for view $v$ and $m$ is the
    number of views. After $t$ iterations, the fused matrix is the average of all
    $\mathbf{P}^{(v)}$. Spectral clustering on the fused matrix yields the final partition.

    For condition 8 (naive concatenation), affinity matrix rows are concatenated horizontally
    — forming a feature vector of length $3n$ per code — and pairwise cosine similarity
    replaces the fused matrix. No iterative diffusion is applied. This isolates the value
    of SNF's non-linear cross-view reinforcement.

    All spectral clustering calls use `random_state=42` for reproducibility.
    """)
    return


@app.cell
def _(
    W_clin,
    W_demo,
    W_temp,
    cluster,
    compute,
    cosine_similarity,
    np,
    run_meta,
):
    _snf_k = run_meta["snf_k"]
    _n_clusters = run_meta["n_clusters"]
    _view_affinities = {"clinical": W_clin, "demographic": W_demo, "temporal": W_temp}

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

    def _run_condition(cond):
        if cond["naive"]:
            _concat = np.hstack([_view_affinities[v] for v in cond["views"]])
            _fused = cosine_similarity(_concat)
            np.fill_diagonal(_fused, 0.0)
        elif len(cond["views"]) == 1:
            # Single view: affinity matrix is used directly as the fused matrix
            _fused = _view_affinities[cond["views"][0]].copy()
        else:
            _affinities = [_view_affinities[v] for v in cond["views"]]
            _fused = compute.snf(_affinities, K=_snf_k)
            _fused = np.nan_to_num(_fused, nan=0.0, posinf=1.0, neginf=0.0)
        _labels = cluster.spectral_clustering(_fused, n_clusters=_n_clusters, random_state=42)
        return {"labels": _labels, "fused": _fused, **cond}

    ablation_results = [_run_condition(c) for c in _CONDITIONS]
    return (ablation_results,)


@app.cell
def _(
    ablation_results,
    adjusted_rand_score,
    calinski_harabasz_score,
    metrics,
    mo,
    normalized_mutual_info_score,
    pl,
    reference_labels,
):
    _rows = [
        {
            "condition": r["name"],
            "group": r["group"],
            "ari": round(adjusted_rand_score(reference_labels, r["labels"]), 4),
            "nmi": round(normalized_mutual_info_score(reference_labels, r["labels"]), 4),
            "silhouette": round(float(metrics.silhouette_score(r["fused"], r["labels"])), 4),
            "calinski_harabasz": round(calinski_harabasz_score(r["fused"], r["labels"]), 2),
        }
        for r in ablation_results
    ]

    results_df = pl.DataFrame(_rows)

    mo.md(
        f"Ablation complete — {len(_rows)} conditions evaluated over {len(reference_labels)} codes."
    )
    return (results_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Results

    ### How to read the metrics

    - **ARI** (Adjusted Rand Index, $[-1, 1]$): agreement between a condition's partition
      and the full model. ARI = 1 means identical partitions; ARI ≈ 0 means random
      agreement. The _Full model_ row should show ARI ≈ 1 (sanity check; minor variation
      is possible due to spectral clustering initialisation).

    - **NMI** (Normalised Mutual Information, $[0, 1]$): shared information between two
      partitions, normalised by entropy. Complements ARI for distributions with large
      variation in cluster sizes.

    - **Silhouette** ($[-1, 1]$): intra-cluster cohesion vs. inter-cluster separation,
      computed on the fused affinity matrix (high affinity = similar). Higher is better.

    - **Calinski-Harabasz** ($[0, +\infty)$): ratio of between-cluster to within-cluster
      dispersion over the fused matrix rows. Higher is better; sensitive to cluster count.

    ### Key patterns to look for

    1. **Full model dominates** silhouette and Calinski-Harabasz → fusing all three views
       produces tighter, better-separated clusters than any subset.
    2. **The drop-one condition with the largest ARI/NMI drop** reveals the view that
       contributes most uniquely to the partition — removing it causes the greatest
       divergence from the full model.
    3. **Ordering of single-view conditions** on silhouette → which view carries the most
       intrinsic clustering signal when used alone.
    4. **Full model > Naive concat** on all metrics → SNF's non-linear iterative diffusion
       adds value beyond simple feature concatenation.
    """)
    return


@app.cell
def _(mo, results_df):
    mo.ui.table(results_df)
    return


@app.cell
def _(go, mo, results_df):
    _GROUP_COLORS = {
        "full": "#2E86AB",
        "drop_one": "#A23B72",
        "single": "#F18F01",
        "baseline": "#C73E1D",
    }

    _metric_specs = [
        ("ari", "ARI vs. Full Model", "Adjusted Rand Index"),
        ("nmi", "NMI vs. Full Model", "Normalised Mutual Information"),
        ("silhouette", "Silhouette Score", "Silhouette"),
        ("calinski_harabasz", "Calinski-Harabasz", "Calinski-Harabasz Index"),
    ]

    _rows = results_df.to_dicts()
    # Reverse so Full model appears at the top of each horizontal chart
    _rows_rev = list(reversed(_rows))
    _conditions = [r["condition"] for r in _rows_rev]
    _colors = [_GROUP_COLORS[r["group"]] for r in _rows_rev]

    _full_row = next(r for r in _rows if r["condition"] == "Full model")

    _figs = []
    for _col, _title, _xaxis_title in _metric_specs:
        _values = [r[_col] for r in _rows_rev]
        _x_max = max(_values)
        _x_min = min(_values)
        _padding = (_x_max - _x_min) * 0.25 + 0.01
        _f = go.Figure(
            go.Bar(
                y=_conditions,
                x=_values,
                orientation="h",
                marker_color=_colors,
                text=[f"{v:.3f}" for v in _values],
                textposition="outside",
            )
        )
        if _col in ("ari", "nmi"):
            _f.add_vline(
                x=_full_row[_col],
                line_dash="dot",
                line_color="grey",
                annotation_text="Full model",
                annotation_position="top right",
            )
        _f.update_layout(
            title=_title,
            xaxis_title=_xaxis_title,
            xaxis=dict(range=[min(_x_min, 0) - 0.02, _x_max + _padding], gridcolor="#eeeeee"),
            height=320,
            margin=dict(t=50, b=50, l=160, r=80),
            showlegend=False,
            plot_bgcolor="white",
        )
        _figs.append(_f)

    mo.vstack([mo.ui.plotly(_f) for _f in _figs])
    return


if __name__ == "__main__":
    app.run()
