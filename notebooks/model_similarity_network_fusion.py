import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Similarity Network Fusion

    ## Dimensions 🎲

    * **Clinical**: semantic similarity
    * **Temporal**: temporal frequency
    * **Demographic**: sex and gender characteristics

    Calculate the similarity in all these dimensions, having the diagnosis as the central factor.
    """)
    return


@app.cell
def _():
    from datetime import datetime
    from pathlib import Path

    import igraph
    import leidenalg
    import marimo as mo
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    import umap
    from dotenv import load_dotenv
    from sklearn import cluster
    from snf import compute, metrics

    return (
        Path,
        cluster,
        compute,
        datetime,
        go,
        igraph,
        leidenalg,
        load_dotenv,
        metrics,
        mo,
        np,
        pl,
        px,
        umap,
    )


@app.cell
def _():
    from vir.config import (
        AFFINITY_ZSCORE_PERMS,
        AFFINITY_ZSCORE_SEED,
        CLUSTERING_DIR,
        get_indicative_label,
    )
    from vir.db import insert_run_leiden, insert_run_multi_k, update_run_finished_at
    from vir.helpers import extract_dataset_name
    from vir.notebook_widgets import (
        create_file_dropdown,
        load_and_prepare_dataframe,
        load_indicative_codes,
        load_terminology_labels,
        normalize_distance,
    )
    from vir.results import save_run
    from vir.views import ClinicalView, DemographicView, TemporalView

    return (
        CLUSTERING_DIR,
        ClinicalView,
        AFFINITY_ZSCORE_PERMS,
        AFFINITY_ZSCORE_SEED,
        DemographicView,
        TemporalView,
        create_file_dropdown,
        extract_dataset_name,
        get_indicative_label,
        insert_run_leiden,
        insert_run_multi_k,
        load_and_prepare_dataframe,
        load_indicative_codes,
        load_terminology_labels,
        normalize_distance,
        update_run_finished_at,
        save_run,
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
def _(CLUSTERING_DIR, Path, args, datetime, AFFINITY_ZSCORE_PERMS):
    year_min = args.get("year-min", 2022)
    year_max = args.get("year-max", 2024)
    snf_k = args.get("snf-k", 10)
    n_clusters = args.get("n-clusters", 200)
    output_dir = Path(args.get("output") or str(CLUSTERING_DIR))
    notes = args.get("notes") or None
    run_start = datetime.now().isoformat()
    n_permutations = int(args.get("permutations", AFFINITY_ZSCORE_PERMS))
    return n_clusters, notes, output_dir, run_start, snf_k, year_max, year_min, n_permutations


@app.cell
def _(mo):
    mo.md(r"""
    ## Data loading
    """)
    return


@app.cell
def _(load_terminology_labels):
    terminology_labels = load_terminology_labels()
    return (terminology_labels,)


@app.cell
def _(args, load_indicative_codes):
    exclude_codes, indicative_codes_enabled = load_indicative_codes(
        enabled="no-indicative" not in args
    )
    return exclude_codes, indicative_codes_enabled


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
def _(mo):
    mo.md(r"""
    ## Dimensions

    Each view computes a distance matrix, then converts to affinity using adaptive Gaussian kernel.
    """)
    return


@app.cell
def _(df, extract_dataset_name, file_path):
    view_params = {"df": df, "dataset_name": extract_dataset_name(file_path)}
    return (view_params,)


@app.cell
def _(ClinicalView, DemographicView, TemporalView, view_params):
    # Instantiate views
    clinical_view = ClinicalView(**view_params)
    temporal_view = TemporalView(**view_params)
    demographic_view = DemographicView(**view_params)

    # Get distance matrices (cached)
    clinical_distance = clinical_view.get_matrix()
    temporal_distance = temporal_view.get_matrix()
    demographic_distance = demographic_view.get_matrix()

    print(f"Clinical distance shape: {clinical_distance.shape}")
    print(f"Temporal distance shape: {temporal_distance.shape}")
    print(f"Demographic distance shape: {demographic_distance.shape}")
    return clinical_view, demographic_view, temporal_view


@app.cell
def _(mo):
    mo.md(r"""
    ### Align matrices to common diagnosis IDs

    Ensure all three views have the same diagnoses in the same order.
    """)
    return


@app.cell
def _(clinical_view, demographic_view, temporal_view):
    clinical_ids = set(clinical_view.get_terminology_ids())
    temporal_ids = set(temporal_view.get_terminology_ids())
    demographic_ids = set(demographic_view.get_terminology_ids())

    n_codes_clinical = len(clinical_ids)
    n_codes_temporal = len(temporal_ids)
    n_codes_demographic = len(demographic_ids)

    print(f"Clinical: {n_codes_clinical} diagnoses")
    print(f"Temporal: {n_codes_temporal} diagnoses")
    print(f"Demographic: {n_codes_demographic} diagnoses")

    common_ids = sorted(clinical_ids & temporal_ids & demographic_ids)
    print(f"\nCommon diagnoses: {len(common_ids)}")

    missing_in_temporal = clinical_ids - temporal_ids
    missing_in_demographic = clinical_ids - demographic_ids
    n_missing_temporal = len(missing_in_temporal)
    n_missing_demographic = len(missing_in_demographic)
    if missing_in_temporal:
        print(f"Missing in temporal: {n_missing_temporal} diagnoses (filtered by min_weeks)")
    if missing_in_demographic:
        print(f"Missing in demographic: {n_missing_demographic} diagnoses")
    return (
        common_ids,
        n_codes_clinical,
        n_codes_demographic,
        n_codes_temporal,
        n_missing_demographic,
        n_missing_temporal,
    )


@app.cell
def _(clinical_view, common_ids, demographic_view, temporal_view):
    clinical_distance_aligned = clinical_view.align_to_common_ids(common_ids)
    temporal_distance_aligned = temporal_view.align_to_common_ids(common_ids)
    demographic_distance_aligned = demographic_view.align_to_common_ids(common_ids)

    print("Aligned matrix shapes:")
    print(f"  Clinical: {clinical_distance_aligned.shape}")
    print(f"  Temporal: {temporal_distance_aligned.shape}")
    print(f"  Demographic: {demographic_distance_aligned.shape}")

    assert (
        clinical_distance_aligned.shape
        == temporal_distance_aligned.shape
        == demographic_distance_aligned.shape
    )
    return (
        clinical_distance_aligned,
        demographic_distance_aligned,
        temporal_distance_aligned,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Normalize distance matrices

    Normalize each distance matrix to [0, 1] before computing affinities.
    This ensures the adaptive Gaussian kernel in `affinity_matrix` operates on
    comparable scales across all views, so no single view dominates the fusion.
    """)
    return


@app.cell
def _(
    clinical_distance_aligned,
    demographic_distance_aligned,
    normalize_distance,
    temporal_distance_aligned,
):
    clinical_distance_norm = normalize_distance(clinical_distance_aligned)
    temporal_distance_norm = normalize_distance(temporal_distance_aligned)
    demographic_distance_norm = normalize_distance(demographic_distance_aligned)

    print(
        f"Clinical distance range after norm:    [{clinical_distance_norm.min():.4f}, {clinical_distance_norm.max():.4f}]"
    )
    print(
        f"Temporal distance range after norm:    [{temporal_distance_norm.min():.4f}, {temporal_distance_norm.max():.4f}]"
    )
    print(
        f"Demographic distance range after norm: [{demographic_distance_norm.min():.4f}, {demographic_distance_norm.max():.4f}]"
    )
    return clinical_distance_norm, demographic_distance_norm, temporal_distance_norm


@app.cell
def _(mo):
    mo.md(r"""
    ### Convert distances to affinities

    Using SNF library's affinity_matrix with adaptive Gaussian kernel.
    """)
    return


@app.cell
def _(
    clinical_distance_norm,
    compute,
    demographic_distance_norm,
    np,
    temporal_distance_norm,
):
    # Use SNF library's affinity_matrix with adaptive kernel
    # K is number of neighbors for local scaling (different from SNF's K parameter)
    affinity_k = 20  # default
    affinity_mu = 0.5  # default

    def _to_affinity(dist):
        W = compute.affinity_matrix(dist, K=affinity_k, mu=affinity_mu)
        # Normalize to [0, 1] so all views enter SNF on equal footing.
        # SNF assumes views are comparable; without this the view with the
        # largest raw affinity scale dominates the matrix product in fusion.
        np.fill_diagonal(W, 0.0)
        max_val = W.max()
        if max_val > 0:
            W = W / max_val
        return W

    clinical_affinity = _to_affinity(clinical_distance_norm)
    temporal_affinity = _to_affinity(temporal_distance_norm)
    demographic_affinity = _to_affinity(demographic_distance_norm)

    print(
        f"Clinical affinity range:    [{clinical_affinity.min():.4f}, {clinical_affinity.max():.4f}]"
    )
    print(
        f"Temporal affinity range:    [{temporal_affinity.min():.4f}, {temporal_affinity.max():.4f}]"
    )
    print(
        f"Demographic affinity range: [{demographic_affinity.min():.4f}, {demographic_affinity.max():.4f}]"
    )
    return clinical_affinity, demographic_affinity, temporal_affinity


@app.cell
def _(clinical_affinity, demographic_affinity, temporal_affinity):
    assert clinical_affinity.shape == temporal_affinity.shape == demographic_affinity.shape, (
        clinical_affinity.shape,
        temporal_affinity.shape,
        demographic_affinity.shape,
    )

    affinities = [clinical_affinity, temporal_affinity, demographic_affinity]
    return (affinities,)


@app.cell
def _(n_clusters, snf_k):
    print(f"Using parameters: K={snf_k}, n_clusters={n_clusters}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## SNF Fusion

    Apply SNF with optimal parameters to fuse the three views.
    """)
    return


@app.cell
def _(affinities, compute, np, snf_k):
    print(f"Running SNF with K={snf_k}...")
    fused = compute.snf(affinities, K=snf_k)

    print(f"Contains NaN: {np.isnan(fused).any()}")
    print(f"Contains Inf: {np.isinf(fused).any()}")
    fused = np.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=0.0)

    print(f"Fused matrix shape: {fused.shape}")
    print(f"Fused matrix range: [{fused.min():.4f}, {fused.max():.4f}]")
    return (fused,)


@app.cell
def _(fused):
    fused
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Clustering
    """)
    return


@app.cell
def _(cluster, fused, n_clusters, np):
    # Apply spectral clustering with optimal number of clusters
    print(f"Performing spectral clustering with n_clusters={n_clusters}...")
    fused_labels = cluster.spectral_clustering(fused, n_clusters=n_clusters)

    # Show cluster distribution
    unique, counts = np.unique(fused_labels, return_counts=True)
    print("\nCluster distribution:")
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} diagnoses")
    return (fused_labels,)


@app.cell
def _(fused, umap):
    reducer = umap.UMAP()
    embedding_spectral = reducer.fit_transform(fused)
    return (embedding_spectral,)


@app.cell
def _(embedding_spectral, fused_labels, go, np, pl):
    df_viz = pl.DataFrame(
        {
            "umap_x": embedding_spectral[:, 0],
            "umap_y": embedding_spectral[:, 1],
            "cluster_label": fused_labels,
        }
    )

    unique_clusters = sorted(df_viz["cluster_label"].unique().to_list())

    _fig = go.Figure()

    # LAYER A: The "Ghost" Background
    # This displays ALL points in light grey so you always see global context.
    _fig.add_trace(
        go.Scattergl(
            x=df_viz["umap_x"],
            y=df_viz["umap_y"],
            mode="markers",
            marker=dict(color="rgba(200, 200, 200, 0.3)", size=4),  # Light grey, semi-transparent
            name="All Data",
            hoverinfo="skip",  # Don't show hover data for the background noise
        )
    )

    # LAYER B: The Individual Cluster Highlights
    # We create one trace per cluster, but set them to visible=False initially.
    for label in unique_clusters:
        # Filter using Polars (very fast)
        subset = df_viz.filter(pl.col("cluster_label") == label)

        _fig.add_trace(
            go.Scattergl(
                x=subset["umap_x"],
                y=subset["umap_y"],
                mode="markers",
                # You can change 'red' to any hex color you prefer (e.g., '#EF553B')
                marker=dict(color="red", size=6, opacity=1.0),
                name=f"Cluster {label}",
                visible=False,  # Hidden by default
                # Custom hover data
                customdata=np.stack((subset["cluster_label"],), axis=-1),
                hovertemplate="<b>Cluster %{customdata[0]}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>",
            )
        )

    # --- 3. Construct the Dropdown Logic ---
    buttons = []

    # Button 1: "Reset/None" (Shows only background)
    buttons.append(
        dict(
            label="Select a Cluster...",
            method="update",
            args=[
                # Trace 0 (Background) is True. All others (1 to 200) are False.
                {"visible": [True] + [False] * len(unique_clusters)},
                {"title": "Global View (Spectral + UMAP)"},
            ],
        )
    )

    # Buttons 2...N: One for each cluster
    for i, label in enumerate(unique_clusters):
        # Logic:
        # [0] is Background -> Always True
        # [i+1] is the current Cluster -> True
        # All others -> False
        visibility = [False] * (len(unique_clusters) + 1)
        visibility[0] = True
        visibility[i + 1] = True

        buttons.append(
            dict(
                label=f"Cluster {label}",
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"Highlighting Cluster {label}"},  # Update chart title dynamically
                ],
            )
        )

    # --- 4. Final Layout ---
    _fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                x=1.15,
                y=1,  # Place menu to the right
                xanchor="left",
                yanchor="top",
            )
        ],
        showlegend=False,  # Hide the legend (the dropdown replaces it)
        width=1000,
        height=700,
        title="Global View (Spectral + UMAP)",
        plot_bgcolor="white",
        margin=dict(r=150),  # Add right margin to accommodate dropdown
    )
    _fig
    return


@app.cell
def _(common_ids, fused_labels, pl):
    df_clusters = pl.DataFrame({"cluster": fused_labels, "terminology_id": common_ids})
    df_clusters
    return (df_clusters,)


@app.cell
def _(df_clusters, mo, pl, terminology_labels):
    # Identify largest cluster and inspect its contents
    _sizes = df_clusters.group_by("cluster").agg(pl.len().alias("size"))
    _giant_cluster_id = _sizes.sort("size", descending=True)["cluster"][0]

    _giant_members = (
        df_clusters.filter(pl.col("cluster") == _giant_cluster_id)
        .join(terminology_labels, left_on="terminology_id", right_on="id")
        .select(["code", "code_type", "description"])
        .sort(["code_type", "code"])
    )

    _by_type = (
        _giant_members.group_by("code_type")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    print(f"Giant cluster: {_giant_cluster_id} ({_giant_members.height} codes)")
    print("\nBreakdown by code_type:")
    print(_by_type)
    print("\nAll codes in the giant cluster:")
    if mo.app_meta().mode != "script":
        _giant_members
    else:
        print(_giant_members)
    return


@app.cell
def _(fused, fused_labels, metrics, n_permutations, AFFINITY_ZSCORE_SEED):
    # Compute comprehensive evaluation metrics
    silhouette = metrics.silhouette_score(fused, fused_labels)
    zaff = metrics.affinity_zscore(
        fused, fused_labels, n_perms=n_permutations, seed=AFFINITY_ZSCORE_SEED
    )

    print("Clustering Evaluation Metrics:")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Affinity Z-Score: {zaff:.4f}")
    print("\nInterpretation:")
    print("  - Silhouette > 0.5: Good clustering")
    print("  - Affinity Z-Score > 2: Significantly better than random")
    return silhouette, zaff


@app.cell
def _(df_clusters, n_clusters, pl):
    cluster_sizes = (
        df_clusters.group_by("cluster").agg(pl.len().alias("size")).sort("size", descending=True)
    )

    total_clusters = cluster_sizes.height
    cluster_largest = int(cluster_sizes["size"].max())
    cluster_median = float(cluster_sizes["size"].median())
    n_singletons = cluster_sizes.filter(pl.col("size") == 1).height
    pct_singletons = (n_singletons / total_clusters) * 100

    print(f"--- Cluster Stats (K={n_clusters}) ---")
    print(f"Total Clusters:   {total_clusters}")
    print(f"Largest Cluster:  {cluster_largest} diagnoses")
    print(f"Median Size:      {cluster_median} diagnoses")
    print(f"Singletons:       {n_singletons} ({pct_singletons:.1f}%)")
    return cluster_largest, cluster_median, cluster_sizes, n_singletons, pct_singletons


@app.cell
def _(cluster_sizes, n_clusters, px):
    _fig = px.histogram(
        cluster_sizes,
        x="size",
        nbins=50,
        title=f"Cluster Size Distribution (K={n_clusters})",
        labels={"size": "Number of Diagnoses in Cluster"},
        text_auto=True,
    )

    _fig.update_layout(xaxis_title="Cluster Size", yaxis_title="Count of Clusters")
    _fig
    return


@app.cell
def _(cluster, common_ids, fused, metrics, np, pl, n_permutations, AFFINITY_ZSCORE_SEED):
    multi_k_results = []

    for _nr_cluster in [50, 100, 150, 200, 250, 300]:
        _labels = cluster.spectral_clustering(fused, n_clusters=_nr_cluster)
        _silhouette = metrics.silhouette_score(fused, _labels)
        _zaff = metrics.affinity_zscore(
            fused, _labels, n_perms=n_permutations, seed=AFFINITY_ZSCORE_SEED
        )

        _sizes = (
            pl.DataFrame({"cluster": _labels, "id": common_ids})
            .group_by("cluster")
            .agg(pl.len().alias("size"))
        )

        _unique, _counts = np.unique(_labels, return_counts=True)
        multi_k_results.append(
            {
                "n_clusters": _nr_cluster,
                "silhouette": round(_silhouette, 4),
                "affinity_zscore": round(float(_zaff), 4),
                "largest_cluster": int(_sizes["size"].max()),
                "median_size": float(_sizes["size"].median()),
            }
        )

    pl.DataFrame(multi_k_results)
    return (multi_k_results,)


@app.cell
def _(fused, igraph, leidenalg, np):
    _resolutions = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    _knn = 20
    _n = fused.shape[0]
    _sparse = np.zeros_like(fused)
    for _i in range(_n):
        _top = np.argpartition(fused[_i], -_knn)[-_knn:]
        _sparse[_i, _top] = fused[_i, _top]
    _sparse = np.maximum(_sparse, _sparse.T)

    _graph = igraph.Graph.Weighted_Adjacency(_sparse.tolist(), mode="undirected")

    leiden_results = []
    for _resolution in _resolutions:
        _partition = leidenalg.find_partition(
            _graph,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=_resolution,
            seed=42,
        )
        leiden_results.append({"resolution": _resolution, "n_communities": len(_partition)})

    leiden_results
    return (leiden_results,)


@app.cell
def _(
    cluster_largest,
    cluster_median,
    common_ids,
    df,
    extract_dataset_name,
    file_path,
    fused_labels,
    indicative_codes_enabled,
    insert_run_leiden,
    leiden_results,
    multi_k_results,
    n_clusters,
    n_codes_clinical,
    n_codes_demographic,
    n_codes_temporal,
    n_missing_demographic,
    n_missing_temporal,
    n_singletons,
    notes,
    pct_singletons,
    run_start,
    save_run,
    silhouette,
    snf_k,
    update_run_finished_at,
    year_max,
    year_min,
    zaff,
):
    run_id = save_run(
        dataset_name=extract_dataset_name(file_path),
        year_min=int(year_min),
        year_max=int(year_max),
        snf_k=int(snf_k),
        n_clusters=int(n_clusters),
        fused_labels=fused_labels,
        common_ids=common_ids,
        indicative_enabled=indicative_codes_enabled,
        silhouette_score=float(silhouette),
        affinity_zscore=float(zaff),
        notes=notes,
        created_at=run_start,
        df=df,
        n_codes_clinical=n_codes_clinical,
        n_codes_temporal=n_codes_temporal,
        n_codes_demographic=n_codes_demographic,
        n_missing_temporal=n_missing_temporal,
        n_missing_demographic=n_missing_demographic,
        cluster_largest=cluster_largest,
        cluster_median=cluster_median,
        n_singletons=n_singletons,
        pct_singletons=pct_singletons,
        multi_k_results=multi_k_results,
    )
    insert_run_leiden(run_id, leiden_results)
    update_run_finished_at(run_id)
    print(f"💾 Run {run_id} saved to virola.db")
    return (run_id,)


@app.cell
def _(mo):
    if mo.app_meta().mode == "script":
        print("✓ SNF clustering completed")
    return


if __name__ == "__main__":
    app.run()
