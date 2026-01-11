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
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    import umap
    from dotenv import load_dotenv
    from sklearn import cluster
    from snf import compute, metrics

    return Path, cluster, compute, go, load_dotenv, metrics, mo, np, pl, px, umap


@app.cell
def _():
    from vir.config import CLUSTERING_DIR
    from vir.notebook_widgets import (
        create_file_dropdown,
        extract_dataset_name,
        load_and_prepare_dataframe,
        load_indicative_codes,
        load_terminology_labels,
    )
    from vir.views import ClinicalView, DemographicView, TemporalView

    return (
        CLUSTERING_DIR,
        ClinicalView,
        DemographicView,
        TemporalView,
        create_file_dropdown,
        extract_dataset_name,
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
def _(CLUSTERING_DIR, Path, args):
    year_min = args.get("year_min", 2022)
    year_max = args.get("year_max", 2024)
    snf_k = args.get("snf_k", 10)
    n_clusters = args.get("n_clusters", 200)
    output_dir = Path(args.get("output", str(CLUSTERING_DIR)))
    return n_clusters, output_dir, snf_k, year_max, year_min


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
def _(load_indicative_codes):
    include_codes, indicative_codes_enabled = load_indicative_codes()
    return (include_codes,)


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

    print(f"Clinical: {len(clinical_ids)} diagnoses")
    print(f"Temporal: {len(temporal_ids)} diagnoses")
    print(f"Demographic: {len(demographic_ids)} diagnoses")

    common_ids = sorted(clinical_ids & temporal_ids & demographic_ids)
    print(f"\nCommon diagnoses: {len(common_ids)}")

    missing_in_temporal = clinical_ids - temporal_ids
    missing_in_demographic = clinical_ids - demographic_ids
    if missing_in_temporal:
        print(f"Missing in temporal: {len(missing_in_temporal)} diagnoses (filtered by min_weeks)")
    if missing_in_demographic:
        print(f"Missing in demographic: {len(missing_in_demographic)} diagnoses")
    return (common_ids,)


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
    ### Convert distances to affinities

    Using SNF library's affinity_matrix with adaptive Gaussian kernel.
    """)
    return


@app.cell
def _(
    clinical_distance_aligned,
    compute,
    demographic_distance_aligned,
    temporal_distance_aligned,
):
    # Use SNF library's affinity_matrix with adaptive kernel
    # K is number of neighbors for local scaling (different from SNF's K parameter)
    affinity_k = 20  # default
    affinity_mu = 0.5  # default

    clinical_affinity = compute.affinity_matrix(
        clinical_distance_aligned, K=affinity_k, mu=affinity_mu
    )
    temporal_affinity = compute.affinity_matrix(
        temporal_distance_aligned, K=affinity_k, mu=affinity_mu
    )
    demographic_affinity = compute.affinity_matrix(
        demographic_distance_aligned, K=affinity_k, mu=affinity_mu
    )

    print(
        f"Clinical affinity range: [{clinical_affinity.min():.4f}, {clinical_affinity.max():.4f}]"
    )
    print(
        f"Temporal affinity range: [{temporal_affinity.min():.4f}, {temporal_affinity.max():.4f}]"
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
def _(clinical_view, fused_labels, pl):
    df_clusters = pl.DataFrame(
        {"cluster": fused_labels, "terminology_id": clinical_view.get_terminology_ids()}
    )
    df_clusters
    return (df_clusters,)


@app.cell
def _(fused, fused_labels, metrics):
    # Compute comprehensive evaluation metrics
    silhouette = metrics.silhouette_score(fused, fused_labels)
    # nmi_score = metrics.nmi([fused_labels])  # TODO compare with mappings?
    zaff = metrics.affinity_zscore(fused, fused_labels, n_perms=1000, seed=42)

    print("Clustering Evaluation Metrics:")
    print(f"  Silhouette Score: {silhouette:.4f}")
    # print(f"  NMI Score: {nmi_score}")
    print(f"  Affinity Z-Score: {zaff:.4f}")
    print("\nInterpretation:")
    print("  - Silhouette > 0.5: Good clustering")
    print("  - Affinity Z-Score > 2: Significantly better than random")
    return


@app.cell
def _(df_clusters, n_clusters, pl):
    cluster_sizes = (
        df_clusters.group_by("cluster").agg(pl.len().alias("size")).sort("size", descending=True)
    )

    total_clusters = cluster_sizes.height
    max_size = cluster_sizes["size"].max()
    median_size = cluster_sizes["size"].median()
    singletons = cluster_sizes.filter(pl.col("size") == 1).height
    dust_percent = (singletons / total_clusters) * 100

    print(f"--- Cluster Stats (K={n_clusters}) ---")
    print(f"Total Clusters:   {total_clusters}")
    print(f"Largest Cluster:  {max_size} diagnoses")
    print(f"Median Size:      {median_size} diagnoses")
    print(f"Singletons:       {singletons} ({dust_percent:.1f}%)")
    return (cluster_sizes,)


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
def _(cluster, fused, metrics):
    experiments_cluster_silhouette = {}

    for _nr_cluster in [20, 40, 60, 100, 150, 200, 250, 300]:
        _fused_labels = cluster.spectral_clustering(fused, n_clusters=_nr_cluster)
        _silhouette = metrics.silhouette_score(fused, _fused_labels)
        experiments_cluster_silhouette[_nr_cluster] = _silhouette

    experiments_cluster_silhouette
    return


@app.cell
def _(
    df_clusters,
    extract_dataset_name,
    file_path,
    n_clusters,
    output_dir,
    snf_k,
):
    output_file = (
        output_dir
        / f"clusters_{extract_dataset_name(file_path)}_nclusters={n_clusters}_k={snf_k}.json"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    df_clusters.write_json(output_file)
    print(f"💾 Results saved to: {output_file}")
    return (output_file,)


@app.cell
def _(mo):
    if mo.app_meta().mode == "script":
        print("✓ SNF clustering completed")
    return


if __name__ == "__main__":
    app.run()
