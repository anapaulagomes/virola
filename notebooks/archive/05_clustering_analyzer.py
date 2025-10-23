import marimo

__generated_with = "0.15.5"
app = marimo.App(width="full")


@app.cell
def _():
    import hdbscan
    import marimo as mo
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    from plotly.subplots import make_subplots
    from sklearn.decomposition import PCA
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )
    from umap import UMAP

    from vir.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
    from vir.db import get_all_embeddings
    from vir.features import AGE_GROUP
    from vir.viz import plot_categories

    return (
        AGE_GROUP,
        INTERIM_DATA_DIR,
        PCA,
        PROCESSED_DATA_DIR,
        UMAP,
        calinski_harabasz_score,
        davies_bouldin_score,
        get_all_embeddings,
        go,
        hdbscan,
        make_subplots,
        mo,
        np,
        pl,
        plot_categories,
        px,
        silhouette_score,
    )


@app.cell
def _(mo):
    mo.md(r"""# 🧬 [AESOP] Clustering""")
    return


@app.cell
def _(PROCESSED_DATA_DIR, mo):
    _options = [
        str(_file.resolve())
        for _file in PROCESSED_DATA_DIR.glob("**/processed_optimized_*.parquet")
    ]
    files_dropdown = mo.ui.dropdown(
        options=_options,
        label="Choose one dataset",
        searchable=True,
    )
    files_dropdown
    return (files_dropdown,)


@app.cell
def _(files_dropdown, pl):
    df = pl.read_parquet(files_dropdown.value)
    features = set([_col.split("_")[0] for _col in df.columns])
    return df, features


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(features):
    features
    return


@app.cell
def _(df, features, mo):
    # Filter for embeddings and temporal features only
    _feature_options = [
        _feature
        for _feature in features
        if _feature in ["embed", "quantity"] and _feature != "unique"
    ]

    data_batch = mo.md("""
    **Processed data loaded (Embeddings + Temporal Features)**

    - Total records: {total_records}
    - Shape: {shape}
    - Features: {features}
    """).batch(
        total_records=mo.ui.text(value=str(len(df)), disabled=True),
        shape=mo.ui.text(value=str(df.shape), disabled=True),
        features=mo.ui.multiselect(
            options=_feature_options,
            value=_feature_options,
        ),
    )
    data_batch
    return (data_batch,)


@app.cell
def _(data_batch, df):
    selected_features = [
        _col for _col in df.columns if _col.split("_")[0] in data_batch.value["features"]
    ]
    return (selected_features,)


@app.cell
def _(mo):
    mo.md(r"""## 📊 Dimensionality Reduction Analysis (Embeddings + Temporal Features)""")
    return


@app.cell
def _(PCA, data_batch, df, mo, selected_features):
    feature_df = df[selected_features]
    unique_ids = df.select("unique_id")

    pca = PCA(n_components=10)
    pca.fit_transform(feature_df.to_numpy())
    prop_var = pca.explained_variance_ratio_
    eigenvalues = pca.explained_variance_

    mo.md(f"""
    **PCA Analysis Complete**

    - Input features: {data_batch.value["features"]}
    - PCA components: {pca.n_components_}
    - Total variance explained: {prop_var.sum():.1%}
    - First 3 components: {prop_var[:3].sum():.1%}
    """)
    return eigenvalues, feature_df, pca, prop_var, unique_ids


@app.cell
def _(eigenvalues, mo, prop_var):
    eigenvalue_info = "\n".join(
        [
            f"PC{_index + 1}: eigenvalue={ev:.2f}, variance={pv:.1%}"
            for _index, (ev, pv) in enumerate(zip(eigenvalues, prop_var))
        ]
    )

    mo.md(f"""
    **Eigenvalues (variance explained by each component)**

    ```
    {eigenvalue_info}
    ```
    """)
    return


@app.cell
def _(go, np, pca, prop_var):
    _fig = go.Figure(
        data=go.Scatter(
            x=np.arange(pca.n_components_) + 1,
            y=prop_var,
            mode="lines+markers",
            marker=dict(color="red", size=8),
            line=dict(color="red", width=2),
            name="Variance Explained",
        )
    )

    _fig.update_layout(
        title="PCA: Variance explained by each component",
        yaxis_title="Proportion of Variance Explained",
        xaxis_title="Principal Component",
        showlegend=False,
        height=500,
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""## 🎯 Clustering Analysis with HDBSCAN (Embeddings + Temporal)""")
    return


@app.cell
def _(PCA, feature_df):
    # Reduce dimensions for clustering
    reduced_features = PCA(n_components=3).fit_transform(feature_df.to_numpy())
    return (reduced_features,)


@app.cell
def _(feature_df, mo, np, reduced_features):
    # Calculate optimal HDBSCAN parameters based on data size
    n_samples = len(feature_df)
    default_optimal_min_cluster_size = int(np.sqrt(n_samples))
    default_optimal_min_samples = max(1, default_optimal_min_cluster_size // 2)

    params = mo.md("""
    **HDBSCAN Parameters Calculated**

    - Data size: {n_samples} samples
    - min_cluster_size: {optimal_min_cluster_size} (≈ √{n_samples:,})
    - min_samples: {optimal_min_samples} (default: min_cluster_size // 2)
    - Reduced feature matrix: {reduced_features_shape}
    """).batch(
        n_samples=mo.ui.text(value=str(n_samples), disabled=True),
        reduced_features_shape=mo.ui.text(value=str(reduced_features.shape), disabled=True),
        optimal_min_cluster_size=mo.ui.slider(
            start=default_optimal_min_cluster_size,
            stop=n_samples,
            value=default_optimal_min_cluster_size,
        ),
        optimal_min_samples=mo.ui.slider(
            start=default_optimal_min_cluster_size // 4,
            stop=n_samples,
            value=default_optimal_min_samples,
        ),
    )

    params
    return (params,)


@app.cell
def _(hdbscan, mo, params, reduced_features):
    optimal_min_cluster_size = params.value["optimal_min_cluster_size"]
    optimal_min_samples = params.value["optimal_min_samples"]

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=optimal_min_cluster_size,
        min_samples=optimal_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    cluster_labels = clusterer.fit_predict(reduced_features)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    mo.md(f"""
    **HDBSCAN Clustering Results (Optimized Parameters)**

    - Parameters: min_cluster_size={optimal_min_cluster_size}, min_samples={optimal_min_samples}
    - Clusters found: {n_clusters}
    - Noise points: {n_noise:,}
    - Clustered points: {len(cluster_labels) - n_noise:,}
    - Clustering coverage: {(len(cluster_labels) - n_noise) / len(cluster_labels):.1%}
    """)
    return cluster_labels, clusterer


@app.cell
def _(cluster_labels, df, mo, pl, unique_ids):
    # Create cluster results with unique ID tracking
    cluster_results = pl.concat(
        [unique_ids, pl.DataFrame({"cluster": cluster_labels})], how="horizontal"
    )

    mo.md(f"""
    **Cluster Assignment Complete**

    - Total records: {len(df):,}
    - Total records with cluster assignments: {len(cluster_results):,}
    - Unique clusters: {cluster_results["cluster"].n_unique()}
    - Sample assignments shown in table below
    """)
    return (cluster_results,)


@app.cell
def _(mo):
    mo.md(r"""## 📈 Clustering Quality Metrics""")
    return


@app.cell
def _(
    calinski_harabasz_score,
    cluster_labels,
    clusterer,
    davies_bouldin_score,
    mo,
    np,
    reduced_features,
    silhouette_score,
):
    # Calculate clustering quality metrics (only for non-noise points)
    enable_metrics = False
    if enable_metrics:
        mask = cluster_labels != -1
        if mask.sum() > 1:  # Need at least 2 points for metrics
            clustered_data = reduced_features[mask]
            clustered_labels = cluster_labels[mask]

            if len(np.unique(clustered_labels)) > 1:  # Need more than 1 cluster
                silhouette = silhouette_score(clustered_data, clustered_labels)
                calinski_harabasz = calinski_harabasz_score(clustered_data, clustered_labels)
                davies_bouldin = davies_bouldin_score(clustered_data, clustered_labels)
            else:
                silhouette = calinski_harabasz = davies_bouldin = "N/A (only 1 cluster)"
        else:
            silhouette = calinski_harabasz = davies_bouldin = "N/A (insufficient data)"

        # Get cluster stability scores if available
        cluster_persistence = getattr(clusterer, "cluster_persistence_", "N/A")

        mo.md(f"""
        **Clustering Quality Metrics**

        - **Silhouette Score**: {silhouette if isinstance(silhouette, str) else f"{silhouette:.3f}"}
          _(Range: -1 to 1, higher is better)_
        - **Calinski-Harabasz Score**: {calinski_harabasz if isinstance(calinski_harabasz, str) else f"{calinski_harabasz:.2f}"}
          _(Higher is better)_
        - **Davies-Bouldin Score**: {davies_bouldin if isinstance(davies_bouldin, str) else f"{davies_bouldin:.3f}"}
          _(Lower is better)_
        - **HDBSCAN Cluster Persistence**: {cluster_persistence}
        """)
    return


@app.cell
def _(mo):
    mo.md(r"""## 🔍 Cluster Member Explorer""")
    return


@app.cell
def _(UMAP):
    umap_reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    return (umap_reducer,)


@app.cell
def _(get_all_embeddings, mo, np):
    @mo.cache
    def load_embeddings():
        embeddings_data = get_all_embeddings()
        if not embeddings_data:
            return None, None, None, None

        embeddings = np.array([item["embedding"] for item in embeddings_data])
        codes = [item["code"] for item in embeddings_data]
        code_types = [item["code_type"] for item in embeddings_data]
        descriptions = [item["description"] for item in embeddings_data]

        return embeddings, codes, code_types, descriptions

    embeddings, codes, code_types, descriptions = load_embeddings()
    return code_types, codes, descriptions, embeddings


@app.cell
def _(embeddings, mo, umap_reducer):
    @mo.cache
    def compute_umap(embeddings):
        """Compute UMAP embeddings for visualization."""
        if embeddings is None:
            return None
        return umap_reducer.fit_transform(embeddings)

    umap_coords = compute_umap(embeddings)
    return (umap_coords,)


@app.cell
def _(INTERIM_DATA_DIR, cluster_results, files_dropdown, pl):
    # expected format: cleaned_aesop_raw_dados_2017_20250223_20municipios_082025_ibitiara.parquet
    # 20municipios for all and city name for individual analysis
    target = files_dropdown.value.replace(".parquet", "").rsplit("_", 1)[1]

    df_with_clusters = (
        pl.read_parquet(
            [_file for _file in INTERIM_DATA_DIR.glob(f"**/cleaned_*{target}.parquet")][0]
        )
        .join(cluster_results, on="unique_id")
        .drop(["state", "city", "region", "population_range"])
    )
    return (df_with_clusters,)


@app.cell
def _(df_with_clusters):
    df_with_clusters.group_by(["cluster", "age_group"]).len().sort(by=["cluster"])
    return


@app.cell
def _(df_with_clusters, px):
    _fig = px.histogram(df_with_clusters.group_by(["cluster"]).len("code"), x="cluster", y="code")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""## Individual cluster inspection""")
    return


@app.cell
def _(cluster_labels, mo):
    cluster_dropdown = mo.ui.dropdown(
        options=sorted(set([int(cl) for cl in cluster_labels])),
        label="Choose one cluster",
        searchable=True,
    )
    cluster_dropdown
    return (cluster_dropdown,)


@app.cell
def _(PROCESSED_DATA_DIR, pl):
    codes_df = pl.read_csv(f"{PROCESSED_DATA_DIR}/cid10_ciap2_ab.csv").with_columns(
        code_type=pl.col("code_type").str.to_uppercase()
    )
    return (codes_df,)


@app.cell
def _(cluster_dropdown, codes_df, df_with_clusters, pl):
    current_selection = (
        df_with_clusters.filter(pl.col("cluster").eq(cluster_dropdown.value))
        .sort(descending=False, by="code")
        .unique()
    )
    current_selection = current_selection.join(codes_df, on=["code", "code_type"])
    current_selection
    return (current_selection,)


@app.cell
def _(embeddings):
    from sklearn.decomposition import IncrementalPCA

    ipca = IncrementalPCA(n_components=64, batch_size=1000)
    reduced_embeddings = ipca.fit_transform(embeddings)
    return (reduced_embeddings,)


@app.cell
def _(px):
    color_mapping = {
        "CID10": px.colors.qualitative.Plotly[0],
        "CIAP2": px.colors.qualitative.Plotly[1],
        "AB": px.colors.qualitative.Plotly[2],
    }
    return (color_mapping,)


@app.cell
def _(
    code_types,
    codes,
    color_mapping,
    current_selection,
    descriptions,
    go,
    reduced_embeddings,
    umap_coords,
):
    def compute_similarities_and_plot(
        cluster_selection, embeddings, umap_coords, codes, code_types, descriptions
    ):
        if cluster_selection.is_empty() or embeddings is None:
            return go.Figure(), []

        # Create mapping from code to index in the full embeddings dataset
        code_to_index = {code: idx for idx, code in enumerate(codes)}

        # Get cluster codes and find their indices in the full dataset
        cluster_codes = cluster_selection["code"].to_list()
        cluster_indices = []
        results_data = []

        for code in cluster_codes:
            if code in code_to_index:
                idx = code_to_index[code]
                cluster_indices.append(idx)
                results_data.append(
                    {
                        "code": codes[idx],
                        "code_type": code_types[idx],
                        "description": descriptions[idx],
                    }
                )

        fig = go.Figure().add_annotation(
            text="",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.add_trace(
            go.Scatter(
                x=umap_coords[:, 0],
                y=umap_coords[:, 1],
                mode="markers",
                marker=dict(size=6, color="lightgray", opacity=0.6),
                text=[f"{code}: {desc}" for code, desc in zip(codes, descriptions)],
                hovertemplate="<b>%{text}</b><br>UMAP: (%{x:.2f}, %{y:.2f})<extra></extra>",
                name="All terms",
                # showlegend=False
            )
        )

        # Highlight cluster points with colors by code type
        for idx in cluster_indices:
            code_type = code_types[idx]
            color = color_mapping.get(code_type.upper(), "gray")

            fig.add_trace(
                go.Scatter(
                    x=[umap_coords[idx, 0]],
                    y=[umap_coords[idx, 1]],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=color,
                    ),
                    hovertemplate=f"<b>{codes[idx]}: {descriptions[idx]}</b><br>UMAP: (%{{x:.2f}}, %{{y:.2f}})<extra></extra>",
                    name=code_type,
                    showlegend=False,
                )
            )

        # legend for code types
        for code_type, color in color_mapping.items():
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=8, color=color),
                    name=code_type,
                    showlegend=True,
                )
            )

        fig.update_layout(
            title="Medical terminologies",
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            hovermode="closest",
            width=800,
            height=600,
            margin=dict(l=50, r=50, t=80, b=50),
        )
        return fig, results_data

    similarities_fig, results_data = compute_similarities_and_plot(
        current_selection, reduced_embeddings, umap_coords, codes, code_types, descriptions
    )
    return (similarities_fig,)


@app.cell
def _(
    AGE_GROUP,
    cluster_dropdown,
    current_selection,
    make_subplots,
    plot_categories,
):
    _fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{}, {}],
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
        ],
    )

    _fig_sex = plot_categories(current_selection, "sex")
    _fig_age = plot_categories(current_selection, "age_group", x_categoryorder=AGE_GROUP)
    _fig_code = plot_categories(current_selection.sort("code"), "code")
    _fig.add_trace(_fig_sex["data"][0], row=1, col=1)
    _fig.add_trace(_fig_age["data"][0], row=1, col=2)
    _fig.add_trace(_fig_code["data"][0], row=2, col=1)
    _fig.update_layout(title_text=f"Composition of the cluster {cluster_dropdown.value}")

    _fig
    return


@app.cell
def _(similarities_fig):
    similarities_fig
    return


@app.cell
def _(mo):
    mo.md(r"""## 💾 Export Cluster Results with Traceability""")
    return


@app.cell
def _(PROCESSED_DATA_DIR, clusterer, files_dropdown):
    model = str(clusterer)
    model = model.replace("(", "_").replace(")", "").replace(" ", "").replace(",", "_")
    output_path = (
        PROCESSED_DATA_DIR / f"cluster_results_{model}_{files_dropdown.value.rsplit('/', 1)[1]}"
    )
    return (output_path,)


@app.cell
def _(cluster_results, mo, output_path, pl):
    cluster_results.write_parquet(output_path)

    mo.md(f"""
    **✅ Interactive cluster results exported successfully!**

    - **Location**: `{output_path}`
    - **Records**: {len(cluster_results):,}
    - **Clusters**: {cluster_results.filter(pl.col("cluster") != -1)["cluster"].n_unique()}
    - **Traceability**: Each cluster assignment linked to original raw data via unique_id

    **Usage**: Load this file to trace cluster assignments back through the entire pipeline:
    ```python
    # Full traceability example
    cluster_data = pl.read_parquet("{output_path}")
    raw_data_with_clusters = raw_df.join(cluster_data, on="unique_id")
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
