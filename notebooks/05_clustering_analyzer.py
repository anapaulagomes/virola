import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import hdbscan
    import marimo as mo
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    from sklearn.decomposition import PCA
    from plotly.subplots import make_subplots
    from pathlib import Path

    from vir.config import PROCESSED_DATA_DIR
    return PCA, PROCESSED_DATA_DIR, Path, go, hdbscan, mo, np, pl


@app.cell
def _(mo):
    mo.md(r"""# 🧬 [AESOP] Clustering""")
    return


@app.cell
def _(PROCESSED_DATA_DIR, Path, mo):
    _options = [str(_file.relative_to(Path.cwd())) for _file in PROCESSED_DATA_DIR.glob("**/processed_optimized_*.parquet")]
    files_dropdown = mo.ui.dropdown(
        options=_options,
        label="Choose one",
        searchable=True,
    )
    files_dropdown
    return (files_dropdown,)


@app.cell
def _():
    # selected_file = files_dropdown.value
    return


@app.cell
def _(files_dropdown, pl):
    df = pl.read_parquet(files_dropdown.value)
    features = set([_col.split('_')[0] for _col in df.columns])
    return df, features


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df, features, mo):
    _feature_options = [_feature for _feature in features if _feature != "unique"]  # exclude unique_id by now

    data_batch = mo.md("""
    **Processed data loaded**

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
    selected_features = [_col for _col in df.columns if _col.split('_')[0] in data_batch.value['features']]
    return (selected_features,)


@app.cell
def _(mo):
    mo.md(r"""## 📊 Dimensionality Reduction Analysis""")
    return


@app.cell
def _(PCA, data_batch, df, mo, selected_features):
    feature_df = df.drop("unique_id")
    feature_df = df[selected_features]
    unique_ids = df.select("unique_id")

    pca = PCA(n_components=10)
    pca_features = pca.fit_transform(feature_df.to_numpy())
    prop_var = pca.explained_variance_ratio_
    eigenvalues = pca.explained_variance_

    mo.md(f"""
    **PCA Analysis Complete**

    - Input features: {data_batch.value['features']}
    - PCA components: {pca.n_components_}
    - Total variance explained: {prop_var.sum():.1%}
    - First 3 components: {prop_var[:3].sum():.1%}
    """)
    return eigenvalues, feature_df, pca, prop_var, unique_ids


@app.cell
def _(eigenvalues, mo, prop_var):
    eigenvalue_info = "\n".join([
        f"PC{_index + 1}: eigenvalue={ev:.2f}, variance={pv:.1%}"
        for _index, (ev, pv) in enumerate(zip(eigenvalues, prop_var))
    ])

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
    mo.md(r"""## 🎯 Clustering Analysis with HDBSCAN""")
    return


@app.cell
def _(PCA, feature_df):
    # Reduce dimensions for clustering
    reduced_features = PCA(n_components=3).fit_transform(feature_df.to_numpy())
    return (reduced_features,)


@app.cell
def _():
    return


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
        optimal_min_cluster_size=mo.ui.slider(start=default_optimal_min_cluster_size, stop=n_samples, value=default_optimal_min_cluster_size),
        optimal_min_samples=mo.ui.slider(start=default_optimal_min_cluster_size//4, stop=n_samples, value=default_optimal_min_samples),
    )

    params
    return (params,)


@app.cell
def _(params):
    params.value
    return


@app.cell
def _(hdbscan, mo, params, reduced_features):
    optimal_min_cluster_size = params.value['optimal_min_cluster_size']
    optimal_min_samples = params.value['optimal_min_samples']

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=optimal_min_cluster_size,
        min_samples=optimal_min_samples,
        metric="euclidean",
        cluster_selection_method="eom"
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
def _(cluster_labels, mo, pl, unique_ids):
    # Create cluster results with unique ID tracking
    cluster_results = pl.concat(
        [unique_ids, pl.DataFrame({"cluster": cluster_labels})], how="horizontal"
    )

    mo.md(f"""
    **Cluster Assignment Complete**

    - Total records with cluster assignments: {len(cluster_results):,}
    - Unique clusters: {cluster_results['cluster'].n_unique()}
    - Sample assignments shown in table below
    """)
    return (cluster_results,)


@app.cell
def _(cluster_results, mo):
    mo.ui.table(cluster_results)
    return


@app.cell
def _():
    from vir.config import INTERIM_DATA_DIR
    return (INTERIM_DATA_DIR,)


@app.cell
def _(cluster_results, mo):
    # Show cluster distribution
    cluster_dist = cluster_results["cluster"].value_counts().sort("cluster")

    mo.md("**Cluster Distribution**")
    mo.ui.table(cluster_dist)
    return


@app.cell
def _(mo):
    mo.md(r"""## 📈 Clustering Quality Metrics""")
    return


@app.cell
def _(cluster_labels, clusterer, mo, np, reduced_features):
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    # Calculate clustering quality metrics (only for non-noise points)
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
    cluster_persistence = getattr(clusterer, 'cluster_persistence_', 'N/A')

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
def _(INTERIM_DATA_DIR, cluster_results, files_dropdown, pl):
    # expected format: cleaned_aesop_raw_dados_2017_20250223_20municipios_082025_ibitiara.parquet
    # 20municipios for all and city name for individual analysis
    target = files_dropdown.value.replace('.parquet', '').rsplit('_', 1)[1]

    # INTERIM_DATA_DIR

    df_with_clusters = pl.read_parquet(
        [_file for _file in INTERIM_DATA_DIR.glob(f"**/cleaned_*{target}.parquet")][0]
    ).join(cluster_results, on="unique_id")
    return (df_with_clusters,)


@app.cell
def _(df_with_clusters):
    df_with_clusters
    return


@app.cell
def _(df_with_clusters, pl):
    df_with_clusters.filter(pl.col('cluster').eq(1))['code'].sort(descending=False).unique()
    return


@app.cell
def _():
    # TODO next
    # allow to select one cluster and see basic charts
    # composition: sex, age group
    # umap from embeddings?
    return


@app.cell
def _(mo):
    mo.md(r"""## 🎨 Interactive Cluster Visualization""")
    return


@app.cell
def _(mo):
    mo.md(r"""## 💾 Export Cluster Results with Traceability""")
    return


@app.cell
def _(PROCESSED_DATA_DIR, files_dropdown):
    output_path = PROCESSED_DATA_DIR / f"cluster_results_interactive{files_dropdown.value}"
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
