import marimo

__generated_with = "0.19.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import plotly.express as px
    import polars as pl
    import umap
    from dotenv import load_dotenv
    from embedding_atlas.widget import EmbeddingAtlasWidget
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    return (
        AgglomerativeClustering,
        EmbeddingAtlasWidget,
        load_dotenv,
        np,
        pl,
        px,
        silhouette_score,
        umap,
    )


@app.cell
def _():
    from vir.helpers import get_n_jobs
    from vir.notebook_widgets import (
        create_file_dropdown,
        extract_dataset_name,
        load_and_prepare_dataframe,
        load_indicative_codes,
        load_terminology_labels,
    )
    from vir.views import ClinicalView

    return (
        ClinicalView,
        create_file_dropdown,
        extract_dataset_name,
        get_n_jobs,
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
def _(args):
    year_min = args.get("year_min", 2022)
    year_max = args.get("year_max", 2024)
    return year_max, year_min


@app.cell
def _(mo):
    mo.md(r"""
    # Clinical View 🏥

    Captures semantic and ontological similarity between diagnoses using embeddings.

    * **Input:** Diagnosis embeddings from `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`
    * **Embedding Construction:** Mean pooling of token embeddings (precomputed)
    * **Preprocessing:** Row-wise L2 normalization (done during embedding generation)
    * **Distance Metric:** Cosine distance
    * **Rationale:** Measures semantic similarity in embedding space independent of vector magnitude

    ## Database Schema

    The clinical dimension is precomputed and stored in the database:
    - **medical_terminologies table:** Contains embeddings (768-dimensional vectors)
    - **dimensions table:** Contains pairwise cosine similarities (dimension='clinical')
    - **conversion:** Distance = 1 - Similarity
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Fetch diagnosis terminology
    """)
    return


@app.cell
def _(load_terminology_labels):
    terminology_labels = load_terminology_labels()
    return (terminology_labels,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Select codes that are indicative
    """)
    return


@app.cell
def _(load_indicative_codes):
    include_codes, indicative_codes_enabled = load_indicative_codes()
    return (include_codes,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Load and filter dataset
    """)
    return


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
def _(extract_dataset_name, file_path):
    dataset_name = extract_dataset_name(file_path) if file_path else "unknown"
    dataset_name
    return (dataset_name,)


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
        df.head()  # Only display in interactive mode
    return (df,)


@app.cell
def _(df):
    df.shape
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Clinical View

    Using the `ClinicalView` class with automatic caching.
    """)
    return


@app.cell
def _(ClinicalView, dataset_name, df):
    clinical_view = ClinicalView(df=df, dataset_name=dataset_name)
    return (clinical_view,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Compute or load distance matrix

    The matrix is loaded from precomputed database values and cached.
    Re-running with the same data/params will load from cache.
    """)
    return


@app.cell
def _(clinical_view):
    clinical_distance_matrix = clinical_view.get_matrix()
    clinical_distance_matrix.shape
    return (clinical_distance_matrix,)


@app.cell
def _(clinical_distance_matrix):
    clinical_distance_matrix
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Visualize distance matrix
    """)
    return


@app.cell
def _(clinical_distance_matrix, px):
    _fig = px.imshow(
        clinical_distance_matrix,
        color_continuous_scale="viridis",
        title="Clinical Distance Matrix (Cosine Distance)",
        labels={"x": "Diagnosis", "y": "Diagnosis", "color": "Distance"},
        aspect="equal",
    )
    _fig.update_layout(height=600, width=700)
    _fig
    return


@app.cell
def _(clinical_distance_matrix, np):
    np.nanmin(clinical_distance_matrix), np.nanmax(clinical_distance_matrix)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Analysis: Embeddings

    ### Get embedding statistics
    """)
    return


@app.cell
def _(clinical_view):
    embedding_stats = clinical_view.get_embedding_stats()
    embedding_stats
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Get similarity matrix (for comparison)

    Clinical view returns distances by default, but we can also get similarities.
    """)
    return


@app.cell
def _(clinical_view):
    similarity_matrix = clinical_view.get_similarity_matrix()
    similarity_matrix
    return (similarity_matrix,)


@app.cell
def _(mo):
    mo.md(r"""
    Verify relationship: distance = 1 - similarity
    """)
    return


@app.cell
def _(clinical_distance_matrix, np, similarity_matrix):
    np.allclose(clinical_distance_matrix, 1.0 - similarity_matrix)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Distribution of similarities
    """)
    return


@app.cell
def _(np, px, similarity_matrix):
    # Flatten upper triangle (excluding diagonal)
    _upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]

    _fig = px.histogram(
        _upper_triangle,
        nbins=50,
        title="Distribution of Cosine Similarities",
        labels={"value": "Cosine Similarity", "count": "Frequency"},
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exploratory Analysis

    ### Find most similar diagnosis pairs
    """)
    return


@app.cell
def _(clinical_view, df, np, pl, similarity_matrix):
    _terminology_ids = sorted(df["terminology_id"].unique().to_list())

    # upper triangle indices
    _i, _j = np.triu_indices_from(similarity_matrix, k=1)
    _similarity_pairs = pl.DataFrame(
        {
            "idx1": _i,
            "idx2": _j,
            "similarity": similarity_matrix[_i, _j],
        }
    )

    # map indices to terminology IDs
    _id_map = {idx: tid for idx, tid in enumerate(_terminology_ids)}
    _similarity_pairs = _similarity_pairs.with_columns(
        [
            pl.col("idx1").replace(_id_map).alias("terminology_id_1"),
            pl.col("idx2").replace(_id_map).alias("terminology_id_2"),
        ]
    )

    # get embeddings to join descriptions
    embeddings_df = clinical_view.get_embeddings()
    _desc_map_1 = {row["id"]: row["description"] for row in embeddings_df.iter_rows(named=True)}
    _desc_map_2 = {
        row["id"]: row["code"] + ":" + row["code_type"]
        for row in embeddings_df.iter_rows(named=True)
    }

    similarity_pairs = _similarity_pairs.join(
        embeddings_df.select(["id", "description", "code", "code_type"]).rename(
            {"id": "terminology_id_1"}
        ),
        on=["terminology_id_1"],
        suffix="_1",
    ).join(
        embeddings_df.select(["id", "description", "code", "code_type"]).rename(
            {"id": "terminology_id_2"}
        ),
        on=["terminology_id_2"],
        suffix="_2",
    )
    return embeddings_df, similarity_pairs


@app.cell
def _(similarity_pairs):
    top_similar = (
        similarity_pairs.select(
            [
                "similarity",
                "code",
                "code_type",
                "description",
                "code_2",
                "code_type_2",
                "description_2",
            ]
        )
        .sort("similarity", descending=True)
        .head(20)
    )
    top_similar
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Find most dissimilar diagnosis pairs
    """)
    return


@app.cell
def _(similarity_pairs):
    bottom_similar = (
        similarity_pairs.select(
            [
                "similarity",
                "code",
                "code_type",
                "description",
                "code_2",
                "code_type_2",
                "description_2",
            ]
        )
        .sort("similarity")
        .head(20)
    )
    bottom_similar
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exploratory Clustering

    Use hierarchical clustering to explore semantic groupings.
    """)
    return


@app.cell
def _(AgglomerativeClustering, clinical_distance_matrix):
    clustering = AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average", distance_threshold=0.5
    )
    cluster_labels = clustering.fit_predict(clinical_distance_matrix)
    return (cluster_labels,)


@app.cell
def _(clinical_distance_matrix, cluster_labels, silhouette_score):
    sil_score = silhouette_score(clinical_distance_matrix, cluster_labels, metric="precomputed")
    sil_score
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Visualize clustered distance matrix
    """)
    return


@app.cell
def _(clinical_distance_matrix, cluster_labels, np, px):
    _sorted_indices = np.argsort(cluster_labels)
    _sorted_matrix = clinical_distance_matrix[_sorted_indices][:, _sorted_indices]

    _fig = px.imshow(
        _sorted_matrix,
        color_continuous_scale="viridis",
        title="Clinical Distance Matrix (sorted by cluster)",
        aspect="equal",
    )
    _fig.update_layout(height=700, width=700)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Cluster composition
    """)
    return


@app.cell
def _(cluster_labels, df, embeddings_df, pl):
    _terminology_ids = sorted(df["terminology_id"].unique().to_list())
    _cluster_df = pl.DataFrame(
        {
            "terminology_id": _terminology_ids,
            "cluster": cluster_labels,
        }
    )

    cluster_df = _cluster_df.join(
        embeddings_df.select(["id", "code", "code_type", "description", "embedding"]),
        left_on="terminology_id",
        right_on="id",
    )
    cluster_df
    return (cluster_df,)


@app.cell
def _(cluster_df, pl):
    cluster_composition = (
        cluster_df.group_by("cluster")
        .agg(
            [
                pl.len().alias("n_diagnoses"),
                pl.col("code").str.join(", ").alias("codes"),
            ]
        )
        .sort("cluster")
    )
    cluster_composition
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### UMAP for clusters and embeddings
    """)
    return


@app.cell
def _(cluster_df, get_n_jobs, np, pl, umap):
    _X = np.array(cluster_df["embedding"].to_list())
    _reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=get_n_jobs())
    _projections = _reducer.fit_transform(_X)
    df_projected = cluster_df.with_columns(
        [
            pl.Series(name="projection_x", values=_projections[:, 0]),
            pl.Series(name="projection_y", values=_projections[:, 1]),
        ]
    )
    return (df_projected,)


@app.cell
def _(EmbeddingAtlasWidget, df_projected):
    EmbeddingAtlasWidget(
        df_projected,
        text="description",
        x="projection_x",
        y="projection_y",
        neighbors="cluster",
        show_table=False,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Cache Management
    """)
    return


@app.cell
def _(clinical_view):
    clinical_view.get_cache_info()
    return


@app.cell
def _(cluster_labels, mo, sil_score):
    if mo.app_meta().mode == "script":
        n_clusters = len(set(cluster_labels))
        print("✓ Clinical view analysis completed")
        print(f"  Number of clusters: {n_clusters} Silhouette score: {sil_score}")
    return


if __name__ == "__main__":
    app.run()
