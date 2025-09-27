import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    from dtaidistance import dtw, similarity
    from sklearn import cluster
    from snf import compute, metrics

    from vir.db import (
        dimension_exists,
        fetch_terminologies,
        insert_or_update_dimension,
        read_dimension,
    )

    return (
        cluster,
        compute,
        dimension_exists,
        dtw,
        fetch_terminologies,
        insert_or_update_dimension,
        metrics,
        mo,
        np,
        pl,
        read_dimension,
        similarity,
    )


@app.cell
def _(pl):
    df = pl.read_parquet(
        "data/interim/cleaned_aesop_raw_dados_2017_20250223_20municipios_salvador.parquet"
    )
    df = df.filter(pl.col("year").ge(2022))
    df.head()
    return (df,)


@app.cell
def _(df):
    df.shape
    return


@app.cell
def _(fetch_terminologies):
    terminology_labels = fetch_terminologies()
    return (terminology_labels,)


@app.cell
def _(terminology_labels):
    terminology_labels
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Similarity Network Fusion

    ## Dimensions 🎲

    * **Clinical**: semantic similarity
    * **Temporal**: temporal frequency
    * **Demographic**: sex and gender characteristics

    Calculate the similarity in all these dimensions, having the diagnosis as the central factor.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Clinical dimension""")
    return


@app.cell
def _():
    # # each is an (n_diagnosis x n_diagnosis) NumPy array
    # clinical = np.array(...)
    # temporal = np.array(...)
    # demographics = np.array(...)

    # alpha = 0.5   # clinical weight
    # beta  = 0.3   # temporal weight
    # delta = 0.2   # demographic weight

    # similarity_matrix = (alpha * S_clinical) + (beta  * S_temporal) + (delta * S_demo)

    # # Optional: convert to distance matrix for clustering
    # distance_matrix = 1 - similarity_matrix

    # similarity_matrix, distance_matrix
    return


@app.cell
def _(np, pl, read_dimension):
    def load_dimension(dimension: str):
        """
        Lê os dados de uma dimensão do banco de dados e constrói uma matriz de afinidade.

        Retorna a matriz e a lista de IDs na ordem correspondente.
        """
        df = read_dimension(dimension)

        id_col1 = df.select("terminology_id_1").rename({"terminology_id_1": "id"})
        id_col2 = df.select("terminology_id_2").rename({"terminology_id_2": "id"})
        all_ids_df = pl.concat([id_col1, id_col2]).unique().sort("id")

        n = all_ids_df.height
        # Criar um mapa de ID para índice (0 a n-1)
        id2idx = {id_val: i for i, id_val in enumerate(all_ids_df["id"].to_list())}

        # Usar map_elements para converter IDs para índices
        i = (
            df["terminology_id_1"]
            .map_elements(lambda x: id2idx[x], return_dtype=pl.Int32)
            .to_numpy()
        )
        j = (
            df["terminology_id_2"]
            .map_elements(lambda x: id2idx[x], return_dtype=pl.Int32)
            .to_numpy()
        )
        v = df["value"].to_numpy()

        # Escalar a similaridade de cosseno de [-1, 1] para [0, 1]
        v_scaled = (v + 1) / 2

        mat = np.zeros((n, n), dtype=np.float32)
        mat[i, j] = v_scaled
        mat[j, i] = v_scaled  # Garantir simetria

        # Preencher a diagonal com 1 (auto-similaridade)
        np.fill_diagonal(mat, 1)

        return mat, all_ids_df["id"].to_list()

    return (load_dimension,)


@app.cell
def _(load_dimension):
    clinical_mat, clinical_ids = load_dimension("clinical")
    return clinical_ids, clinical_mat


@app.cell
def _(clinical_mat):
    clinical_mat
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Temporal dimension ⌛

    * Features: week, year
    """
    )
    return


@app.cell
def _(df, pl, terminology_labels):
    df_with_ids = df.join(terminology_labels, on=["code", "code_type"], how="left")
    df_with_ids = (
        df_with_ids.with_columns(
            (pl.col("year").cast(pl.Utf8) + "-" + pl.col("week").cast(pl.Utf8)).alias("year_week")
        )
        .group_by(["id", "year", "week", "year_week"])
        .agg(pl.col("quantity").sum().alias("total"))
    )

    temporal_pivot = df_with_ids.pivot(values="total", index="id", on="year_week").fill_null(0)

    year_weeks = (
        df_with_ids.select("year_week", "year", "week")
        .unique()
        .sort(["year", "week"])  # ordena cronologicamente
        .get_column("year_week")
        .to_list()
    )

    temporal_pivot = temporal_pivot.select(["id"] + year_weeks)
    temporal_pivot
    return (temporal_pivot,)


@app.cell
def _(pl, temporal_pivot):
    value_cols = [c for c in temporal_pivot.columns if c != "id"]
    pivot_scaled = temporal_pivot.with_columns(
        [
            (
                (pl.col(col) - pl.min_horizontal(pl.col(value_cols)))
                / (pl.max_horizontal(pl.col(value_cols)) - pl.min_horizontal(pl.col(value_cols)))
            ).alias(col)
            for col in value_cols
        ]
    )
    pivot_scaled
    return


@app.cell
def _(temporal_pivot):
    _labels = temporal_pivot["id"].to_list()
    _series = temporal_pivot.drop("id").to_numpy()

    # min-max scaling por linha
    _series = (_series - _series.min(axis=1, keepdims=True)) / (
        _series.max(axis=1, keepdims=True) - _series.min(axis=1, keepdims=True) + 1e-9
    )
    _series
    return


@app.cell
def _(dtw, pl):
    def make_temporal_matrix(temporal_pivot: pl.DataFrame, window: int | None = None):
        labels = temporal_pivot["id"].to_list()
        series = temporal_pivot.drop("id").to_numpy()

        series = (series - series.min(axis=1, keepdims=True)) / (
            series.max(axis=1, keepdims=True) - series.min(axis=1, keepdims=True) + 1e-9
        )
        series_list = [s for s in series]

        dist_matrix = dtw.distance_matrix(series_list, parallel=True, window=window)

        df_dtw = pl.DataFrame(dist_matrix, schema=labels).with_columns(pl.Series("id", labels))
        return df_dtw

    return (make_temporal_matrix,)


@app.cell
def _(make_temporal_matrix, temporal_pivot):
    df_dtw = make_temporal_matrix(temporal_pivot, window=10)
    return (df_dtw,)


@app.cell
def _(df_dtw, similarity):
    temporal_mat = similarity.distance_to_similarity(df_dtw)
    return (temporal_mat,)


@app.cell
def _():
    # hyperparameters
    K = 1494
    return (K,)


@app.cell
def _():
    # From: https://snfpy.readthedocs.io
    # While the appropriate numbers for these varies based on scenario,
    # a good rule is that K should be around N // 10,
    # and mu should be in the range (0.2 - 0.8).
    return


@app.cell
def _(
    clinical_ids,
    dimension_exists,
    insert_or_update_dimension,
    mo,
    np,
    pl,
    temporal_mat,
):
    if not dimension_exists("temporal"):
        mo.md("Salvando a dimensão temporal no banco de dados...")

        n = temporal_mat.shape[0]
        rows, cols = np.triu_indices(n, k=1)

        df_to_insert = pl.DataFrame(
            {
                "terminology_id_1": np.array(clinical_ids)[rows],
                "terminology_id_2": np.array(clinical_ids)[cols],
                "value": temporal_mat[rows, cols],
            }
        ).with_columns(pl.lit("temporal").alias("dimension"))

        insert_or_update_dimension(df_to_insert)

        mo.md("✅ Dimensão temporal salva.")
    else:
        mo.md("A dimensão temporal já existe no banco de dados.")
    return


@app.cell
def _(mo):
    mo.md(r"""## Affinities""")
    return


@app.cell
def _(clinical_mat, temporal_mat):
    # all views
    affinities = [clinical_mat, temporal_mat]
    return (affinities,)


@app.cell
def _(affinities, np):
    np.isnan(affinities).any()
    return


@app.cell
def _(K, affinities, compute, np):
    fused = compute.snf(affinities, K=K)  # 14 min
    print(np.isnan(fused).any())  # nan might occurr due to divisions to 0
    fused = np.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=0.0)
    return (fused,)


@app.cell
def _(mo):
    mo.md(r"""## Clustering 🩻""")
    return


@app.cell
def _(compute, fused):
    # estimate the number of clusters present in the fused matrix, derived via
    # an "eigengap" method (i.e., largest difference in eigenvalues of the
    # laplacian of the graph). note this function returns the top two options;
    # we'll only use the first
    first, second = compute.get_n_clusters(
        fused, range(2, 20)
    )  # TODO establish a better range given the number of codes
    return first, second


@app.cell
def _(first, second):
    first, second
    return


@app.cell
def _(cluster, first, fused):
    fused_labels = cluster.spectral_clustering(fused, n_clusters=first)
    return (fused_labels,)


@app.cell
def _(affinities, cluster, first, fused_labels, metrics, terminology_labels):
    # clustering depois da fusão
    labels = [terminology_labels["label"], fused_labels]
    for arr in affinities:
        labels += [cluster.spectral_clustering(arr, n_clusters=first)]
    nmi = metrics.nmi(labels)
    return (nmi,)


@app.cell
def _(nmi):
    nmi
    return


@app.cell
def _(fused, fused_labels, metrics):
    silhouette = metrics.silhouette_score(fused, fused_labels)  # nan se tiver apenas uma amostra!
    silhouette
    return


if __name__ == "__main__":
    app.run()
