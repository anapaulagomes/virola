import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    from sklearn import cluster
    from snf import compute, metrics

    from vir.db import fetch_terminologies, read_dimension

    return (
        cluster,
        compute,
        fetch_terminologies,
        metrics,
        mo,
        np,
        pl,
        read_dimension,
    )


@app.cell
def _(pl):
    df = pl.read_parquet(
        "data/interim/cleaned_aesop_raw_dados_2017_20250223_20municipios_salvador.parquet"
    )
    df.head()
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
    def load_dimension(dimension: str) -> np.ndarray:
        df = read_dimension(dimension)

        id_col1 = df.select("terminology_id_1").rename({"terminology_id_1": "id"})
        id_col2 = df.select("terminology_id_2").rename({"terminology_id_2": "id"})
        all_ids = pl.concat([id_col1, id_col2]).unique().sort("id")

        n = all_ids.height
        map_df = all_ids.with_row_index("idx")

        df_with_indices = (
            df.join(map_df, left_on="terminology_id_1", right_on="id")
            .rename({"idx": "i"})
            .join(map_df, left_on="terminology_id_2", right_on="id")
            .rename({"idx": "j"})
        )

        i = df_with_indices["i"].to_numpy()
        j = df_with_indices["j"].to_numpy()
        v = df_with_indices["value"].to_numpy()

        # A similaridade de cosseno do DB varia de [-1, 1].
        # A matriz de afinidade precisa ter valores no intervalo [0, 1].
        # Portanto, escalamos os valores.
        v_scaled = (v + 1) / 2

        mat = np.zeros((n, n), dtype=np.float32)
        mat[i, j] = v_scaled
        mat[j, i] = v_scaled  # Garante a simetria

        # Preenche a diagonal com 1 (auto-similaridade máxima)
        np.fill_diagonal(mat, 1)

        return mat

    return (load_dimension,)


@app.cell
def _(load_dimension):
    clinical_mat = load_dimension("clinical")
    return (clinical_mat,)


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
def _():
    # TODO
    return


@app.cell
def _(mo):
    mo.md(r"""## Affinities""")
    return


@app.cell
def _(clinical_mat):
    # all views
    affinities = [clinical_mat]  # , genomic_mat, imaging_mat
    return (affinities,)


@app.cell
def _(affinities, np):
    np.isnan(affinities).any()
    return


@app.cell
def _(affinities, compute, np):
    fused = compute.snf(affinities)  # 14 min
    print(np.isnan(fused).any())  # nan might occurr due to divisions to 0
    fused = np.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=0.0)
    return (fused,)


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
def _(fetch_terminologies):
    terminology_labels = fetch_terminologies()
    return (terminology_labels,)


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
