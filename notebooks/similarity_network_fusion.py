import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


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
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from aeon.distances import dtw_pairwise_distance
    from sklearn import cluster
    from snf import compute, metrics

    return cluster, compute, dtw_pairwise_distance, metrics, mo, np, pl, plt


@app.cell
def _():
    from vir.db import (
        dimension_exists,
        fetch_terminologies,
        insert_or_update_dimension,
        read_dimension,
    )

    return (
        dimension_exists,
        fetch_terminologies,
        insert_or_update_dimension,
        read_dimension,
    )


@app.cell
def _(mo):
    mo.md(r"""## Data loading""")
    return


@app.cell
def _(fetch_terminologies):
    terminology_labels = fetch_terminologies()
    return (terminology_labels,)


@app.cell
def _(terminology_labels):
    terminology_labels
    return


@app.cell
def _(pl, terminology_labels):
    df = pl.read_parquet(
        "data/interim/cleaned_aesop_raw_dados_2017_20250223_20municipios_salvador.parquet",
    )
    df = df.drop(["unique_id", "state", "city", "region", "population_range"])  # make it per city
    df = df.group_by(["year", "week", "sex", "age_group", "code_type", "code"]).agg(
        pl.col("quantity").sum()
    )
    df = df.join(terminology_labels, on=["code", "code_type"]).rename(
        {"id": "terminology_id", "label": "terminology_label"}
    )
    df = df.filter(pl.col("year").ge(2022))
    df.head()
    return (df,)


@app.cell
def _(df):
    terminology_to_ids = {
        terminology_id: _index
        for _index, terminology_id in enumerate(df["terminology_id"].unique())
    }
    ids_to_terminology = {
        _index: terminology_id
        for _index, terminology_id in enumerate(df["terminology_id"].unique())
    }
    return ids_to_terminology, terminology_to_ids


@app.cell
def _(np, terminology_to_ids):
    matrix_len = len(terminology_to_ids)
    empty_matrix = np.zeros((matrix_len, matrix_len), dtype=np.float32)
    return (empty_matrix,)


@app.cell
def _(empty_matrix):
    empty_matrix
    return


@app.cell
def _(mo):
    mo.md(r"""## Dimensions""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Clinical dimension""")
    return


@app.cell
def _(empty_matrix, np, pl, read_dimension, terminology_to_ids):
    def load_clinical_dimension():
        clinical_df = read_dimension("clinical")
        _ids = list(terminology_to_ids.keys())
        clinical_df = clinical_df.filter(
            pl.col("terminology_id_1").is_in(_ids) & pl.col("terminology_id_2").is_in(_ids)
        )

        i = clinical_df["terminology_id_1"].replace_strict(terminology_to_ids).to_numpy()
        j = clinical_df["terminology_id_2"].replace_strict(terminology_to_ids).to_numpy()
        v = clinical_df["value"].to_numpy()  # scale later

        clinical_matrix = empty_matrix.copy()
        clinical_matrix[i, j] = v
        clinical_matrix[j, i] = v

        np.fill_diagonal(clinical_matrix, 0)  # enforce 0
        return clinical_matrix

    return (load_clinical_dimension,)


@app.cell
def _(load_clinical_dimension):
    clinical_mat = load_clinical_dimension()
    return (clinical_mat,)


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
def _(df, pl):
    df_year_week = (
        df.with_columns(
            (pl.col("year").cast(pl.Utf8) + "-" + pl.col("week").cast(pl.Utf8)).alias("year_week")
        )
        .group_by(["terminology_id", "year", "week", "year_week"])
        .agg(pl.col("quantity").sum().alias("total"))
    )
    df_year_week
    return (df_year_week,)


@app.cell
def _(df_year_week):
    ordered_year_week = (
        df_year_week.select("year_week", "year", "week")
        .unique()
        .sort(["year", "week"])  # ordena cronologicamente
        .get_column("year_week")
        .to_list()
    )
    return (ordered_year_week,)


@app.cell
def _(df_year_week, ordered_year_week):
    temporal_pivot = (
        df_year_week.pivot(
            index="terminology_id",  # rows
            on="year_week",  # new columns
            values="total",  # cell value
        )
        .fill_null(0)
        .select(["terminology_id"] + ordered_year_week)
    )
    temporal_pivot
    return (temporal_pivot,)


@app.cell
def _(dtw_pairwise_distance, ordered_year_week, temporal_pivot):
    dtw_results = dtw_pairwise_distance(
        temporal_pivot.select(ordered_year_week).to_numpy(), n_jobs=-1
    )
    dtw_results
    return (dtw_results,)


@app.cell
def _():
    # dtw took 14 min with n jobs = -1
    return


@app.cell
def _(dtw_results, np):
    def load_temporal_dimension():
        temporal_matrix = dtw_results.copy()

        np.fill_diagonal(temporal_matrix, 0)  # enforce 0
        return temporal_matrix

    return (load_temporal_dimension,)


@app.cell
def _(load_temporal_dimension):
    temporal_matrix = load_temporal_dimension()
    return (temporal_matrix,)


@app.cell
def _(
    dimension_exists,
    ids_to_terminology,
    insert_or_update_dimension,
    mo,
    pl,
    temporal_matrix,
):
    if not dimension_exists("temporal"):
        mo.md("Salvando a dimensão temporal no banco de dados...")

        to_db = {
            "terminology_id_1": [],
            "terminology_id_2": [],
            "value": [],
        }
        for _out in range(len([1, 2])):
            for _in in range(len([1, 2])):
                print(
                    ids_to_terminology[_out], ids_to_terminology[_in], temporal_matrix[_out][_in]
                )
                to_db["terminology_id_1"].append(ids_to_terminology[_out])
                to_db["terminology_id_2"].append(ids_to_terminology[_in])
                to_db["value"].append(temporal_matrix[_out][_in])

        df_to_insert = pl.DataFrame(to_db).with_columns(pl.lit("temporal").alias("dimension"))

        insert_or_update_dimension(df_to_insert)

        mo.md("✅ Dimensão temporal salva.")
    else:
        mo.md("A dimensão temporal já existe no banco de dados.")
    return


@app.cell
def _(epsilon, neighbors, np):
    def stable_affinity_from_distance(a_distance_matrix, K=30, mu=0.5, eps=1e-8):
        """
        This function converts a distance matrix into an affinity (similarity) matrix
        using an adaptive Gaussian kernel, following the idea from Similarity Network Fusion (SNF).
        It prevents numerical instability when sigma (the adaptive scale) becomes very small,
        which can otherwise lead to overflow or NaN values.
        """
        distance_matrix = np.nan_to_num(
            a_distance_matrix, nan=0.0, posinf=np.nanmax(a_distance_matrix), neginf=0.0
        )

        # ensure perfect symmetry (averaging the matrix with its transpose)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2.0

        np.fill_diagonal(distance_matrix, 0.0)

        # normalize all distances to the range [0, 1]
        distance_matrix = distance_matrix / (np.nanmax(distance_matrix) + epsilon)

        # sort distances within each row to find K nearest neighbors
        sorted_distances = np.sort(distance_matrix, axis=1)

        # compute the average distance to the K nearest neighbors for each sample
        mean_neighbor_distance = np.vstack(
            sorted_distances[:, 1 : neighbors + 1].mean(axis=1) + epsilon
        )

        # compute adaptive local scale sigma according to SNF definition
        sigma = (mean_neighbor_distance + mean_neighbor_distance.T + distance_matrix) / 3.0

        # avoid extremely small sigma values (numerical underflow)
        sigma = np.clip(mu * sigma, epsilon, None)

        # compute Gaussian affinity kernel
        affinity_matrix = np.exp(-(distance_matrix**2) / (2.0 * sigma**2))
        affinity_matrix = np.nan_to_num(affinity_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(affinity_matrix, 0.0)

        return affinity_matrix

    return (stable_affinity_from_distance,)


@app.cell
def _(np, stable_affinity_from_distance, temporal_matrix):
    temporal_matrix_normalized = stable_affinity_from_distance(temporal_matrix, K=20, mu=0.5)
    print(np.nanmin(temporal_matrix_normalized), np.nanmax(temporal_matrix_normalized))

    return (temporal_matrix_normalized,)


@app.cell
def _(plt, temporal_matrix_normalized):
    plt.imshow(temporal_matrix_normalized[:100, :100], cmap="viridis")
    plt.colorbar()
    plt.title("Affinity matrix (temporal - DTW)")
    plt.show()

    return


@app.cell
def _(clinical_mat, plt):
    plt.imshow(clinical_mat[:100, :100], cmap="viridis")
    plt.colorbar()
    plt.title("Affinity matrix (temporal - DTW)")
    plt.show()

    return


@app.cell
def _(np, temporal_matrix_normalized):
    np.nanmin(temporal_matrix_normalized), np.nanmax(temporal_matrix_normalized)
    return


@app.cell
def _(clinical_mat, np):
    np.nanmin(clinical_mat), np.nanmax(clinical_mat)
    return


@app.cell
def _(mo):
    mo.md(r"""## Affinities""")
    return


@app.cell
def _():
    # hyperparameters
    K = 30
    return (K,)


@app.cell
def _(clinical_mat, temporal_matrix_normalized):
    # all views
    affinities = [clinical_mat, temporal_matrix_normalized]
    return (affinities,)


@app.cell
def _(K, affinities, compute, np):
    fused = compute.snf(affinities, K=K)  # 5 min
    print(np.isnan(fused).any())  # nan might occurr due to divisions to 0
    fused = np.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=0.0)
    return (fused,)


@app.cell
def _(mo):
    mo.md(r"""### Finding the perfect K""")
    return


@app.cell
def _():
    # TODO
    return


@app.cell
def _(mo):
    mo.md(r"""## Clustering 🩻""")
    return


@app.cell
def _(compute, fused):
    # estimate the number of clusters present in the fused matrix, derived via
    # an "eigengap" method (i.e., largest difference in eigenvalues of the
    # laplacian of the graph). note this function returns the top two options
    first, second = compute.get_n_clusters(
        fused, range(2, 50)
    )  # TODO establish a better range given the number of codes
    return first, second


@app.cell
def _(first, second):
    first, second
    return


@app.cell
def _(fused, fused_labels, metrics):
    silhouette = metrics.silhouette_score(fused, fused_labels)  # nan se tiver apenas uma amostra!
    silhouette
    return


@app.cell
def _(cluster, first, fused):
    fused_labels = cluster.spectral_clustering(fused, n_clusters=first)
    return (fused_labels,)


if __name__ == "__main__":
    app.run()
