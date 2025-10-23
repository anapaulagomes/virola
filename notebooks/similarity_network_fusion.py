import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
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
    import json
    import os
    import time
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from aeon.distances import dtw_pairwise_distance
    from dotenv import load_dotenv
    from sklearn import cluster
    from sklearn.metrics.pairwise import cosine_similarity
    from snf import compute, metrics

    return (
        Path,
        cluster,
        compute,
        cosine_similarity,
        dtw_pairwise_distance,
        json,
        load_dotenv,
        metrics,
        mo,
        np,
        os,
        pl,
        plt,
        time,
    )


@app.cell
def _():
    from vir.config import CLUSTERING_DIR, INTERIM_DATA_DIR
    from vir.db import (
        fetch_terminologies,
        read_dimension,
    )

    return (
        CLUSTERING_DIR,
        INTERIM_DATA_DIR,
        fetch_terminologies,
        read_dimension,
    )


@app.cell
def _(load_dotenv):
    _ = load_dotenv()
    return


@app.cell
def _(mo):
    mo.md(r"""## Data loading""")
    return


@app.cell
def _(fetch_terminologies):
    embeddings_model_name = "medicalai/ClinicalBERT"
    terminology_labels = fetch_terminologies(embeddings_model_name)
    return embeddings_model_name, terminology_labels


@app.cell
def _(terminology_labels):
    terminology_labels
    return


@app.cell
def _(INTERIM_DATA_DIR, mo):
    _all_options = [
        str(_file.resolve()) for _file in INTERIM_DATA_DIR.glob("**/cleaned_*.parquet")
    ]
    default = [
        _option
        for _option in _all_options
        if "cleaned_aesop_raw_dados_2017_20250223_20municipios_salvador.parquet" in _option
    ]

    files_dropdown = mo.ui.dropdown(
        options=_all_options,
        label="Choose dataset",
        searchable=True,
        value=default[0] if default else None,
    )
    files_dropdown
    return (files_dropdown,)


@app.cell
def _(files_dropdown, pl, terminology_labels):
    selected_filename = files_dropdown.value
    df = pl.read_parquet(selected_filename)
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
    # TODO note: the order of diagnosis code is determined by df["terminology_id"].unique() == terminology_to_ids.keys()
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
def _(clinical_mat):
    len(clinical_mat)
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
        .sort(["year", "week"])  # chronological order
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
def _(dtw_pairwise_distance, ordered_year_week, os, temporal_pivot):
    dtw_results = dtw_pairwise_distance(
        temporal_pivot.select(ordered_year_week).to_numpy(),
        n_jobs=int(os.getenv("NUMBER_OF_THREADS", -1)),
    )
    dtw_results
    return (dtw_results,)


@app.cell
def _():
    # dtw took 14 min with n jobs = -1
    # 25 min with 16 threads
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
def _():
    # FIXME add column city
    # if not dimension_exists("temporal"):
    #     mo.md("Salvando a dimensão temporal no banco de dados...")
    #
    #     to_db = {
    #         "terminology_id_1": [],
    #         "terminology_id_2": [],
    #         "value": [],
    #     }
    #     for _out in range(len([1, 2])):
    #         for _in in range(len([1, 2])):
    #             print(
    #                 ids_to_terminology[_out], ids_to_terminology[_in], temporal_matrix[_out][_in]
    #             )
    #             to_db["terminology_id_1"].append(ids_to_terminology[_out])
    #             to_db["terminology_id_2"].append(ids_to_terminology[_in])
    #             to_db["value"].append(temporal_matrix[_out][_in])
    #
    #     df_to_insert = pl.DataFrame(to_db).with_columns(pl.lit("temporal").alias("dimension"))
    #
    #     insert_or_update_dimension(df_to_insert)
    #
    #     mo.md("✅ Dimensão temporal salva.")
    # else:
    #     mo.md("A dimensão temporal já existe no banco de dados.")
    return


@app.cell
def _():
    return


@app.cell
def _(df, pl):
    df_demographics = (
        df.with_columns(
            (pl.col("sex").cast(pl.Utf8) + "-" + pl.col("age_group").cast(pl.Utf8)).alias(
                "sex_age"
            )
        )
        .group_by(["terminology_id", "sex", "age_group", "sex_age"])
        .agg(pl.col("quantity").sum().alias("total"))
    )
    df_demographics
    return (df_demographics,)


@app.cell
def _(df_demographics):
    pivoted_df_demographics = df_demographics.pivot(
        index="terminology_id",  # rows
        on="sex_age",  # new columns
        values="total",  # cell value
    ).fill_null(0)
    pivoted_df_demographics
    return (pivoted_df_demographics,)


@app.cell
def _(pivoted_df_demographics, cosine_similarity, pl):
    def demographics_view(df):
        ids = df["terminology_id"]
        feature_cols = df.columns[1:]
        features_matrix = df.select(feature_cols).to_numpy()
        sim_matrix_np = cosine_similarity(features_matrix)
        id_labels = ids.to_list()
        id_labels_str = [str(i) for i in id_labels]
        sim_df = pl.DataFrame(sim_matrix_np, schema=id_labels_str)
        sim_df = sim_df.with_columns(terminology_id=ids).select(["terminology_id"] + id_labels_str)
        return sim_df

    demo_view = demographics_view(pivoted_df_demographics)
    return (demo_view,)


@app.cell
def _(np):
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
        distance_matrix = distance_matrix / (np.nanmax(distance_matrix) + eps)

        # sort distances within each row to find K nearest neighbors
        sorted_distances = np.sort(distance_matrix, axis=1)

        # compute the average distance to the K nearest neighbors for each sample
        mean_neighbor_distance = np.vstack(sorted_distances[:, 1 : K + 1].mean(axis=1) + eps)

        # compute adaptive local scale sigma according to SNF definition
        sigma = (mean_neighbor_distance + mean_neighbor_distance.T + distance_matrix) / 3.0

        # avoid extremely small sigma values (numerical underflow)
        sigma = np.clip(mu * sigma, eps, None)

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
def _(clinical_mat, temporal_matrix_normalized, demo_view):
    # all views
    affinities = [clinical_mat, temporal_matrix_normalized, demo_view]
    return (affinities,)


@app.cell
def _(mo):
    mo.md(r"""### Finding the perfect K""")
    run_k_validation = False
    return (run_k_validation,)


@app.cell
def _(cluster, compute, metrics, np, time):
    def evaluate_snf_k_range(affinities, K_values, t=20, n_range=(2, 80)):
        results = []

        for K in K_values:
            start_time = time.time()
            print(f"\n Running SNF with K={K}...")
            fused = compute.snf(affinities, K=K, t=t)
            fused = np.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=0.0)

            n_opt, n_alt = compute.get_n_clusters(fused, range(*n_range))
            print(f"  Estimated number of clusters: {n_opt} (second best: {n_alt})")

            fused_labels = cluster.spectral_clustering(fused, n_clusters=n_opt)

            silhouette = metrics.silhouette_score(fused, fused_labels)
            nmi = metrics.nmi([fused_labels])

            end_time = time.time()
            elapsed = end_time - start_time

            print(f"  Done in {elapsed / 60:.2f} min | Silhouette = {silhouette:.4f}")

            results.append(
                {
                    "K": K,
                    "clusters": n_opt,
                    "silhouette": float(silhouette),
                    "nmi": float(nmi),
                }
            )
        return results

    return (evaluate_snf_k_range,)


@app.cell
def _(affinities, evaluate_snf_k_range, run_k_validation):
    if run_k_validation:
        K_values = [20, 50, 75, 100, 150]
        results = evaluate_snf_k_range(affinities, K_values, t=20, n_range=(2, 100))
    else:
        results = []
    return (results,)


@app.cell
def _(plt, results, run_k_validation):
    if run_k_validation:
        number_of_neighbors = []
        silhouettes = []
        clusters = []

        for result in results:
            number_of_neighbors.append(result["K"])
            silhouettes.append(result["silhouette"])
            clusters.append(result["clusters"])

        _fig, _ax1 = plt.subplots(figsize=(8, 4))
        _ax1.plot(number_of_neighbors, silhouettes, "-o", color="tab:blue")
        _ax1.set_xlabel("K (number of neighbors)")
        _ax1.set_ylabel("Silhouette", color="tab:blue")
        _ax1.tick_params(axis="y", labelcolor="tab:blue")

        _ax2 = _ax1.twinx()
        _ax2.plot(number_of_neighbors, clusters, "--s", color="tab:orange")
        _ax2.set_ylabel("# Clusters (eigengap)", color="tab:orange")
        _ax2.tick_params(axis="y", labelcolor="tab:orange")

        plt.title("K evaluation in SNF")
        plt.tight_layout()
        plt.show()
    return


@app.cell
def _():
    # hyperparameters chosen based on K validation step
    K = 150
    number_of_clusters = 20
    return K, number_of_clusters


@app.cell
def _(K, affinities, compute, np):
    fused = compute.snf(affinities, K=K)  # 5 min
    print(np.isnan(fused).any())  # nan might occurr due to divisions to 0
    fused = np.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=0.0)
    # np.savetxt(f'fused_snf_{selected_filename}.txt', fused)
    return (fused,)


@app.cell
def _(mo):
    mo.md(r"""## Clustering 🩻""")
    return


@app.cell
def _(cluster, fused, number_of_clusters):
    fused_labels = cluster.spectral_clustering(fused, n_clusters=number_of_clusters)
    return (fused_labels,)


@app.cell
def _(fused, fused_labels, metrics):
    silhouette = metrics.silhouette_score(fused, fused_labels)  # nan se tiver apenas uma amostra!
    silhouette
    return


@app.cell
def _(
    CLUSTERING_DIR,
    K,
    Path,
    embeddings_model_name,
    fused_labels,
    ids_to_terminology,
    json,
    number_of_clusters,
):
    # notes
    # terminology_to_ids: terminology ids from db to index db used during execution
    # ids_to_terminology: index to terminology ids
    # fused labels: list with assigned clusters
    terminology_id_clusters = []
    for _index, _cluster in enumerate(fused_labels):
        _terminology_id = ids_to_terminology.get(_index)
        terminology_id_clusters.append(
            {"terminology_id": _terminology_id, "cluster": str(_cluster)}
        )

    _terminology_id_clusters_file = Path(
        CLUSTERING_DIR
        / f"snf_k{K}_{number_of_clusters}_clusters__{embeddings_model_name.replace('/', '_')}.txt"
    )
    _terminology_id_clusters_file.write_text(json.dumps(terminology_id_clusters))
    return


if __name__ == "__main__":
    app.run()
