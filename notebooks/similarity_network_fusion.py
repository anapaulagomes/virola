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
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    from aeon.distances import dtw_pairwise_distance
    from sklearn import cluster
    from snf import compute, metrics

    return cluster, compute, dtw_pairwise_distance, metrics, mo, np, pl


@app.cell
def _():
    from vir.db import fetch_terminologies, read_dimension

    return fetch_terminologies, read_dimension


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
    return (terminology_to_ids,)


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
def _(empty_matrix, pl, read_dimension, terminology_to_ids):
    def load_clinical_dimension():
        clinical_df = read_dimension("clinical")
        _ids = list(terminology_to_ids.keys())
        clinical_df = clinical_df.filter(
            pl.col("terminology_id_1").is_in(_ids) & pl.col("terminology_id_2").is_in(_ids)
        )

        i = clinical_df["terminology_id_1"].replace_strict(terminology_to_ids).to_numpy()
        j = clinical_df["terminology_id_2"].replace_strict(terminology_to_ids).to_numpy()
        v = clinical_df["value"].to_numpy()  # TODO check if scaling is needed here

        clinical_matrix = empty_matrix.copy()
        clinical_matrix[i, j] = v
        clinical_matrix[j, i] = v

        # np.fill_diagonal(clinical_matrix, 1)  # TODO check if 0 or 1
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
    ordered_year_week
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
def _(pl, temporal_pivot):
    # TODO refactor from here

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
def _(dtw_pairwise_distance, pl):
    def make_temporal_matrix(temporal_pivot: pl.DataFrame, window: int | None = None):
        labels = [str(_id) for _id in temporal_pivot["id"].to_list()]
        series = temporal_pivot.drop("id").to_numpy()
        print(type(series), series.shape)

        series = (series - series.min(axis=1, keepdims=True)) / (  # normalized
            series.max(axis=1, keepdims=True) - series.min(axis=1, keepdims=True) + 1e-9
        )
        print("Temporal matrix normalized...")
        series_list = [s for s in series]

        print("Let's calculate the distance using DTW...")
        dist_matrix = dtw_pairwise_distance(series_list, n_jobs=-1)

        print("Putting in the matrix...")
        df_dtw = pl.DataFrame(dist_matrix, schema=labels).with_columns(pl.Series("id", labels))
        return df_dtw

    return (make_temporal_matrix,)


@app.cell
def _(make_temporal_matrix, temporal_pivot):
    df_dtw = make_temporal_matrix(temporal_pivot, window=10)
    return (df_dtw,)


@app.cell
def _(df_dtw, np, pl):
    # normalized_df_dtw = (df_dtw[:, :-1] / np.nanmax(df_dtw[:, :-1])).to_numpy()  # escala para [0, 1]
    normalized_df_dtw = df_dtw.select(
        pl.all().exclude("id")
    ).to_numpy()  # (n x n) só as distâncias  # TODO check this
    normalized_df_dtw = normalized_df_dtw / np.nanmax(normalized_df_dtw)

    # Usando kernel RBF gaussiano (como sugerido no paper original do SNF)
    sigma = np.nanmedian(normalized_df_dtw[normalized_df_dtw > 0])  # ignora zeros da diagonal
    temporal_mat = np.exp(-(normalized_df_dtw**2) / (2 * sigma**2)).astype(np.float32)
    return (temporal_mat,)


@app.cell
def _(temporal_mat):
    temporal_mat
    return


@app.cell
def _():
    # hyperparameters
    K = 30
    return (K,)


@app.cell
def _():
    # if not dimension_exists("temporal"):
    #     mo.md("Salvando a dimensão temporal no banco de dados...")

    #     n = temporal_mat.shape[0]
    #     rows, cols = np.triu_indices(n, k=1)

    #     df_to_insert = pl.DataFrame({
    #         "terminology_id_1": np.array(clinical_ids)[rows],
    #         "terminology_id_2": np.array(clinical_ids)[cols],
    #         "value": temporal_mat[rows, cols],
    #     }).with_columns(
    #         pl.lit("temporal").alias("dimension")
    #     )

    #     insert_or_update_dimension(df_to_insert)

    #     mo.md("✅ Dimensão temporal salva.")
    # else:
    #     mo.md("A dimensão temporal já existe no banco de dados.")
    return


@app.cell
def _(mo):
    mo.md(r"""## Affinities""")
    return


@app.cell
def _(load_dimension):
    temp_mat, temp_ids = load_dimension("temporal")
    return (temp_mat,)


@app.cell
def _(clinical_mat):
    clinical_mat.shape
    return


@app.cell
def _(clinical_mat, temp_mat):
    # all views
    affinities = [clinical_mat, temp_mat]
    return (affinities,)


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
