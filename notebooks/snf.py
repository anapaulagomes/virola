import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import duckdb
    import marimo as mo
    import numpy as np
    import polars as pl
    import snf

    return duckdb, mo, np, pl, snf


@app.cell
def _(duckdb):
    dc = duckdb.connect("virola.db")
    return (dc,)


@app.cell
def _(dc, medical_terminologies, mo):
    med_df = mo.sql(
        """
        SELECT * FROM medical_terminologies LIMIT 100;
        """,
        engine=dc,
    )
    return (med_df,)


@app.cell
def _(dc, mo):
    _df = mo.sql(
        """
        DESCRIBE medical_terminologies;
        """,
        engine=dc,
    )
    return


@app.cell
def _(dc, mo):
    _df = mo.sql(
        """
        SELECT * FROM read_parquet('data/interim/cleaned_aesop_raw_dados_2017_20250223_20municipios_salvador.parquet');
        """,
        engine=dc,
    )
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
    mo.md(r"""## Clinical dimension""")
    return


@app.cell
def _(dc, mo):
    _df = mo.sql(
        """
        -- CREATE OR REPLACE TABLE dimensions AS
        -- SELECT
        --   t1.code || ':' || t1.code_type AS code_id_1,
        --   t2.code || ':' || t2.code_type AS code_id_2,
        --   array_cosine_similarity(t1.embedding::float[768], t2.embedding::float[768]) AS value,
        --   'clinical' AS dimension
        -- FROM medical_terminologies t1
        -- JOIN medical_terminologies t2 ON TRUE;

        -- 6 min

        -- CREATE OR REPLACE TABLE dimensions (
        --     code_id_1 BIGINT NOT NULL REFERENCES medical_terminologies(id),
        --     code_id_2 BIGINT NOT NULL REFERENCES medical_terminologies(id),
        --     similarity FLOAT4 NOT NULL,
        --     PRIMARY KEY (terminology_id_1, terminology_id_2)
        -- );
        """,
        engine=dc,
    )
    return


@app.cell
def _(dc, dimensions, mo):
    _df = mo.sql(
        """
        SELECT * FROM dimensions LIMIT 10;
        """,
        engine=dc,
    )
    return


@app.cell
def _(dc, dimensions, mo):
    _df = mo.sql(
        """
        select min(value), max(value) from dimensions;
        """,
        engine=dc,
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
def _(dc, pl):
    clinical_df = pl.read_database(
        """
        SELECT code_id_1, code_id_2, value
        FROM dimensions
        WHERE dimension = 'clinical'
        """,
        connection=dc,
    )
    return (clinical_df,)


@app.cell
def _(np, pl):
    # Exemplo: converter edge list para matriz densa
    def edge_to_dense(df: pl.DataFrame, n: int) -> np.ndarray:
        mat = np.zeros((n, n), dtype=np.float32)
        i = df["code_id_1"].to_numpy()
        j = df["code_id_2"].to_numpy()
        v = df["value"].to_numpy()
        mat[i, j] = v
        mat[j, i] = v  # simetrizar
        return mat

    return (edge_to_dense,)


@app.cell
def _(med_df):
    n = med_df["code"].len()
    return (n,)


@app.cell
def _(clinical_df, edge_to_dense, n):
    clinical_mat = edge_to_dense(clinical_df, n)
    return (clinical_mat,)


@app.cell
def _(clinical_mat):
    # suponha que você tenha mais vistas
    W_list = [clinical_mat]  # , genomic_mat, imaging_mat
    return (W_list,)


@app.cell
def _(W_list, snf):
    # ---- SNF ----
    W_fused = snf.snf(W_list, K=30, t=20)
    return (W_fused,)


@app.cell
def _(W_fused, snf):
    # clustering depois da fusão
    labels = snf.clustering.spectral(W_fused, K=30, n_clusters=5)
    return


@app.cell
def _():
    # # Pivot para formato wide (NxN)
    # sim_matrix_pl = clinical_df.pivot(
    #     index="code_uid_1",
    #     columns="code_uid_2",
    #     values="similarity"
    # )

    # # Converter para numpy
    # labels = sim_matrix_pl["code_uid_1"].to_list()
    # sim_matrix = sim_matrix_pl.drop("code_uid_1").to_numpy()

    # # Garantir simetria (caso faltem pares)
    # sim_matrix = np.nan_to_num(sim_matrix)
    # sim_matrix = (sim_matrix + sim_matrix.T) / 2
    return


@app.cell
def _(row_normalize_csr, symmetrize_and_renorm, topk_per_row_csr):
    clinical_csr = row_normalize_csr(clinical_csr)
    S_clinical = symmetrize_and_renorm(topk_per_row_csr(clinical_csr, K=30))

    # Depois: usar snf_update([clinical_csr, outras], [S_clinical, outras_S], T=20)
    return (clinical_csr,)


if __name__ == "__main__":
    app.run()
