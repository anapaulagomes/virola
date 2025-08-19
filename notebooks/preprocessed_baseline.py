import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import cross_validate
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder, OneHotEncoder, FunctionTransformer

    from vir.config import PROCESSED_DATA_DIR
    from vir.features import age_group, population_range, sin_transformer, cos_transformer
    return (
        ColumnTransformer,
        FunctionTransformer,
        OneHotEncoder,
        OrdinalEncoder,
        PROCESSED_DATA_DIR,
        Pipeline,
        StandardScaler,
        age_group,
        mo,
        np,
        pl,
        population_range,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # 🌳 [AESOP] Pre-processing

    First, we discard what we won't use. Then, we transform the columns to a proper ML-friendly format.
    """
    )
    return


@app.cell
def _(np):
    _period = 53  # weeks
    week_cos = {
        _week: np.cos(_week / _period * 2 * np.pi)
        for _week in range(_period + 1)
    }
    week_sin = {
        _week: np.sin(_week / _period * 2 * np.pi)
        for _week in range(_period + 1)
    }
    return (week_sin,)


@app.cell
def _(PROCESSED_DATA_DIR, pl):
    _code_types = {
        "cid10": "CID",
        "ciap2": "CIAP",
        "ab": "AB",
    }
    codes_embeddings = pl.read_parquet(
        f"{PROCESSED_DATA_DIR}/code_type_embeddings_pucpr__clinicalnerpt-medical.parquet"
    ).with_columns(
        code_type=pl.col("code_type").map_elements(
            lambda _df_code_type: _code_types.get(_df_code_type), return_dtype=pl.String
        )
    ).with_columns(
        pl.col("code").cast(pl.Categorical),
        pl.col("code_type").cast(pl.Categorical)
    )
    return (codes_embeddings,)


@app.cell
def _():
    return


@app.cell
def _(pl, week_sin):
    df = pl.scan_parquet(
        "data/interim/cleaned_aesop_raw_dados_2017_20250223_20municipios.parquet"
    ).with_columns(
        pl.col("code").cast(pl.Categorical),
        pl.col("code_type").cast(pl.Categorical),
        week_sin=pl.col('week').map_elements(lambda _week: week_sin.get(_week), return_dtype=pl.Float64),
        week_cos=pl.col('week').map_elements(lambda _week: week_sin.get(_week), return_dtype=pl.Float64)
    ).drop(['facility_code_cnes', 'week', 'state', 'city'])
    return (df,)


@app.cell
def _():
    return


@app.cell
def _(codes_embeddings, df):
    # df_embeddings = df.join(codes_embeddings, how="left", on=["code", "code_type"])
    # df_embeddings = df.clone()

    # for row in df.group_by(["code", "code_type"]).all().iter_rows(named=True):
    #     df_embeddings = df_embeddings.with_columns(
    #         code_embedding=codes_embeddings.filter(
    #             pl.col("code_type").eq(row["code_type"]), pl.col("code").eq(row["code"])
    #         )["embeddings"].first()
    #     )
    #     break

    result_lazy = df.join(
        codes_embeddings.lazy(), # Make the small one lazy too for consistency
        on=['code', 'code_type'],
        how='left'
    )

    # Execute in streaming mode
    result_df = result_lazy.collect(streaming=True)
    return


@app.cell
def _():
    # df_embeddings.write_parquet("data/with_embeddings.parquet")

    # Use scan_* so nothing loads into memory yet
    # small = (
    #     pl.scan_parquet("code_embeddings.parquet")      # or scan_csv(...)
    #     .with_columns(
    #         pl.col(["code", "code_type"]).cast(pl.Categorical),   # shrink/join-friendly
    #         # store vectors as Float32 to halve memory
    #         pl.col("embeddings").arr.eval(pl.element().cast(pl.Float32))
    #     )
    # )

    # big = (
    #     pl.scan_parquet("big_43M.parquet")
    #     .with_columns(pl.col(["code", "code_type"]).cast(pl.Categorical))
    #     .select(["code", "code_type", "other_needed_cols"])       # project only what you need
    # )

    # # Streamed join; never collect — write directly to disk
    # (
    #     big.join(small, on=["code", "code_type"], how="left", streaming=True)
    #        .sink_parquet("joined.parquet")  # or sink_ipc("joined.arrow")
    # )

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## ⚙️ Feature engineering

    * year: keep as it is
    * week: to capture the cycles of the weeks, we decompose it with sin/cos.
    * state: one-hot
    * sex: one-hot
    * age_group: ordinal encoding
    * quantity: keep as it is
    * facility_code_cnes: _to be discarded because there is no use for it by now_
    * code_type: one-hot
    * code: one-hot
    * city: one-hot
    * region: one-hot
    * population_range: ordinal encoding
    """
    )
    return


@app.cell
def _(FunctionTransformer, Pipeline, np):
    emb_pipe = Pipeline(steps=[
        ("to_2d", FunctionTransformer(lambda x: np.vstack(x), validate=False)),
        # Opcional: normalizar/escala dependendo do modelo
        # ("scale", StandardScaler())            # bom p/ modelos lineares
        # ("l2norm", Normalizer(norm="l2"))      # bom p/ cosine/NearestNeighbors
    ])
    return (emb_pipe,)


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    age_group,
    df,
    emb_pipe,
    population_range,
):
    numeric_features = ['quantity', 'week_sin', 'week_cos']
    ordinal_features = ['age_group', 'population_range']
    categorical_features = ['sex', 'region', 'year']
    embedding_features = ['code']

    preprocessor = ColumnTransformer(
        transformers=[
            ("emb", emb_pipe, 'code'),
            ('numeric', StandardScaler(), numeric_features),
            ('ordinal', OrdinalEncoder(categories=[age_group, population_range]), ordinal_features),
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ],
        remainder='passthrough' # Mantém outras colunas se houver (nenhuma neste caso) / MinMaxScaler()
    )
    preprocessor.set_output(transform="polars")
    preprocessed_data = preprocessor.fit_transform(df.sample(100_000))  # FIXME

    cat_names = preprocessor.named_transformers_['categorical'].get_feature_names_out(categorical_features)
    final_columns = ['code'] + numeric_features + ordinal_features + list(cat_names)

    return final_columns, preprocessed_data


@app.cell
def _(final_columns, preprocessed_data):
    preprocessed_data.columns = final_columns
    preprocessed_data.head()
    return


@app.cell
def _(PROCESSED_DATA_DIR, preprocessed_data):
    preprocessed_data.write_parquet(PROCESSED_DATA_DIR / "processed_aesop_raw_dados_2017_20250223_20municipios_embeddings.parquet")
    return


@app.cell
def _(df):
    df["code"]
    return


if __name__ == "__main__":
    app.run()
