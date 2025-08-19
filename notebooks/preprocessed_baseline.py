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
    from vir.features import AGE_GROUP, POPULATION_RANGE, sin_transformer, cos_transformer
    return (
        AGE_GROUP,
        ColumnTransformer,
        FunctionTransformer,
        OneHotEncoder,
        OrdinalEncoder,
        POPULATION_RANGE,
        PROCESSED_DATA_DIR,
        Pipeline,
        StandardScaler,
        mo,
        np,
        pl,
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
    result_lazy = df.join(
        codes_embeddings.lazy(),
        on=['code', 'code_type'],
        how='left',
        engine='streaming'
    )

    result_df = result_lazy.collect(engine='streaming')
    return (result_df,)


@app.cell
def _(result_df):
    result_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## ⚙️ Feature engineering

    * year: keep as it is
    * week: to capture the cycles of the weeks, we decompose it with sin/cos.
    * sex: one-hot
    * age_group: ordinal encoding
    * quantity: keep as it is
    * code_type: one-hot
    * code: one-hot
    * region: one-hot
    * population_range: ordinal encoding
    """
    )
    return


@app.cell(disabled=True)
def _(
    AGE_GROUP,
    ColumnTransformer,
    FunctionTransformer,
    IncrementalPCA,
    OneHotEncoder,
    OrdinalEncoder,
    POPULATION_RANGE,
    Pipeline,
    StandardScaler,
    df,
    np,
    result_df,
):
    numeric_features = ['quantity', 'week_sin', 'week_cos']
    ordinal_features = ['age_group', 'population_range']
    categorical_features = ['sex', 'region', 'year']
    embedding_features = ['embeddings']

    SEX = df["sex"].unique().to_list()
    REGIONS = df["region"].unique().to_list()
    YEARS = df["year"].unique().to_list()

    embedding_pipeline = Pipeline([
        ('to_f32', FunctionTransformer(lambda x: np.asarray(x.to_list(), dtype=np.float32), validate=False)),
        ('ipca', IncrementalPCA(n_components=64, batch_size=100_000))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('embedding', embedding_pipeline, embedding_features),
            ('numeric', StandardScaler(), numeric_features),
            ('ordinal', OrdinalEncoder(categories=[AGE_GROUP, POPULATION_RANGE], dtype=np.float32), ordinal_features),
            ('categorical', OneHotEncoder(categories=[SEX, REGIONS, YEARS], handle_unknown='ignore', sparse_output=False, dtype=np.float32), categorical_features),
        ],
        remainder='drop'
    )
    preprocessor.set_output(transform="polars")
    preprocessed_data = preprocessor.fit_transform(result_df)

    embedding_names = [f'embed_{i}' for i in range(4)]  # FIXME

    cat_names = preprocessor.named_transformers_['categorical'].get_feature_names_out(categorical_features)
    final_columns = embedding_names + numeric_features + ordinal_features + list(cat_names)

    assert len(final_columns) == preprocessed_data.shape[1]

    return final_columns, preprocessed_data


@app.cell
def _(final_columns, preprocessed_data):
    preprocessed_data.columns = final_columns
    preprocessed_data.head()
    return


@app.cell
def _(PROCESSED_DATA_DIR, preprocessed_data):
    preprocessed_data.write_parquet(PROCESSED_DATA_DIR / "processed_with_embeddings_aesop_raw_dados_2017_20250223_20municipios_embeddings.parquet")
    return


if __name__ == "__main__":
    app.run()
