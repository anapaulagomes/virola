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
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder, OneHotEncoder
    from skrub import Cleaner, TableReport

    from vir.config import PROCESSED_DATA_DIR
    from vir.features import age_group, population_range, sin_transformer, cos_transformer
    return (
        Cleaner,
        ColumnTransformer,
        OneHotEncoder,
        OrdinalEncoder,
        StandardScaler,
        TableReport,
        age_group,
        mo,
        np,
        pl,
        population_range,
    )


@app.cell
def _(mo):
    mo.md(r"""# 🌳 [AESOP] Pre-processing""")
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
def _(pl, week_sin):
    df = pl.read_parquet("data/interim/cleaned_aesop_raw_dados_2017_20250223_20municipios.parquet")
    df = df.with_columns(
        week_sin=pl.col('week').map_elements(lambda _week: week_sin.get(_week), return_dtype=pl.Float64),
        week_cos=pl.col('week').map_elements(lambda _week: week_sin.get(_week), return_dtype=pl.Float64)
    )
    df = df.drop(['facility_code_cnes', 'code_type', 'week'])
    df
    return (df,)


@app.cell
def _(Cleaner, TableReport, df):
    cleaned_df = Cleaner().fit_transform(df)
    TableReport(cleaned_df)
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
def _(df):
    # TODO icd, ciap, abp
    df.columns
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    age_group,
    df,
    population_range,
):
    numeric_features = ['quantity', 'week_sin', 'week_cos']
    ordinal_features = ['age_group', 'population_range']
    categorical_features = ['state', 'sex', 'region', 'code', 'year', 'city']

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', StandardScaler(), numeric_features),
            ('ordinal', OrdinalEncoder(categories=[age_group, population_range]), ordinal_features),
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ],
        remainder='passthrough' # Mantém outras colunas se houver (nenhuma neste caso) / MinMaxScaler()
    )
    preprocessor.set_output(transform="polars")
    preprocessed_data = preprocessor.fit_transform(df.sample(100_000))

    cat_names = preprocessor.named_transformers_['categorical'].get_feature_names_out(categorical_features)
    final_columns = numeric_features + ordinal_features + list(cat_names)

    return final_columns, preprocessed_data


@app.cell
def _(final_columns, preprocessed_data):
    preprocessed_data.columns = final_columns
    preprocessed_data.head()
    return


@app.cell
def _():
    # preprocessed_data.write_parquet(PROCESSED_DATA_DIR / "processed_aesop_raw_dados_2017_20250223_20municipios.parquet")
    return


if __name__ == "__main__":
    app.run()
