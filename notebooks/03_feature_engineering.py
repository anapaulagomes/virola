import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import IncrementalPCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        FunctionTransformer,
        OneHotEncoder,
        OrdinalEncoder,
        StandardScaler,
    )

    from vir.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
    from vir.features import (
        AGE_GROUP,
        POPULATION_RANGE,
    )

    return (
        AGE_GROUP,
        ColumnTransformer,
        FunctionTransformer,
        INTERIM_DATA_DIR,
        IncrementalPCA,
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
    mo.md(r"""# ⚙️ [AESOP] Feature Engineering Pipeline""")
    return


@app.cell
def _(mo):
    mo.md(r"""## 🔄 Temporal features""")
    return


@app.cell
def _(mo, np):
    _period = 53  # weeks in year
    week_cos = {_week: np.cos(_week / _period * 2 * np.pi) for _week in range(_period + 1)}
    week_sin = {_week: np.sin(_week / _period * 2 * np.pi) for _week in range(_period + 1)}

    sample_values = []
    for week in [1, 13, 26, 39, 52]:
        sample_values.append(f"Week {week}: sin={week_sin[week]:.3f}, cos={week_cos[week]:.3f}")

    mo.md(f"""
    **Cyclical encoding created for weeks 0-53**

    Sample values:
    {chr(10).join(f"- {val}" for val in sample_values)}
    """)
    return week_cos, week_sin


@app.cell
def _(mo):
    mo.md(r"""## 📥 Loading data""")
    return


@app.cell
def _(INTERIM_DATA_DIR, mo, pl, week_cos, week_sin):
    # from_city = "cleaned_aesop_raw_dados_2017_20250223_20municipios_082025_salvador.parquet"
    # full_dataset = "cleaned_aesop_raw_dados_2017_20250223_20municipios.parquet"
    dataset_filename = "cleaned_aesop_raw_dados_2017_20250223_20municipios_082025_salvador.parquet"
    df = pl.read_parquet(INTERIM_DATA_DIR / dataset_filename)

    df = df.with_columns(
        pl.col("code").cast(pl.Categorical),
        pl.col("code_type").cast(pl.Categorical),
        week_sin=pl.col("week").map_elements(
            lambda w: week_sin.get(w, 0.0), return_dtype=pl.Float64
        ),
        week_cos=pl.col("week").map_elements(
            lambda w: week_cos.get(w, 0.0), return_dtype=pl.Float64
        ),
    )

    mo.md(f"""
    **Added cyclical week features:**

    - week_sin: min={df["week_sin"].min():.3f}, max={df["week_sin"].max():.3f}
    - week_cos: min={df["week_cos"].min():.3f}, max={df["week_cos"].max():.3f}
    """)
    return dataset_filename, df


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(mo):
    mo.md(r"""## 🧬 Medical code embeddings""")
    return


@app.cell
def _(PROCESSED_DATA_DIR, mo, pl):
    _code_types = {
        "cid10": "CID",
        "ciap2": "CIAP",
        "ab": "AB",
    }

    codes_embeddings = (
        pl.read_parquet(
            f"{PROCESSED_DATA_DIR}/bra_medter_cambridgeltl__SapBERT-from-PubMedBERT-fulltext.parquet"
        )
        .with_columns(
            code_type=pl.col("code_type").map_elements(
                lambda _df_code_type: _code_types.get(_df_code_type), return_dtype=pl.String
            )
        )
        .with_columns(
            pl.col("code").cast(pl.Categorical), pl.col("code_type").cast(pl.Categorical)
        )
    )

    mo.md(f"""
    **Medical code embeddings loaded:** {len(codes_embeddings):,} entries
    """)
    return (codes_embeddings,)


@app.cell
def _(codes_embeddings, df):
    result_df = df.join(codes_embeddings, on=["code", "code_type"], how="left")
    return (result_df,)


@app.cell
def _(mo):
    mo.md(r"""## 🧹 Feature selection""")
    return


@app.cell
def _(result_df):
    processed_df = result_df.drop(["week", "state", "city"])
    return (processed_df,)


@app.cell
def _(processed_df):
    processed_df.head()
    return


@app.cell
def _(mo):
    mo.md(r"""## 🏗️ Preprocessing pipeline""")
    return


@app.cell
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
    mo,
    np,
    processed_df,
):
    numeric_features = ["quantity", "week_sin", "week_cos", "year"]
    ordinal_features = ["age_group", "population_range"]
    categorical_features = ["sex", "region"]
    embedding_features = ["embeddings"]

    SEX = processed_df["sex"].unique().to_list()
    REGIONS = processed_df["region"].unique().to_list()

    embedding_pipeline = Pipeline(
        [
            (
                "to_f32",
                FunctionTransformer(
                    lambda x: np.asarray([list(arr) for arr in x], dtype=np.float32),
                    validate=False,
                ),
            ),
            ("ipca", IncrementalPCA(n_components=64, batch_size=10_000)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("embedding", embedding_pipeline, embedding_features),
            ("numeric", StandardScaler(), numeric_features),
            (
                "ordinal",
                OrdinalEncoder(categories=[AGE_GROUP, POPULATION_RANGE], dtype=np.float32),
                ordinal_features,
            ),
            (
                "categorical",
                OneHotEncoder(
                    categories=[SEX, REGIONS],
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32,
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    mo.md(f"""
    **Feature groups defined:**

    - Numeric: {numeric_features}
    - Ordinal: {ordinal_features}
    - Categorical: {categorical_features}
    - Embeddings: {len(embedding_features[0])}

    **Categories:**

    - SEX: {SEX}
    - REGIONS: {REGIONS}
    - AGE_GROUP: {AGE_GROUP}
    - POPULATION_RANGE: {POPULATION_RANGE}
    """)
    return categorical_features, preprocessor


@app.cell
def _(pl, processed_df):
    processed_df.filter(pl.col("embeddings").is_not_null())
    return


@app.cell
def _(categorical_features, mo, pl, preprocessor, processed_df):
    embedding_subset = processed_df.filter(pl.col("embeddings").is_not_null())

    preprocessor.set_output(transform="polars")
    preprocessed_data = preprocessor.fit_transform(embedding_subset)

    embedding_names = [f"embed_{i}" for i in range(64)]
    numeric_names = ["quantity", "week_sin", "week_cos", "year"]
    ordinal_names = ["age_group", "population_range"]
    cat_names = preprocessor.named_transformers_["categorical"].get_feature_names_out(
        categorical_features
    )
    final_columns = embedding_names + numeric_names + ordinal_names + list(cat_names)

    preprocessed_data.columns = final_columns

    processed_unique_ids = embedding_subset.select("unique_id")
    final_df = pl.concat([processed_unique_ids, preprocessed_data], how="horizontal")

    mo.md(f"""
    **Preprocessing applied**

    - Input records: {len(embedding_subset):,}
    - Output features: {len(final_columns)}
    - Feature categories: {len(embedding_names)} embeddings + {len(numeric_names)} numeric + {len(ordinal_names)} ordinal + {len(cat_names)} categorical
    - **Unique ID preserved**: ✓ Range {final_df["unique_id"].min()} to {final_df["unique_id"].max()}
    """)
    return (final_df,)


@app.cell
def _(final_df):
    final_df.head()
    return


@app.cell
def _(final_df, mo):
    mo.md(f"**Final dataset shape**: {final_df.shape}")
    final_df.describe()
    return


@app.cell
def _(PROCESSED_DATA_DIR, dataset_filename, final_df, mo):
    output_path = PROCESSED_DATA_DIR / f"processed_with_embeddings_{dataset_filename}"
    final_df.write_parquet(output_path)

    mo.md(f"""
    **Processed data:**: `{output_path}`

    - **Records**: {len(final_df):,}
    - **Features**: {len(final_df.columns)}
    """)
    return


if __name__ == "__main__":
    app.run()
