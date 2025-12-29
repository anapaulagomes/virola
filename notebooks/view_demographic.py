import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import plotly.express as px
    import polars as pl
    from dotenv import load_dotenv
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity

    return KMeans, cosine_similarity, load_dotenv, pl, px


@app.cell
def _():
    from vir.config import DATA_DIR, INTERIM_DATA_DIR
    from vir.db import (
        fetch_terminologies,
    )

    return DATA_DIR, INTERIM_DATA_DIR, fetch_terminologies


@app.cell
def _(load_dotenv):
    _ = load_dotenv()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Demographic dimension 🧑‍🍼

    How the diagnosis appears in each demographic group.

    * Features: sex, age group
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Preparation

    ### Fetch diagnosis terminology
    """)
    return


@app.cell
def _(fetch_terminologies):
    embeddings_model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    terminology_labels = fetch_terminologies(embeddings_model_name)
    return (terminology_labels,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Select codes that are indicative
    """)
    return


@app.cell
def _():
    indicative_codes_enabled = True
    return (indicative_codes_enabled,)


@app.cell
def _(DATA_DIR, indicative_codes_enabled, pl):
    include_codes = []
    if indicative_codes_enabled:
        indicative_codes = pl.read_csv(
            f"{DATA_DIR}/indicative_probabilities_sentence-transformers_all-MiniLM-L6-v2_codes.csv"
        )
        for _data in indicative_codes.iter_rows(named=True):
            if _data["indicative_probability"] > 0.50:
                include_codes.append(f"{_data['code']}:{_data['code_type']}")
    return (include_codes,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Prepare dataframe
    """)
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
def _(
    files_dropdown,
    include_codes,
    indicative_codes_enabled,
    pl,
    terminology_labels,
):
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
    if indicative_codes_enabled:
        df = df.filter(pl.col("terminology_label").is_in(include_codes))
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
    return (terminology_to_ids,)


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
def _(df_demographics):
    columns = df_demographics["sex_age"].unique().to_list()
    return (columns,)


@app.cell
def _(columns, pivoted_df_demographics):
    only_sex_age_group = pivoted_df_demographics.select(columns)
    only_sex_age_group
    return (only_sex_age_group,)


@app.cell
def _(cosine_similarity, pl, terminology_to_ids):
    def demographics_view(pivoted_df_demographics):
        # calculate similarity using cosine
        feature_cols = pivoted_df_demographics.columns[1:]
        features_matrix = pivoted_df_demographics.select(feature_cols).to_numpy()
        sim_matrix_np = cosine_similarity(features_matrix)

        # make sure they have the same order from the other views
        id_labels_str = [str(i) for i in terminology_to_ids.keys()]
        sim_df = pl.DataFrame(sim_matrix_np, schema=id_labels_str)
        #     .with_columns(
        #     terminology_id=pl.Series(list(terminology_to_ids.keys()))
        # ).select(["terminology_id"] + id_labels_str)  # order both rows and columns
        return sim_df

    return


@app.cell
def _():
    # demo_view = demographics_view(df)
    # demo_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Clustering
    """)
    return


@app.cell
def _(columns, only_sex_age_group, pl):
    only_sex_age_group_norm = (
        only_sex_age_group.with_columns(pl.sum_horizontal(columns).alias("total"))
        .with_columns((pl.col(c) / pl.col("total")).alias(c) for c in columns)
        .drop("total")
    )
    return (only_sex_age_group_norm,)


@app.cell
def _(KMeans, only_sex_age_group_norm):
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(only_sex_age_group_norm.to_numpy())
    return (labels,)


@app.cell
def _(labels):
    labels
    return


@app.cell
def _(labels, only_sex_age_group_norm, pl):
    df_result = only_sex_age_group_norm.with_columns(pl.Series(name="cluster", values=labels))
    df_plot = df_result.sort("cluster")
    return (df_plot,)


@app.cell
def _(df_plot):
    df_plot
    return


@app.cell
def _(df_plot, px):
    _fig = px.imshow(df_plot, aspect="equal")
    # plt.colorbar()
    # plt.title("Affinity matrix (temporal - DTW)")
    # _fig.update_layout(height=600, yaxis={"dtick": 1}) # Garante que mostre todas as linhas
    # plt.show()
    _fig
    return


@app.cell
def _():
    import dash_bio

    return (dash_bio,)


@app.cell
def _(columns, dash_bio, only_sex_age_group_norm):
    dash_bio.Clustergram(
        data=only_sex_age_group_norm.to_pandas(),
        column_labels=columns,
        # row_labels=list(df.index),
        height=800,
        width=700,
    )
    return


if __name__ == "__main__":
    app.run()
