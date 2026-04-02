import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.express as px
    import polars as pl
    from scipy.stats import chi2_contingency

    from vir.config import RAW_DATA_FILE
    from vir.features import IBGE_CITY_MAP, ORIGINAL_AGE_GROUP, POPULATION_RANGE
    from vir.viz import plot_categories, scatter_plot_year_per_week

    return (
        IBGE_CITY_MAP,
        ORIGINAL_AGE_GROUP,
        POPULATION_RANGE,
        RAW_DATA_FILE,
        chi2_contingency,
        mo,
        np,
        pl,
        plot_categories,
        px,
        scatter_plot_year_per_week,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Exploratory data analysis - Primary care - Fiocruz/Brazil

    The data includes 20 municipalities sampled from different regions and population ranges in Brazil.

    | IBGE Code | City/Name               | State | Region         | Population Range  |
    |-----------|-------------------------|-------|----------------|-------------------|
    | 292740   | Salvador                | BA    | Nordeste       | > 500 mil         |
    | 431490   | Porto Alegre            | RS    | Sul            | > 500 mil         |
    | 130260   | Manaus                  | AM    | Norte          | > 500 mil         |
    | 310620   | Belo Horizonte          | MG    | Sudeste        | > 500 mil         |
    | 530010   | Distrito Federal        | DF    | Centro-Oeste   | > 500 mil         |
    | 230370    | Caucaia                 | CE    | Nordeste       | 250 - 500 mil     |
    | 430920    | Gravataí                | RS    | Sul            | 250 - 500 mil     |
    | 120040    | Rio Branco              | AC    | Norte          | 250 - 500 mil     |
    | 354980    | São José do Rio Preto   | SP    | Sudeste        | 250 - 500 mil     |
    | 520110    | Anápolis                | GO    | Centro-Oeste   | 250 - 500 mil     |
    | 241200    | São Gonçalo do Amarante | CE    | Nordeste       | 100 - 250 mil     |
    | 410765    | Fazenda Rio Grande      | PR    | Sul            | 100 - 250 mil     |
    | 150180    | Breves                  | PA    | Norte          | 100 - 250 mil     |
    | 330340    | Nova Friburgo           | RJ    | Sudeste        | 100 - 250 mil     |
    | 510792    | Sorriso                 | MT    | Centro-Oeste   | 100 - 250 mil     |
    | 291300    | Ibitiara                | BA    | Nordeste       | < 100 mil         |
    | 430230    | Bom Jesus               | RS    | Sul            | < 100 mil         |
    | 120035    | Marechal Thaumaturgo    | AC    | Norte          | < 100 mil         |
    | 311320    | Carandaí                | MG    | Sudeste        | < 100 mil         |
    | 510622    | Nova Mutum              | MT    | Centro-Oeste   | < 100 mil         |
    """)
    return


@app.cell
def _(IBGE_CITY_MAP, pl):
    ibge_df = pl.DataFrame([{"CO_MUNICIPIO_IBGE": str(k), **v} for k, v in IBGE_CITY_MAP.items()])
    return (ibge_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 📊 Raw data
    """)
    return


@app.cell
def _(RAW_DATA_FILE, mo, pl):
    raw_df = pl.read_parquet(RAW_DATA_FILE)
    raw_df = raw_df.with_row_index("unique_id", offset=0)
    raw_df = raw_df.rename(
        mapping={
            "ANO": "year",
            "SEMANA": "week",
            "SG_UF": "state",
            "SEXO": "sex",
            "FX_ETARIA": "age_group",
            "QT": "quantity",
        }
    )

    mo.md(f"""
    **Dataset loaded with unique ID tracking**
    - Total records: {len(raw_df):,}
    - Columns: {len(raw_df.columns)}
    """)
    return (raw_df,)


@app.cell
def _(raw_df):
    raw_df.columns
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 🔍 Data overview
    """)
    return


@app.cell
def _(raw_df):
    raw_df.head(10)
    return


@app.cell
def _(raw_df):
    raw_df.describe()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Missing values
    """)
    return


@app.cell
def _(pl, px, raw_df):
    _proportion = []
    for _col, _valid, _null in zip(raw_df.columns, raw_df.count(), raw_df.null_count()):
        _total = _valid[0] + _null[0]
        _not_null = _valid[0] / _total
        _nulls = _null[0] / _total
        _proportion.append({"column": _col, "status": "not-null", "proportion": _not_null})
        _proportion.append({"column": _col, "status": "null", "proportion": _nulls})
        print(f"{_col}: {_valid[0]:,} not-null, {_null[0]:,} null ({_nulls:.1%})")

    _fig = px.bar(
        pl.DataFrame(_proportion),
        x="column",
        y="proportion",
        color="status",
        barmode="stack",
        title="Completeness Analysis: Null vs Non-null values per column",
    )
    _fig.update_layout(yaxis_tickformat=".0%")
    _fig.update_xaxes(tickangle=45)
    _fig
    return


@app.cell
def _(ibge_df, mo, pl, raw_df):
    mo.md(r"""### Variables distribution""")
    df = (
        raw_df.with_columns(
            pl.col("CIDCIAP").str.extract(r"^([A-Z]+)\(", 1).alias("code_type"),
            pl.col("CIDCIAP").str.extract(r"\(([^)]+)\)", 1).alias("code"),
        )
        .drop(["CIDCIAP", "Data_folder"])
        .drop_nulls(subset=["year", "week", "age_group"])
    )
    df = df.with_columns(
        pl.when(pl.col("code_type").eq("AB"))
        .then(pl.col("code").str.replace(r"-\d+", ""))
        .otherwise(pl.col("code"))
    )
    df = df.join(ibge_df, on="CO_MUNICIPIO_IBGE", how="left")
    df = df.drop(["CO_CNES", "CO_MUNICIPIO_IBGE"])
    return (df,)


@app.cell
def _(df, plot_categories):
    plot_categories(df, "state", title="State cases distribution")
    return


@app.cell
def _(df, plot_categories):
    plot_categories(df, "sex", title="Sex distribution")
    return


@app.cell
def _(ORIGINAL_AGE_GROUP, df, plot_categories):
    plot_categories(df, "age_group", ORIGINAL_AGE_GROUP, title="Age group distribution")
    return


@app.cell
def _(df, px):
    px.histogram(
        df["quantity"].value_counts(),
        x="quantity",
        y="count",
        log_y=True,
        title="Distribution of quantity values (log scale)",
    )
    return


@app.cell
def _(df, scatter_plot_year_per_week):
    title = "Number of visits profile per week by year"
    scatter_plot_year_per_week(
        df, "year", "week", color_column="year", color_schema="inferno", title=title
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 🏥 Medical terminologies
    """)
    return


@app.cell
def _(df, pl, px):
    px.bar(
        df.group_by("code_type").agg(
            pl.col("code").n_unique().alias("unique_codes"), pl.len().alias("total_records")
        ),
        x="code_type",
        y="unique_codes",
        title="Number of unique codes per code type",
        text="unique_codes",
    )
    return


@app.cell
def _(df):
    df.group_by(["code_type", "code"]).len().sort("len")
    return


@app.cell
def _(df, pl):
    code_length_df = df.group_by(
        ["code_type", pl.col("code").str.len_chars().alias("code_length")]
    ).agg(pl.len().alias("count"))
    return (code_length_df,)


@app.cell
def _(code_length_df, px):
    px.histogram(
        code_length_df,
        x="code_type",
        y="count",
        color="code_length",
        barmode="group",
        title="Code length distribution by code type",
        height=400,
    )
    return


@app.cell
def _(df, px):
    _fig = px.bar(
        df.group_by(["code", "code_type"]).len().sort("len", descending=True).limit(50),
        x="code",
        y="len",
        color="code_type",
        title="Top 50 most frequent codes by type",
    )
    _fig.update_layout(barmode="stack", xaxis={"categoryorder": "total descending"})
    _fig.update_xaxes(tickangle=45)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 🩺 Diagnosis distribution (ICD-10)
    """)
    return


@app.cell
def _(df, pl, px):
    _icd_df = df.filter(pl.col("code_type") == "CID")
    _top_diagnoses = (
        _icd_df.group_by("code")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .limit(40)
    )
    _fig = px.bar(
        _top_diagnoses,
        x="code",
        y="count",
        title="Top 40 most frequent ICD-10 diagnoses",
        labels={"code": "ICD-10 Code", "count": "Number of records"},
    )
    _fig.update_layout(xaxis={"categoryorder": "total descending"})
    _fig.update_xaxes(tickangle=45)
    _fig


@app.cell
def _(df, pl, px):
    ICD10_CHAPTERS = {
        "A": "I – Infectious & parasitic diseases",
        "B": "I – Infectious & parasitic diseases",
        "C": "II – Neoplasms",
        "D": "II/III – Neoplasms / Blood diseases",
        "E": "IV – Endocrine & metabolic diseases",
        "F": "V – Mental disorders",
        "G": "VI – Nervous system",
        "H": "VII/VIII – Eye & Ear disorders",
        "I": "IX – Circulatory system",
        "J": "X – Respiratory system",
        "K": "XI – Digestive system",
        "L": "XII – Skin diseases",
        "M": "XIII – Musculoskeletal system",
        "N": "XIV – Genitourinary system",
        "O": "XV – Pregnancy & childbirth",
        "P": "XVI – Perinatal conditions",
        "Q": "XVII – Congenital malformations",
        "R": "XVIII – Symptoms & signs",
        "S": "XIX – Injury & poisoning",
        "T": "XIX – Injury & poisoning",
        "V": "XX – External causes",
        "W": "XX – External causes",
        "X": "XX – External causes",
        "Y": "XX – External causes",
        "Z": "XXI – Health status factors",
        "U": "XXII – Special purposes",
    }

    _chapter_map = pl.DataFrame(
        [{"first_char": k, "chapter": v} for k, v in ICD10_CHAPTERS.items()]
    )

    _icd_df = (
        df.filter(pl.col("code_type") == "CID")
        .with_columns(pl.col("code").str.slice(0, 1).alias("first_char"))
        .join(_chapter_map, on="first_char", how="left")
        .with_columns(pl.col("chapter").fill_null("Unknown"))
    )

    _chapter_counts = (
        _icd_df.group_by("chapter").agg(pl.len().alias("count")).sort("count", descending=True)
    )

    _fig = px.bar(
        _chapter_counts,
        x="chapter",
        y="count",
        title="ICD-10 records per chapter",
        labels={"chapter": "ICD-10 Chapter", "count": "Number of records"},
    )
    _fig.update_layout(xaxis={"categoryorder": "total descending"})
    _fig.update_xaxes(tickangle=30)
    _fig


@app.cell
def _(mo):
    mo.md(r"""
    ### 🌍 Geographic distribution
    """)
    return


@app.cell
def _(POPULATION_RANGE, df, plot_categories):
    plot_categories(df, "population_range", POPULATION_RANGE)
    return


@app.cell
def _(df, plot_categories):
    plot_categories(df, "region")
    return


@app.cell
def _(df, pl, px):
    _fig = px.treemap(
        df.group_by(["region", "city"]).agg(pl.len().alias("count")),
        path=["region", "city"],
        values="count",
        title="Record distribution by region and city",
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 🔗 Correlation analysis
    """)
    return


@app.cell
def _(chi2_contingency, np, pl):
    def cramers_v(df, col1, col2) -> float:
        try:
            _pivot = (
                df.group_by([col1, col2])
                .agg(pl.len().alias("count"))
                .pivot(values="count", index=col1, on=col2)
                .fill_null(0)
            )
            table = _pivot.select(pl.all().exclude(col1)).to_numpy()
            chi2, _, _, _ = chi2_contingency(table)
            n = table.sum()
            r, k = table.shape
            return np.sqrt(chi2 / (n * (min(k, r) - 1)))
        except:  # noqa E722
            return 0.0

    return (cramers_v,)


@app.cell
def _(cramers_v, df, pl):
    assoc_results = []
    for _index, col1 in enumerate(df.columns):
        for col2 in df.columns[_index + 1 :]:
            cv = cramers_v(df, col1, col2)
            assoc_results.append((col1, col2, cv))

    assoc_df = pl.DataFrame(assoc_results, schema=["var1", "var2", "cramers_v"], orient="row")
    assoc_df.sort("cramers_v", descending=True)
    return (assoc_df,)


@app.cell
def _(np):
    def df_to_heatmap_matrix(df, variables):
        number_of_variables = len(variables)
        matrix = np.zeros(shape=(number_of_variables, number_of_variables))
        index_variables = {variable: _index for _index, variable in enumerate(variables)}

        for _index in range(number_of_variables):
            matrix[_index][_index] = 1.0

        for _row in df.iter_rows(named=True):
            _var1_index = index_variables.get(_row["var1"])
            _var2_index = index_variables.get(_row["var2"])
            if _var1_index is not None and _var2_index is not None:
                matrix[_var1_index][_var2_index] = _row["cramers_v"]
                matrix[_var2_index][_var1_index] = _row["cramers_v"]
        return matrix, index_variables

    return (df_to_heatmap_matrix,)


@app.cell
def _(assoc_df, df, df_to_heatmap_matrix, px):
    matrix, index_variables = df_to_heatmap_matrix(assoc_df, df.columns)
    _fig = px.imshow(
        matrix,
        x=list(index_variables.keys()),
        y=list(index_variables.keys()),
        aspect="auto",
        origin="lower",
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Variables correlation (Cramér's V)",
    )
    _fig.update_layout(coloraxis_colorbar=dict(title="Cramér's V"))
    _fig.update_xaxes(tickangle=45)
    _fig
    return


if __name__ == "__main__":
    app.run()
