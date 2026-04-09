import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import plotly.express as px
    import polars as pl

    from vir.config import PROCESSED_DATA_DIR, RAW_DATA_FILE
    from vir.data_filters import filter_min_temporal_coverage, filter_temporal_quantiles
    from vir.features import IBGE_CITY_MAP, ORIGINAL_AGE_GROUP, POPULATION_RANGE
    from vir.viz import plot_categories, scatter_plot_year_per_week

    return (
        IBGE_CITY_MAP,
        ORIGINAL_AGE_GROUP,
        POPULATION_RANGE,
        PROCESSED_DATA_DIR,
        RAW_DATA_FILE,
        filter_min_temporal_coverage,
        filter_temporal_quantiles,
        mo,
        pl,
        plot_categories,
        px,
        scatter_plot_year_per_week,
    )


@app.cell
def _(IBGE_CITY_MAP):
    cities_md_table = [
        "| IBGE Code | City/Name               | State | Region         | Population Range  |",
        "|-----------|-------------------------|-------|----------------|-------------------|",
    ]
    for _code, _data in IBGE_CITY_MAP.items():
        _row = f"| {_code} | {_data['city']} | {_data['state']} | {_data['region']} | {_data['population_range']} |"
        cities_md_table.append(_row)

    cities_md_table = "\n".join(cities_md_table)
    return (cities_md_table,)


@app.cell
def _(mo):
    mo.md(r"""
    # Exploratory data analysis - Primary care - Fiocruz/Brazil

    The data includes 20 municipalities sampled from different regions and population ranges in Brazil.
    """)
    return


@app.cell
def _(cities_md_table, mo):
    mo.md(cities_md_table)
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
    return


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
    return


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
    ## Unique codes per code type
    """)
    return


@app.cell
def _(df):
    df.group_by(["city", "code_type"]).n_unique()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Code sparsity distributions by city

    Aggregation below is built directly from the **raw data**, with small multiples by city:
    """)
    return


@app.cell
def _(df):
    df.shape
    return


@app.cell
def _():
    return


@app.cell
def _(PROCESSED_DATA_DIR, pl):
    processed_df = pl.read_parquet(
        PROCESSED_DATA_DIR
        / "prepared_base_aesop_raw_extracao_01042026_all_cities_multiple_cities.parquet"
    )
    processed_df.shape
    return (processed_df,)


@app.cell
def _(processed_df):
    processed_df.columns
    return


@app.cell
def _(pl, px):
    def codes_per_week(_df):
        codes_per_city = _df.group_by("city").agg(pl.col("code").n_unique().alias("total_codes"))

        present_codes_per_week = (
            _df.filter(pl.col("quantity") > 0)
            .group_by(["city", "year", "week"])
            .agg(pl.col("code").n_unique().alias("present_codes"))
        )

        weekly_zero_count = (
            present_codes_per_week.join(codes_per_city, on="city", how="left")
            .with_columns(
                (pl.col("total_codes") - pl.col("present_codes")).alias("zero_code_count")
            )
            .sort(by=["year", "city"])
        )
        _fig = px.histogram(
            weekly_zero_count,
            facet_row="city",
            facet_col="year",
            x="week",
            y="present_codes",
            labels={"present_codes": "diagnosis codes"},
        )
        for _annotation in _fig.layout.annotations:
            _annotation.text = _annotation.text.replace("year=", "").replace("city=", "")
            if _annotation.textangle is None or _annotation.textangle == 0:
                pass
            else:
                _annotation.textangle = 0
        _fig.for_each_yaxis(lambda axis: axis.update(title_text=""))
        _fig.update_layout(yaxis_title="diagnosis codes")
        return _fig

    return (codes_per_week,)


@app.cell
def _(codes_per_week, processed_df):
    codes_per_week(processed_df)
    return


@app.cell
def _(mo):
    by_year_checkbox = mo.ui.checkbox(label="Apply filter by year", value=True)
    weeks_slider = mo.ui.slider(start=12, stop=24, step=2, value=12, label="Min weeks per year")
    mo.hstack([by_year_checkbox, weeks_slider])
    return by_year_checkbox, weeks_slider


@app.cell
def _(
    by_year_checkbox,
    codes_per_week,
    df,
    filter_min_temporal_coverage,
    mo,
    processed_df,
    weeks_slider,
):
    _df = filter_min_temporal_coverage(processed_df, min_weeks=weeks_slider.value, by_year=True)

    _chart = codes_per_week(_df)
    _stat = mo.stat(
        label="Total",
        bordered=True,
        caption=f"min per week {'by year' if by_year_checkbox.value else 'global'}",
        direction="increase" if _df.shape[0] > df.shape[0] else "decrease",
        value=_df.shape[0],  # "{len(filtered_movies):,.0f}",
    )

    mo.vstack([_stat, _chart])
    return


@app.cell
def _(mo):
    quantile_slider = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.5, label="Quantile threshold"
    )
    quantile_slider
    return (quantile_slider,)


@app.cell
def _(
    codes_per_week,
    df,
    filter_temporal_quantiles,
    mo,
    processed_df,
    quantile_slider,
):
    _df = filter_temporal_quantiles(processed_df, quantile_slider.value)

    _chart = codes_per_week(_df)
    _stat = mo.stat(
        label="Total",
        bordered=True,
        caption=f"P{quantile_slider.value:.0%}",
        direction="increase" if _df.shape[0] > df.shape[0] else "decrease",
        value=_df.shape[0],
    )

    mo.vstack([_stat, _chart])
    return


if __name__ == "__main__":
    app.run()
