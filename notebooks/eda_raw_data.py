import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import unicodedata

    import marimo as mo
    import numpy as np
    import plotly.express as px
    import polars as pl

    from vir.config import PROCESSED_DATA_DIR, RAW_DATA_FILE
    from vir.features import IBGE_CITY_MAP, ORIGINAL_AGE_GROUP, POPULATION_RANGE
    from vir.helpers import extract_dataset_name
    from vir.notebook_widgets import load_and_prepare_dataframe
    from vir.viz import plot_categories, scatter_plot_year_per_week

    return (
        IBGE_CITY_MAP,
        ORIGINAL_AGE_GROUP,
        POPULATION_RANGE,
        PROCESSED_DATA_DIR,
        RAW_DATA_FILE,
        extract_dataset_name,
        load_and_prepare_dataframe,
        mo,
        np,
        pl,
        plot_categories,
        px,
        scatter_plot_year_per_week,
        unicodedata,
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
    ## Code sparsity distributions by city

    Before choosing a filtering threshold, inspect how **non-zero weeks** and
    **weekly mean quantity** are distributed per city. If the distributions have
    similar shapes (even at different scales), a percentile-based threshold
    auto-calibrates across cities. If bimodal, two structurally different code
    populations exist and need separate treatment.

    | Metric | Definition |
    |--------|------------|
    | `nonzero_weeks` | Number of distinct (year, week) pairs with quantity > 0 |
    | `weekly_mean` | Mean quantity across non-zero weeks |
    """)
    return


@app.cell
def _(
    PROCESSED_DATA_DIR,
    extract_dataset_name,
    load_and_prepare_dataframe,
    pl,
    unicodedata,
):
    def _normalize_ascii(text):
        return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode()

    CITIES = {
        "salvador": "Nordeste",
        "belo_horizonte": "Sudeste",
        "gravatai": "Sul",
    }

    _all_stats = []
    for _city_key, _region in CITIES.items():
        _matches = [
            _f
            for _f in PROCESSED_DATA_DIR.glob("**/prepared_*.parquet")
            if _city_key in _normalize_ascii(str(_f.name).lower())
        ]
        if not _matches:
            continue
        _path = _matches[0]
        _city = extract_dataset_name(str(_path))
        _df = load_and_prepare_dataframe(file_path=str(_path))

        _code_stats = (
            _df.group_by("terminology_id")
            .agg(
                pl.col("quantity").filter(pl.col("quantity") > 0).count().alias("nonzero_weeks"),
                pl.col("quantity").filter(pl.col("quantity") > 0).mean().alias("weekly_mean"),
                pl.col("code_type").first().alias("code_type"),
            )
            .with_columns(
                pl.lit(_city).alias("city"),
                pl.lit(_region).alias("region"),
            )
        )
        _all_stats.append(_code_stats)

    sparsity_df = pl.concat(_all_stats) if _all_stats else pl.DataFrame()
    return (sparsity_df,)


@app.cell
def _(mo, np, pl, px, sparsity_df):
    mo.stop(sparsity_df.is_empty())

    _cities = sparsity_df["city"].unique(maintain_order=True).to_list()

    _nonzero_figs = []
    _mean_figs = []
    for _city in _cities:
        _city_df = sparsity_df.filter(pl.col("city") == _city)
        _nonzero = _city_df["nonzero_weeks"].to_numpy()
        _weekly = _city_df["weekly_mean"].drop_nulls().to_numpy()

        _p20_weeks = float(np.percentile(_nonzero, 20))
        _p20_mean = float(np.percentile(_weekly, 20)) if len(_weekly) > 0 else 0.0

        _fig_nz = px.histogram(
            _city_df,
            x="nonzero_weeks",
            nbins=50,
            title=f"{_city} — non-zero weeks (P20={_p20_weeks:.0f})",
            labels={"nonzero_weeks": "Non-zero weeks"},
            color_discrete_sequence=["#2E86AB"],
        )
        _fig_nz.add_vline(
            x=_p20_weeks,
            line_dash="dash",
            line_color="red",
            annotation_text=f"P20={_p20_weeks:.0f}",
        )
        _fig_nz.update_layout(height=300, template="plotly_white")
        _nonzero_figs.append(_fig_nz)

        _fig_wm = px.histogram(
            _city_df.filter(pl.col("weekly_mean").is_not_null()),
            x="weekly_mean",
            nbins=50,
            log_y=True,
            title=f"{_city} — weekly mean (P20={_p20_mean:.2f})",
            labels={"weekly_mean": "Mean weekly quantity"},
            color_discrete_sequence=["#F18F01"],
        )
        _fig_wm.add_vline(
            x=_p20_mean, line_dash="dash", line_color="red", annotation_text=f"P20={_p20_mean:.2f}"
        )
        _fig_wm.update_layout(height=300, template="plotly_white")
        _mean_figs.append(_fig_wm)

    mo.vstack([mo.hstack([_nz, _wm]) for _nz, _wm in zip(_nonzero_figs, _mean_figs)])
    return


@app.cell
def _(mo, np, pl, sparsity_df):
    mo.stop(sparsity_df.is_empty())

    _cities = sparsity_df["city"].unique(maintain_order=True).to_list()
    _rows = []
    for _city in _cities:
        _city_df = sparsity_df.filter(pl.col("city") == _city)
        _nonzero = _city_df["nonzero_weeks"].to_numpy()
        _weekly = _city_df["weekly_mean"].drop_nulls().to_numpy()

        _p20_weeks = float(np.percentile(_nonzero, 50))
        _p20_mean = float(np.percentile(_weekly, 50)) if len(_weekly) > 0 else 0.0
        _n_total = len(_city_df)
        _n_kept = int(
            _city_df.filter(
                (pl.col("nonzero_weeks") >= _p20_weeks) & (pl.col("weekly_mean") >= _p20_mean)
            ).height
        )

        _rows.append(
            {
                "city": _city,
                "n_codes": _n_total,
                "P20_nonzero_weeks": round(_p20_weeks, 0),
                "P20_weekly_mean": round(_p20_mean, 2),
                "kept": _n_kept,
                "dropped": _n_total - _n_kept,
                "kept_pct": round(_n_kept / _n_total * 100, 1),
            }
        )

    _summary = pl.DataFrame(_rows)

    mo.vstack(
        [
            mo.md(r"""
        ### P20 adaptive threshold summary

        Keep a code if `nonzero_weeks >= P20` **and** `weekly_mean >= P20` for
        that city. P20 is conservative — discards only the 20% sparsest codes,
        preserving the long tail that may carry real signal.
        """),
            _summary,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
