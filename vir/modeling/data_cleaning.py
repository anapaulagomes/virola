import marimo

from vir.features import IBGE_CITY_MAP

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from vir.config import (
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        RAW_DATA_DIR,
        RAW_DATA_FILE,
    )
    from vir.features import IBGE_CITY_MAP
    from vir.helpers import suffix_from

    return (
        IBGE_CITY_MAP,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        RAW_DATA_DIR,
        RAW_DATA_FILE,
        mo,
        pl,
        suffix_from,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # Data cleaning pipeline

    This notebook handles systematic data cleaning with unique ID preservation.
    """
    )
    return


@app.cell
def _(IBGE_CITY_MAP, pl):
    ibge_df = pl.DataFrame([{"CO_MUNICIPIO_IBGE": str(k), **v} for k, v in IBGE_CITY_MAP.items()])
    return (ibge_df,)


@app.cell
def _(mo):
    mo.md(r"""## 📊 Raw data""")
    return


@app.cell
def _(RAW_DATA_DIR, mo, pl):
    raw_df = pl.read_parquet(RAW_DATA_DIR / "aesop_raw_dados_2017_20250223_20municipios.parquet")
    raw_df = raw_df.with_row_index("unique_id", offset=0)

    mo.md(f"""
    **Dataset loaded with unique ID tracking**

    - Total records: {len(raw_df):,}
    - Columns: {len(raw_df.columns)}
    """)
    return (raw_df,)


@app.cell
def _(mo):
    mo.md(r"""## 🔍 Data processing""")
    return


@app.cell
def _():
    original_to_reduced_age_group = {
        "< 1 ano": "< 1 ano",
        "1 a 4 anos": "1 a 15 anos",
        "5 a 11 anos": "1 a 15 anos",
        "12 a 15 anos": "1 a 15 anos",
        "16 a 18 anos": "16 a 59 anos",
        "19 a 24 anos": "16 a 59 anos",
        "25 a 29 anos": "16 a 59 anos",
        "30 a 39 anos": "16 a 59 anos",
        "40 a 49 anos": "16 a 59 anos",
        "50 a 59 anos": "16 a 59 anos",
        "60 a 65 anos": "Maior que 60 anos",
        "66 a 85 anos": "Maior que 60 anos",
        "Maior que 85 anos": "Maior que 60 anos",
    }
    code_types_mapping = {
        "CID": "cid10",
        "CIAP": "ciap2",
        "AB": "ab",
    }
    return code_types_mapping, original_to_reduced_age_group


@app.cell
def _(code_types_mapping, ibge_df, original_to_reduced_age_group, pl, raw_df):
    df = (
        raw_df.with_columns(
            pl.col("CIDCIAP").str.extract(r"^([A-Z]+)\(", 1).alias("code_type"),
            pl.col("CIDCIAP").str.extract(r"\(([^)]+)\)", 1).alias("code"),
        )
        .drop(["CIDCIAP", "Data_folder"])
        .drop_nulls(subset=["ANO", "SEMANA", "FX_ETARIA"])
    )
    df = df.with_columns(
        pl.when(pl.col("code_type").eq("AB"))
        .then(pl.col("code").str.replace(r"-\d+", ""))
        .otherwise(pl.col("code"))
    )
    df = df.with_columns(
        pl.col("code_type").map_elements(
            lambda code_type: code_types_mapping.get(code_type), return_dtype=pl.String
        )
    )
    df = df.with_columns(
        pl.col("FX_ETARIA").map_elements(
            lambda age_group: original_to_reduced_age_group.get(age_group), return_dtype=pl.String
        )
    )
    df = df.join(ibge_df, on="CO_MUNICIPIO_IBGE", how="left")
    df = df.drop(["CO_CNES", "CO_MUNICIPIO_IBGE"])
    df = df.rename(
        {
            "ANO": "year",
            "SEMANA": "week",
            "SG_UF": "state",
            "SEXO": "sex",
            "FX_ETARIA": "age_group",
            "QT": "quantity",
        }
    )
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _(mo):
    mo.md(r"""## ✅ Data quality""")
    return


@app.cell
def _(df, pl):
    total_records = df.height
    core_fields = ["unique_id", "year", "week", "state", "sex", "age_group", "quantity"]
    completeness_report = []
    for field in core_fields:
        null_count = df.filter(pl.col(field).is_null()).height
        completeness = (total_records - null_count) / total_records
        status = "✓ PASS" if completeness >= 0.95 else "✗ FAIL"
        completeness_report.append(f"   - {field}: {completeness:.1%} complete {status}")
    return completeness_report, total_records


@app.cell
def _(df, pl, total_records):
    valid_codes = df.filter(
        pl.col("code_type").is_not_null() & pl.col("code").is_not_null()
    ).height
    code_coverage = valid_codes / total_records

    code_distribution = []
    for row in df["code_type"].value_counts().iter_rows(named=True):
        if row["code_type"] is not None:
            code_distribution.append(f"     * {row['code_type']}: {row['count']:,}")
    return code_coverage, code_distribution, valid_codes


@app.cell
def _(df):
    regions = df["region"].n_unique()
    cities = df["city"].n_unique()
    year_range = f"{df['year'].min()} to {df['year'].max()}"
    week_range = f"{df['week'].min()} to {df['week'].max()}"
    return cities, regions, week_range, year_range


@app.cell
def _(
    cities,
    code_coverage,
    code_distribution,
    completeness_report,
    df,
    mo,
    regions,
    total_records,
    valid_codes,
    week_range,
    year_range,
):
    unique_ids = df["unique_id"].n_unique()
    mo.md(f"""
    ### Data Quality Report

    **1. Unique ID Integrity:**

    - Unique IDs: {unique_ids:,}
    - Total records: {total_records:,}
    - ID uniqueness: {"✓ PASS" if unique_ids == total_records else "✗ FAIL"}
    - ID range: {df["unique_id"].min()} to {df["unique_id"].max()}

    **2. Core Fields Completeness:**

    {chr(10).join(completeness_report)}

    **3. Medical Code Quality:**

    - Valid medical codes: {valid_codes:,} ({code_coverage:.1%})
    - Code type distribution:
    {chr(10).join(code_distribution)}

    **4. Temporal Coverage:**

    - Year range: {year_range}
    - Week range: {week_range}

    **5. Geographic Coverage:**

    - Regions: {regions}
    - Cities: {cities}

    **Quality Assessment**: {total_records:,} records processed with unique ID preservation.
    Code coverage: {code_coverage:.1%} | Geographic coverage: {regions} regions, {cities} cities
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""## 📊 Data preview""")
    return


@app.cell
def _(df):
    df.head(10)
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(PROCESSED_DATA_DIR, pl):
    codes_df = pl.read_csv(f"{PROCESSED_DATA_DIR}/cid10_ciap2_ab.csv")
    codes_df
    return (codes_df,)


@app.cell
def _(mo):
    mo.md(r"""### 75% of diagnosis in this dataset""")
    return


@app.cell
def _(codes_df, df, pl):
    count = df.group_by("code_type", "code").len().sort(by="len")
    count = count.join(codes_df, on=["code", "code_type"])
    quantile75 = count["len"].quantile(0.75)
    count.filter(pl.col("len").gt(quantile75))
    return


@app.cell
def _(codes_df, df, pl):
    _count = df.group_by(["code_type", "code"]).agg(pl.col("quantity").sum())
    _count = _count.join(codes_df, on=["code", "code_type"])
    _quantile75 = _count["quantity"].quantile(0.75)
    _count.filter(pl.col("quantity").gt(_quantile75))
    return


@app.cell
def _(df):
    df.group_by(["code_type", "code"]).len().describe()
    return


@app.cell
def _(mo):
    mo.md(r"""## 📋 Cleaning data""")
    return


@app.cell
def _(INTERIM_DATA_DIR, RAW_DATA_FILE, df, mo, pl, suffix_from):
    if mo.app_meta().mode == "script":
        args = mo.cli_args()

        city = args.get("city", "")
        label = args.get("label", "")
        if label:
            label = f"_{label}"

        # load data clean data
        suffix = suffix_from(RAW_DATA_FILE, break_on="", remove_extension=True)
        if not city:  # all data
            output_path = INTERIM_DATA_DIR / f"cleaned_{suffix}{label}_all_cities.parquet"
            cities_over_100 = [
                city["city"]
                for city in IBGE_CITY_MAP.values()
                if city["population_range"] != "< 100 mil"
            ]
            df.filter(pl.col("city").is_in(cities_over_100)).write_parquet(output_path)
        else:  # sample with a city
            output_path = (
                INTERIM_DATA_DIR
                / f"cleaned_{suffix}{label}_{city.lower().replace(' ', '_')}.parquet"
            )
            city_df = df.filter(pl.col("city").eq(city.title()))
            city_df.write_parquet(output_path)

        mo.md(f"""
        **Cleaned data saved successfully!** {city or "all cities"}

        - **Location**: `{output_path}`
        - **Records**: {len(df):,}
        - **Columns**: {len(df.columns)}
        """)
    return


if __name__ == "__main__":
    app.run()
