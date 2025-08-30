import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from vir.config import INTERIM_DATA_DIR, RAW_DATA_DIR
    from vir.features import IBGE_CITY_MAP
    return IBGE_CITY_MAP, INTERIM_DATA_DIR, RAW_DATA_DIR, mo, pl


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
def _(ibge_df, pl, raw_df):
    df = (
        raw_df.with_columns(
            pl.col("CIDCIAP").str.extract(r"^([A-Z]+)\(", 1).alias("code_type"),
            pl.col("CIDCIAP").str.extract(r"\(([^)]+)\)", 1).alias("code"),
        )
        .drop(["CIDCIAP", "Data_folder"])
        .drop_nulls(subset=["ANO", "SEMANA", "FX_ETARIA"])
    )
    df = df.with_columns(
        pl.when(pl.col("code_type").eq("AB")).then(pl.col("code").str.replace(r"-\d+", "")).otherwise(pl.col("code"))
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
def _(df):
    df.group_by(["code_type", "code"]).len()
    return


@app.cell
def _(mo):
    mo.md(r"""## 📋 Cleaning data""")
    return


@app.cell
def _(INTERIM_DATA_DIR, df, mo, pl):
    output_path = (
        INTERIM_DATA_DIR / "cleaned_aesop_raw_dados_2017_20250223_20municipios_082025.parquet"
    )
    df.write_parquet(output_path)

    # sample with Salvador
    output_path_city = (
        INTERIM_DATA_DIR
        / "cleaned_aesop_raw_dados_2017_20250223_20municipios_082025_salvador.parquet"
    )
    salvador_df = df.filter(pl.col("city").eq("Salvador"))
    salvador_df.write_parquet(output_path_city)

    mo.md(f"""
    **Cleaned data saved successfully!**

    - **Location**: `{output_path}`
    - **Records**: {len(df):,}
    - **Columns**: {len(df.columns)}

    By city: Salvador

    - **Location**: `{output_path_city}`
    - **Records**: {len(salvador_df):,}
    - **Columns**: {len(salvador_df.columns)}
    """)
    return


if __name__ == "__main__":
    app.run()
