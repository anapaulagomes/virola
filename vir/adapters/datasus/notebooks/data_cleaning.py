import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl

    from vir.adapters.datasus.cleaning import clean_datasus_raw
    from vir.adapters.datasus.features import (
        CITY_NAMES,
        IBGE_CITY_MAP,
    )
    from vir.config import (
        INTERIM_DATA_DIR,
        RAW_DATA_DIR,
        RAW_DATA_FILE,
    )
    from vir.helpers import suffix_from

    return (
        CITY_NAMES,
        IBGE_CITY_MAP,
        INTERIM_DATA_DIR,
        Path,
        RAW_DATA_DIR,
        RAW_DATA_FILE,
        clean_datasus_raw,
        mo,
        pl,
        suffix_from,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Data cleaning pipeline

    This notebook handles systematic data cleaning with unique ID preservation.
    """)
    return


@app.cell
def _(CITY_NAMES, RAW_DATA_DIR, RAW_DATA_FILE, mo):
    if RAW_DATA_DIR.exists():
        available_parquet_folders = sorted(
            [
                path
                for path in RAW_DATA_DIR.iterdir()
                if path.is_dir() and path.suffix == ".parquet"
            ],
            key=lambda path: path.name,
        )
    else:
        available_parquet_folders = []

    if not available_parquet_folders:
        available_parquet_folders = [RAW_DATA_FILE.parent]

    parquet_folder_dropdown = mo.ui.dropdown(
        options={path.name: path for path in available_parquet_folders},
        value=RAW_DATA_FILE.parent.name,
        label="Raw data folder",
    )
    city_dropdown = mo.ui.dropdown(
        options=["All cities"] + sorted(CITY_NAMES),
        value="All cities",
        label="City",
    )
    force_reprocess_checkbox = mo.ui.checkbox(label="Force reprocess", value=False)
    run_button = mo.ui.run_button(label="Run cleaning pipeline")

    mo.vstack(
        [
            mo.hstack([parquet_folder_dropdown, city_dropdown, force_reprocess_checkbox]),
            run_button,
        ]
    )
    return (
        city_dropdown,
        force_reprocess_checkbox,
        parquet_folder_dropdown,
        run_button,
    )


@app.cell
def _(IBGE_CITY_MAP, pl):
    ibge_df = pl.DataFrame(
        [
            {"CO_MUNICIPIO_IBGE": str(ibge_code), **city_data}
            for ibge_code, city_data in IBGE_CITY_MAP.items()
        ]
    )
    ibge_df = ibge_df.drop(["state"])
    ibge_df
    return (ibge_df,)


@app.cell
def _(
    Path,
    RAW_DATA_FILE,
    city_dropdown,
    force_reprocess_checkbox,
    mo,
    parquet_folder_dropdown,
    run_button,
):
    mo.stop(
        mo.app_meta().mode != "script" and not run_button.value,
        mo.md("Configure settings above and click **Run cleaning pipeline**"),
    )

    if mo.app_meta().mode == "script":
        cli_args = mo.cli_args()
        selected_city = cli_args.get("city", "")
        selected_raw_folder = (
            Path(cli_args["input"]) if cli_args.get("input") else RAW_DATA_FILE.parent
        )
        output_dir_override = Path(cli_args["output"]) if cli_args.get("output") else None
        force_reprocess = "force" in cli_args
        raw_output_label = f"_{cli_args['label']}" if cli_args.get("label") else ""
    else:
        selected_city = "" if city_dropdown.value == "All cities" else city_dropdown.value
        selected_raw_folder = parquet_folder_dropdown.value
        output_dir_override = None
        force_reprocess = force_reprocess_checkbox.value
        raw_output_label = ""

    return (
        force_reprocess,
        output_dir_override,
        raw_output_label,
        selected_city,
        selected_raw_folder,
    )


@app.cell
def _(
    INTERIM_DATA_DIR,
    clean_datasus_raw,
    force_reprocess,
    mo,
    pl,
    selected_city,
    selected_raw_folder,
    suffix_from,
):
    folder_suffix = suffix_from(selected_raw_folder, break_on="", remove_extension=True)
    cleaned_all_cities_path = INTERIM_DATA_DIR / f"cleaned_{folder_suffix}_all_cities.parquet"

    if selected_city and cleaned_all_cities_path.exists() and not force_reprocess:
        df = pl.read_parquet(cleaned_all_cities_path)
        df = df.filter(pl.col("city").eq(selected_city))
        data_source_label = f"loaded from cache: `{cleaned_all_cities_path.name}`"
    else:
        raw_df = pl.read_parquet(selected_raw_folder / "*.parquet")
        raw_df = raw_df.with_row_index("unique_id", offset=0)

        df = clean_datasus_raw(raw_df)
        if selected_city:
            df = df.filter(pl.col("city").eq(selected_city))
        data_source_label = f"processed from raw: `{selected_raw_folder.name}`"

    mo.md(f"""
    **Dataset ready** — {data_source_label}

    - Total records: {len(df):,}
    - Columns: {len(df.columns)}
    """)
    return (cleaned_all_cities_path, data_source_label, df, folder_suffix)


@app.cell
def _(mo):
    mo.md(r"""
    ## Data quality
    """)
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
    ### Data Report

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
    mo.md(r"""
    ## Data preview
    """)
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
def _(df, mo):
    export_button = mo.ui.run_button(label="Export to parquet")
    mo.hstack([mo.md(f"**{len(df):,} records ready to export**"), export_button])
    return (export_button,)


@app.cell
def _(
    IBGE_CITY_MAP,
    INTERIM_DATA_DIR,
    cleaned_all_cities_path,
    df,
    export_button,
    folder_suffix,
    mo,
    output_dir_override,
    pl,
    raw_output_label,
    selected_city,
):
    should_export = mo.app_meta().mode == "script" or export_button.value
    if should_export:
        output_dir = output_dir_override if output_dir_override is not None else INTERIM_DATA_DIR
        if not selected_city:
            cities_over_100 = [
                city_data["city"]
                for city_data in IBGE_CITY_MAP.values()
                if city_data["population_range"] != "< 100 mil"
            ]
            output_path = output_dir / cleaned_all_cities_path.name
            df.filter(pl.col("city").is_in(cities_over_100)).write_parquet(output_path)
        else:
            output_path = (
                output_dir
                / f"cleaned_{folder_suffix}{raw_output_label}_{selected_city.lower().replace(' ', '_')}.parquet"
            )
            df.write_parquet(output_path)

        mo.md(f"""
        **Cleaned data saved successfully!** {selected_city or "all cities"}

        - **Location**: `{output_path}`
        - **Records**: {len(df):,}
        - **Columns**: {len(df.columns)}

        Note: The `city` column is preserved in this file for cross-city analysis.
        """)
    return


if __name__ == "__main__":
    app.run()
