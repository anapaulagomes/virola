import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from vir.adapters import ADAPTERS
    from vir.cleaning.prepare import prepare_dataframe
    from vir.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
    from vir.helpers import extract_dataset_name
    from vir.notebook_widgets import (
        load_indicative_codes,
        load_terminology_labels,
    )

    return (
        ADAPTERS,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        extract_dataset_name,
        load_indicative_codes,
        load_terminology_labels,
        mo,
        pl,
        prepare_dataframe,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Data preparation pipeline

    Transforms a cleaned interim dataset into a prepared parquet file ready for view analysis.

    Steps applied:
    - Drop administrative columns
    - Aggregate quantities per (year, week, sex, age_group, code_type, code)
    - Normalise code_type values
    - Join with terminology labels
    - Filter non-indicative codes
    - Filter ICD-10 Z-chapter codes

    The result is saved to `data/processed/prepared_<suffix>.parquet`.
    Year filtering and temporal coverage are applied later at view time.
    """)
    return


@app.cell
def _(INTERIM_DATA_DIR, mo):
    if INTERIM_DATA_DIR.exists():
        available_cleaned_files = sorted(
            INTERIM_DATA_DIR.glob("cleaned_base*.parquet"),
            key=lambda path: path.name,
        )
    else:
        available_cleaned_files = []

    dataset_dropdown = mo.ui.dropdown(
        options={path.name: path for path in available_cleaned_files},
        label="Cleaned interim file",
        value=available_cleaned_files[0].name if available_cleaned_files else None,
    )
    run_button = mo.ui.run_button(label="Run preparation pipeline")

    mo.vstack([dataset_dropdown, run_button])
    return dataset_dropdown, run_button


@app.cell
def _(dataset_dropdown, mo, run_button):
    mo.stop(
        mo.app_meta().mode != "script" and not run_button.value,
        mo.md("Select a cleaned file above and click **Run preparation pipeline**"),
    )

    if mo.app_meta().mode == "script":
        cli_args = mo.cli_args()
        input_path_override = cli_args.get("dataset")
        adapter_name = cli_args.get("adapter", "datasus")
    else:
        input_path_override = None
        adapter_name = "datasus"

    selected_input_path = input_path_override or (
        str(dataset_dropdown.value) if dataset_dropdown.value else None
    )

    mo.stop(
        selected_input_path is None,
        mo.md("No dataset selected."),
    )
    return adapter_name, selected_input_path


@app.cell
def _(ADAPTERS, adapter_name):
    active_adapter = ADAPTERS[adapter_name]
    return (active_adapter,)


@app.cell
def _(
    active_adapter,
    load_indicative_codes,
    load_terminology_labels,
    mo,
    selected_input_path,
):
    mo.stop(selected_input_path is None)

    terminology_labels = load_terminology_labels()
    exclude_codes, _ = load_indicative_codes(path=active_adapter.indicative_codes_file)

    mo.md(f"""
    **Configuration**
    - Adapter: `{active_adapter.name}`
    - Input: `{selected_input_path}`
    - Terminology terms loaded: {len(terminology_labels):,}
    - Codes to exclude: {len(exclude_codes):,}
    """)
    return exclude_codes, terminology_labels


@app.cell
def _(
    PROCESSED_DATA_DIR,
    active_adapter,
    exclude_codes,
    extract_dataset_name,
    mo,
    pl,
    prepare_dataframe,
    selected_input_path,
    terminology_labels,
):
    mo.stop(selected_input_path is None)

    interim_df = pl.read_parquet(selected_input_path)
    keep_city = interim_df["city"].n_unique() > 1
    dataset_name = extract_dataset_name(selected_input_path)
    output_suffix = "_multiple_cities" if keep_city else ""
    output_path = PROCESSED_DATA_DIR / f"prepared_{dataset_name}{output_suffix}.parquet"

    prepared_df = prepare_dataframe(
        dataframe=interim_df,
        terminology_labels=terminology_labels,
        exclude_codes=exclude_codes,
        skip_patterns=active_adapter.skip_patterns,
        keep_city=keep_city,
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    prepared_df.write_parquet(output_path)

    mo.md(f"""
    **Preparation complete**

    - Input records: {len(interim_df):,}
    - Output records: {len(prepared_df):,}
    - Columns: {len(prepared_df.columns)}
    - Saved to: `{output_path}`
    """)
    return (prepared_df,)


@app.cell
def _(prepared_df):
    prepared_df.head(10)
    return


if __name__ == "__main__":
    app.run()
