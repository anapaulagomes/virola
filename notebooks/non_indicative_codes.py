import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from vir.data_filters import filter_most_representative_rows
    from vir.notebook_widgets import create_file_dropdown, load_terminology_labels

    return (
        create_file_dropdown,
        filter_most_representative_rows,
        load_terminology_labels,
    )


@app.cell
def _():
    import polars as pl

    return (pl,)


@app.cell
def _(mo):
    mo.md(r"""
    # Non-indicative codes

    Retrieve the most representative codes for labelling by an expert and use them as a filtering option.
    """)
    return


@app.cell
def _(load_terminology_labels):
    terminology_labels = load_terminology_labels()
    return (terminology_labels,)


@app.cell
def _(create_file_dropdown):
    file_dropdown = create_file_dropdown(
        default_pattern="cleaned_base_aesop_raw_extracao_01042026_all_cities.parquet"
    )
    file_dropdown
    return (file_dropdown,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Data preview
    """)
    return


@app.cell
def _(file_dropdown, pl):
    df = pl.read_parquet(file_dropdown.value)
    df
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Previously labeled data

    If existing.
    """)
    return


@app.cell
def _(mo):
    indicative_starter_file = mo.ui.file(
        label="Indicative file (labeled by an expert)", filetypes=[".csv"], multiple=False
    )
    indicative_starter_file
    return (indicative_starter_file,)


@app.cell
def _(indicative_starter_file):
    if not indicative_starter_file.value:
        indicative_file = "data/indicative_codes_yes_no__2022_2024.csv"
    else:
        indicative_file = indicative_starter_file.value
    indicative_file
    return (indicative_file,)


@app.cell
def _(indicative_file, pl):
    df_labeled = (
        pl.read_csv(indicative_file)
        .with_columns(pl.col("indicative").fill_null("").str.strip_chars().str.to_lowercase())
        .filter(pl.col("indicative").eq("yes"))
    )
    df_labeled
    return (df_labeled,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 75% of the codes
    """)
    return


@app.cell
def _(df, df_labeled, filter_most_representative_rows, terminology_labels):
    most75_df = (
        filter_most_representative_rows(df, group_by=["code", "code_type"], threshold=0.75)
        .join(
            df_labeled.select(["code_type", "code", "indicative"]),
            how="left",
            on=["code_type", "code"],
        )
        .join(
            terminology_labels.select(["code_type", "code", "description"]),
            how="left",
            on=["code_type", "code"],
        )
    )
    most75_df
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 90% of the codes
    """)
    return


@app.cell
def _(df, df_labeled, filter_most_representative_rows, terminology_labels):
    most90_df = (
        filter_most_representative_rows(df, group_by=["code", "code_type"], threshold=0.90)
        .join(
            df_labeled.select(["code_type", "code", "indicative"]),
            how="left",
            on=["code_type", "code"],
        )
        .join(
            terminology_labels.select(["code_type", "code", "description"]),
            how="left",
            on=["code_type", "code"],
        )
    )
    most90_df
    return


if __name__ == "__main__":
    app.run()
