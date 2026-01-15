import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    from vir.data_filters import filter_most_representative_rows
    from vir.notebook_widgets import load_terminology_labels

    return filter_most_representative_rows, load_terminology_labels


@app.cell
def _():
    import polars as pl

    return (pl,)


@app.cell
def _(load_terminology_labels):
    terminology_labels = load_terminology_labels()
    terminology_labels
    return (terminology_labels,)


@app.cell
def _(pl):
    df = pl.read_parquet(
        "data/interim/cleaned_aesop_raw_dados_2017_20250223_20municipios_all_cities.parquet"
    )
    df
    return (df,)


@app.cell
def _(pl):
    df_labeled = (
        pl.read_csv("data/indicative_codes_yes_no.csv")
        .with_columns(pl.col("indicative").fill_null("").str.strip_chars().str.to_lowercase())
        .filter(pl.col("indicative").eq("yes"))
    )
    df_labeled
    return (df_labeled,)


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


if __name__ == "__main__":
    app.run()
