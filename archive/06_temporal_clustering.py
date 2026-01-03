import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from vir.config import PROCESSED_DATA_DIR

    return PROCESSED_DATA_DIR, mo, pl


@app.cell
def _(PROCESSED_DATA_DIR, mo):
    file_browser = mo.ui.file_browser(initial_path=PROCESSED_DATA_DIR, multiple=False)
    file_browser
    return (file_browser,)


@app.cell
def _(file_browser, pl):
    df = pl.read_parquet(file_browser.path(index=0))
    df = df.with_columns(
        pl.concat_str(
            [
                pl.col("year").cast(pl.Utf8),
                pl.lit("-W"),
                pl.col("week").cast(pl.Utf8).str.zfill(2),
                pl.lit("-1"),  # Monday (ISO 8601)
            ]
        )
        .str.to_date("%G-W%V-%u")  # ISO 8601 week format
        .alias("date")
    )
    df.head()
    return (df,)


@app.cell
def _(df):
    df.columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Temporal Clustering

    Using [tslearn](https://tslearn.readthedocs.io).

    A timeseries is a two-dimensional numpy array with its first dimension corresponding to the time axis and the second one being the feature dimensionality.

    Expected format:

    * `n_series`: number of temporal series (one by diagnosis and its quantity)
    * `n_timesteps`: number of points in time (in our case, number of weeks: 53)
    * `n_features`: number of variables that are being measured for each point in time.


    Notes:

    - Each position in the array is the representation on time.

    """
    )
    return


@app.cell
def _():
    # start = df['date'].min()
    # end = df['date'].max()
    # start, end

    # # year: 2016 -> 2025
    # # week: 1 -> 53
    # timeline = []  # 53 * (2025 - 2016)
    # for year in range(2016, 2026):
    #     for week in range(1, 54):
    #         print(f"{year}-{week}")

    return


@app.cell
def _(df):
    # 1. Pivotar o DataFrame
    # O índice se torna a 'data', as colunas são os 'diagnosticos' e os valores a 'quantidade'
    df_pivoted = df.pivot(
        values="quantity",
        index="date",
        aggregate_function="sum",
        on=["code", "code_type"],  # [_col for _col in df.columns if _col.startswith("embed")]
    ).sort("date")

    # 2. Preencher valores nulos (datas onde um diagnóstico não tinha dados)
    df_filled = df_pivoted.fill_null(0)

    print(df_filled)
    return (df_filled,)


@app.cell
def _(df_filled):
    # 3. Converter para NumPy e ajustar o formato
    # Remove a coluna de data e transpõe para que cada linha seja um diagnóstico
    diagnosticos = df_filled.columns[1:]
    X_pivoted = df_filled.drop("date").to_numpy().T  # Transpor .T é crucial aqui

    # 4. Remodelar para o formato 3D
    X_pivoted_3d = X_pivoted.reshape(X_pivoted.shape[0], X_pivoted.shape[1], 1)

    print("Diagnósticos na ordem do array:", diagnosticos)
    print("Shape do array final:", X_pivoted_3d.shape)
    print("\nArray pronto para tslearn:")
    print(X_pivoted_3d)
    return (X_pivoted_3d,)


@app.cell
def _():
    return


@app.cell
def _(X_pivoted_3d):
    from tslearn.clustering import TimeSeriesKMeans

    X_final = X_pivoted_3d

    kmeans = TimeSeriesKMeans(n_clusters=3, metric="dtw", random_state=42)
    labels = kmeans.fit_predict(X_final)

    # diagnostico_labels = series_df.get_column('diagnostico').to_list()
    # for diagnostico, label in zip(diagnostico_labels, labels):
    #     print(f"Diagnóstico '{diagnostico}' foi alocado ao cluster {label}")

    return X_final, labels


@app.cell
def _(df, labels):
    diagnostico_labels = df.get_column("diagnostico").to_list()
    for diagnostico, label in zip(diagnostico_labels, labels):
        print(f"Diagnóstico '{diagnostico}' foi alocado ao cluster {label}")

    return


@app.cell
def _(X_final, labels):
    from tslearn.clustering import silhouette_score

    silhouette_score(X_final, labels, metric="dtw")
    return


if __name__ == "__main__":
    app.run()
