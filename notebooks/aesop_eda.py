import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import plotly.express as px
    from vir.viz import plot_null_vs_not_null, plot_categories, scatter_plot_year_per_week
    from vir.config import RAW_DATA_DIR, INTERIM_DATA_DIR
    return (
        RAW_DATA_DIR,
        mo,
        pl,
        plot_categories,
        px,
        scatter_plot_year_per_week,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # Unsupervised Learning for Surveillance Indicators

    TODO: purpose

    ## AESOP data

    TODO: source
    TODO: racional

    _a amostra está representada por 1 município por região do país e tamanho de população dentro das categorias (até 100mil hab, >110; <250mil hab; >250mil; ,500mil hab; >500mil hab) (os >500mil foram escolhidos, os outros sorteados. foram excluídos do universo do sorteio os municípios < 5.000 hab (são 1.324 municípios, ou 23,8% do total)_

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
    """
    )
    return


@app.cell
def _():
    ibge_city_map = {
        530010: {"city": "Distrito Federal", "region": "Centro-Oeste", "population_range": "> 500 mil"},
        520110: {"city": "Anápolis", "region": "Centro-Oeste", "population_range": "250 - 500 mil"},
        510792: {"city": "Sorriso", "region": "Centro-Oeste", "population_range": "100 - 250 mil"},
        510622: {"city": "Nova Mutum", "region": "Centro-Oeste", "population_range": "< 100 mil"},
        431490: {"city": "Porto Alegre", "region": "Sul", "population_range": "> 500 mil"},
        430920: {"city": "Gravataí", "region": "Sul", "population_range": "250 - 500 mil"},
        430230: {"city": "Bom Jesus", "region": "Sul", "population_range": "< 100 mil"},
        410765: {"city": "Fazenda Rio Grande", "region": "Sul", "population_range": "100 - 250 mil"},
        354980: {"city": "São José do Rio Preto", "region": "Sudeste", "population_range": "250 - 500 mil"},
        330340: {"city": "Nova Friburgo", "region": "Sudeste", "population_range": "100 - 250 mil"},
        311320: {"city": "Carandaí", "region": "Sudeste", "population_range": "< 100 mil"},
        310620: {"city": "Belo Horizonte", "region": "Sudeste", "population_range": "> 500 mil"},
        292740: {"city": "Salvador", "region": "Nordeste", "population_range": "> 500 mil"},
        291300: {"city": "Ibitiara", "region": "Nordeste", "population_range": "< 100 mil"},
        130260: {"city": "Manaus", "region": "Norte", "population_range": "> 500 mil"},
        120035: {"city": "Marechal Thaumaturgo", "region": "Norte", "population_range": "< 100 mil"},
        120040: {"city": "Rio Branco", "region": "Norte", "population_range": "250 - 500 mil"},
        150180: {"city": "Breves", "region": "Norte", "population_range": "100 - 250 mil"},
        230370: {"city": "Caucaia", "region": "Nordeste", "population_range": "250 - 500 mil"},
        241200: {"city": "São Gonçalo do Amarante", "region": "Nordeste", "population_range": "100 - 250 mil"},
    }

    population_range = [
        "< 100 mil",
        "100 - 250 mil",
        "250 - 500 mil",
        "> 500 mil",
    ]
    return ibge_city_map, population_range


@app.cell
def _(ibge_city_map, pl):
    ibge_df = pl.DataFrame([
        {"CO_MUNICIPIO_IBGE": str(k), **v} for k, v in ibge_city_map.items()
    ])
    return (ibge_df,)


@app.cell
def _(RAW_DATA_DIR, pl):
    raw_df = pl.read_parquet(RAW_DATA_DIR / "aesop_raw_dados_2017_20250223_20municipios.parquet")  # TODO encoding
    raw_df
    return (raw_df,)


@app.cell
def _(mo):
    mo.md(r"""### 🥩 Let's observe the original values first""")
    return


@app.cell
def _(raw_df):
    raw_df.describe()
    return


@app.cell
def _(pl, px, raw_df):
    _proportion = []
    for _col, _valid, _null in zip(raw_df.columns, raw_df.count(), raw_df.null_count()):
        _total = _valid[0] + _null[0]
        _not_null = _valid[0]/_total
        _nulls = _null[0]/_total
        _proportion.append({"column": _col, "status": "not-null", "proportion": _not_null})
        _proportion.append({"column": _col, "status": "null", "proportion": _nulls})
        print(f"{_col} not-nulls {_valid[0]} nulls {_null[0]}")

    _fig = px.bar(
        pl.DataFrame(_proportion),
        x="column",
        y="proportion",
        color="status",
        barmode="stack",
        title="Null vs non-null values per column"
    )
    _fig.update_layout(yaxis_tickformat=".0%")
    _fig
    return


@app.cell
def _(plot_categories, raw_df):
    plot_categories(raw_df, "SG_UF")
    return


@app.cell
def _(plot_categories, raw_df):
    plot_categories(raw_df, "SEXO")
    return


@app.cell
def _():
    age_group = [
        # None,
        "< 1 ano",
        "1 a 4 anos",
        "5 a 11 anos",
        "12 a 15 anos",
        "16 a 18 anos",
        "19 a 24 anos",
        "25 a 29 anos",
        "30 a 39 anos",
        "40 a 49 anos",
        "50 a 59 anos",
        "60 a 65 anos",
        "66 a 85 anos",
        "Maior que 85 anos",
    ]
    return (age_group,)


@app.cell
def _(age_group, plot_categories, raw_df):
    plot_categories(raw_df, "FX_ETARIA", age_group)
    return


@app.cell
def _(px, raw_df):
    px.histogram(raw_df["QT"].value_counts(), x="QT", y="count", log_y=True)
    return


@app.cell
def _(raw_df, scatter_plot_year_per_week):
    scatter_plot_year_per_week(raw_df, "ANO", "SEMANA", color_column="ANO", color_schema="inferno")
    return


@app.cell
def _(mo):
    mo.md(r"""## 🧹 Data cleaning""")
    return


@app.cell
def _(ibge_df, pl, raw_df):
    df = raw_df.with_columns(
        pl.col("CIDCIAP").str.extract(r"^([A-Z]+)\(", 1).alias("code_type"),
        pl.col("CIDCIAP").str.extract(r"\(([^)]+)\)", 1).alias("code"),
    )
    df = df.drop(["CIDCIAP", "Data_folder"])
    df = df.drop_nulls(subset=["ANO", "SEMANA", "FX_ETARIA"])
    df = df.join(ibge_df, on="CO_MUNICIPIO_IBGE", how="left")
    df = df.drop(["CO_MUNICIPIO_IBGE"])
    df = df.rename(mapping={
        "ANO": "year",
        "SEMANA": "week",
        "SG_UF": "state",
        "SEXO": "sex",
        "FX_ETARIA": "age_group",
        "QT": "quantity",
        "CO_CNES": "facility_code_cnes",
    })
    df
    return (df,)


@app.cell
def _(df, plot_categories):
    plot_categories(df, "code_type")
    return


@app.cell
def _(df, plot_categories, population_range):
    plot_categories(df, "population_range", population_range)
    return


@app.cell
def _(df, plot_categories):
    plot_categories(df, "region")
    return


@app.cell
def _(df, pl):
    df.filter(pl.col("code_type") == "AB")["code"].value_counts()
    return


@app.cell
def _():
    # df.write_parquet(INTERIM_DATA_DIR / "cleaned_aesop_raw_dados_2017_20250223_20municipios.parquet")
    return


if __name__ == "__main__":
    app.run()
