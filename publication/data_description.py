import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import pathlib
    from collections import defaultdict

    import marimo as mo
    import plotly.graph_objects as go
    import polars as pl
    from plotly.subplots import make_subplots

    from publication.commons import POPULATION_RANGE_TRANSLATIONS, REGION_TRANSLATIONS
    from vir.adapters.datasus import INDICATIVE_CODES_FILE
    from vir.adapters.datasus.features import CODE_TYPES_MAPPING, IBGE_CITY_MAP
    from vir.adapters.datasus.helpers import load_all_cities
    from vir.config import RAW_DATA_FILE
    from vir.viz import TABLEAU_COLORS

    return (
        CODE_TYPES_MAPPING,
        IBGE_CITY_MAP,
        INDICATIVE_CODES_FILE,
        POPULATION_RANGE_TRANSLATIONS,
        RAW_DATA_FILE,
        REGION_TRANSLATIONS,
        TABLEAU_COLORS,
        defaultdict,
        go,
        json,
        load_all_cities,
        make_subplots,
        mo,
        pathlib,
        pl,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Multiview Clustering

    ## Methods

    ### Dataset

    Geographic distribution of the 10 cities included in the study, across Brazil's five macro-regions.
    Circle color encodes the macro-region; circle size encodes the population bracket.
    """)
    return


@app.cell
def _(load_all_cities, pl):
    year_min = 2022
    year_max = 2024
    all_cities_df = load_all_cities(from_="cleaned").filter(
        pl.col("year").is_between(year_min, year_max)
    )
    return all_cities_df, year_max, year_min


@app.cell
def _(
    IBGE_CITY_MAP,
    POPULATION_RANGE_TRANSLATIONS,
    REGION_TRANSLATIONS,
    TABLEAU_COLORS,
    pl,
):
    REGION_COLORS = dict(zip(REGION_TRANSLATIONS.values(), TABLEAU_COLORS[:5]))
    POPULATION_SIZES = {"250K-500K": 25, "> 500K": 40}

    cities = pl.DataFrame(
        [
            {
                "city": data["city"],
                "region": REGION_TRANSLATIONS[data["region"]],
                "population_range": POPULATION_RANGE_TRANSLATIONS[data["population_range"]],
                "lat": data["lat"],
                "lon": data["lon"],
            }
            for data in IBGE_CITY_MAP.values()
        ]
    )
    return POPULATION_SIZES, REGION_COLORS, cities


@app.cell
def _(defaultdict, json, pathlib):
    _STATE_TO_REGION = {
        "AC": "North",
        "AP": "North",
        "AM": "North",
        "PA": "North",
        "RO": "North",
        "RR": "North",
        "TO": "North",
        "AL": "Northeast",
        "BA": "Northeast",
        "CE": "Northeast",
        "MA": "Northeast",
        "PB": "Northeast",
        "PE": "Northeast",
        "PI": "Northeast",
        "RN": "Northeast",
        "SE": "Northeast",
        "ES": "Southeast",
        "MG": "Southeast",
        "RJ": "Southeast",
        "SP": "Southeast",
        "PR": "South",
        "RS": "South",
        "SC": "South",
        "DF": "Central-West",
        "GO": "Central-West",
        "MS": "Central-West",
        "MT": "Central-West",
    }

    with open(pathlib.Path("publication/brazil-states.geojson")) as _file:
        _geojson = json.load(_file)

    state_lons, state_lats = [], []
    _edge_contributors = defaultdict(list)

    for _feature in _geojson["features"]:
        _state_code = _feature["properties"]["sigla"]
        _region = _STATE_TO_REGION[_state_code]
        _geometry = _feature["geometry"]
        _rings = (
            _geometry["coordinates"]
            if _geometry["type"] == "Polygon"
            else [ring for poly in _geometry["coordinates"] for ring in poly]
        )
        for _ring in _rings:
            for _coordinate in _ring:
                state_lons.append(_coordinate[0])
                state_lats.append(_coordinate[1])
            state_lons.append(None)
            state_lats.append(None)

            for _index in range(len(_ring) - 1):
                _edge_start, _edge_end = tuple(_ring[_index]), tuple(_ring[_index + 1])
                _edge = (min(_edge_start, _edge_end), max(_edge_start, _edge_end))
                _edge_contributors[_edge].append((_state_code, _region))

    # Region boundary = coastline (1 contributor) OR cross-region border (2 contributors, different regions)
    region_lons, region_lats = [], []
    for _edge, _contributors in _edge_contributors.items():
        if len(_contributors) == 1 or (
            len(_contributors) == 2 and _contributors[0][1] != _contributors[1][1]
        ):
            region_lons.extend([_edge[0][0], _edge[1][0], None])
            region_lats.extend([_edge[0][1], _edge[1][1], None])
    return region_lats, region_lons, state_lats, state_lons


@app.cell
def _(
    POPULATION_SIZES,
    REGION_COLORS,
    cities,
    go,
    mo,
    region_lats,
    region_lons,
    state_lats,
    state_lons,
):
    figure = go.Figure()

    # state outlines
    figure.add_trace(
        go.Scattergeo(
            lon=state_lons,
            lat=state_lats,
            mode="lines",
            line=dict(color="#CCCCCC", width=0.4),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # region boundaries
    figure.add_trace(
        go.Scattergeo(
            lon=region_lons,
            lat=region_lats,
            mode="lines",
            line=dict(color="#3A3A3A", width=1.8),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # city markers, one trace per region for the color legend
    for _region in REGION_COLORS:
        _region_cities = cities.filter(cities["region"] == _region)
        if len(_region_cities) == 0:
            continue
        figure.add_trace(
            go.Scattergeo(
                lon=_region_cities["lon"].to_list(),
                lat=_region_cities["lat"].to_list(),
                mode="markers",
                name=_region,
                legendgroup="region",
                legendgrouptitle_text="Region name",
                marker=dict(
                    color=REGION_COLORS[_region],
                    size=[
                        POPULATION_SIZES[p] for p in _region_cities["population_range"].to_list()
                    ],
                    line=dict(color="white", width=0.8),
                    opacity=0.92,
                ),
                customdata=_region_cities.select(["city", "population_range"]).to_numpy(),
                hovertemplate=f"<b>%{{customdata[0]}}</b><br>Region: {_region}<br>Population: %{{customdata[1]}}<extra></extra>",
            )
        )

    # population size legend
    _legend_population_sizes = {"250K-500K": 12, "> 500K": 22}
    for _population_label, _size in _legend_population_sizes.items():
        figure.add_trace(
            go.Scattergeo(
                lon=[None],
                lat=[None],
                mode="markers",
                name=_population_label,
                legendgroup="population",
                legendgrouptitle_text="Population range",
                marker=dict(color="#888888", size=_size, line=dict(color="white", width=0.8)),
            )
        )

    figure.update_geos(
        showland=False,
        showocean=False,
        showcoastlines=False,
        showframe=False,
        showlakes=False,
        showrivers=False,
        showsubunits=False,
        bgcolor="white",
        lataxis_range=[-45, 12],
        lonaxis_range=[-75, -27],
        projection_type="mercator",
    )

    figure.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=0, r=0, t=0, b=0),
        width=820,
        height=880,
        legend=dict(
            bgcolor="white",
            bordercolor="#DDDDDD",
            borderwidth=1,
            font=dict(size=18),
            grouptitlefont=dict(size=15),
            itemsizing="trace",
            tracegroupgap=15,
            x=0.1,
            y=0.2,
            xanchor="left",
            yanchor="bottom",
        ),
    )

    mo.ui.plotly(figure)
    return (figure,)


@app.cell
def _(figure):
    figure.write_image("dataset_map.pdf")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""

    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Unique codes

    Relevant vs irrelevant
    """)
    return


@app.cell
def _(INDICATIVE_CODES_FILE, all_cities_df, go, mo, pl):
    _record_counts = all_cities_df.group_by(["code", "code_type"]).len("quantity")

    _indicative_lookup_freq = pl.read_csv(INDICATIVE_CODES_FILE).select(
        ["code", "code_type", "indicative"]
    )

    _freq_counts = (
        _record_counts.join(_indicative_lookup_freq, on=["code", "code_type"], how="left")
        .with_columns(pl.col("indicative").fill_null("unlabeled"))
        .with_columns(pl.col("code_type").str.to_uppercase().alias("code_type_label"))
        .group_by(["code_type_label", "indicative"])
        .agg(pl.col("quantity").sum())
        .sort(["code_type_label", "indicative"], descending=[False, True])
    )

    _fig_freq = go.Figure()

    _indicative_color_map = [
        ("yes", "Relevant", "#525252"),
        ("no", "Irrelevant", "#969696"),
        ("unlabeled", "Unlabeled", "#CCCCCC"),
    ]

    _code_types = sorted(_freq_counts["code_type_label"].unique().to_list())

    for _indicative_status, _label, _color in _indicative_color_map:
        _subset = _freq_counts.filter(pl.col("indicative") == _indicative_status)
        if len(_subset) == 0:
            continue
        _values_by_code_type = {
            row["code_type_label"]: row["quantity"] for row in _subset.iter_rows(named=True)
        }
        _fig_freq.add_trace(
            go.Bar(
                x=_code_types,
                y=[_values_by_code_type.get(_code_type, 0) for _code_type in _code_types],
                name=_label,
                marker=dict(color=_color),
            )
        )

    _fig_freq.update_layout(
        barmode="stack",
        barnorm="percent",
        xaxis_title="Code type",
        yaxis_title="% of records",
        yaxis_ticksuffix="%",
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            bgcolor="white",
            bordercolor="#DDDDDD",
            borderwidth=1,
            tracegroupgap=10,
        ),
    )

    _fig_freq.write_image("indicative_representation.pdf")
    mo.ui.plotly(_fig_freq)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Records and unique diagnosis

    Per city
    """)
    return


@app.cell
def _(all_cities_df):
    all_cities_df.shape
    return


@app.cell
def _(all_cities_df, pl):
    all_cities_df.select(pl.col("quantity").sum().alias("total"))  # unique records
    return


@app.cell
def _(all_cities_df):
    all_cities_df.select(["code", "code_type"]).n_unique()  # sanity check
    return


@app.cell(hide_code=True)
def _(all_cities_df, pl):
    all_cities_df.group_by("city").agg(
        pl.col("quantity").sum().alias("total"),
        pl.struct("code", "code_type").n_unique().alias("unique_codes"),
    ).sort("total", descending=True)
    return


@app.cell
def _(all_cities_df):
    all_cities_df.sort(["week", "year"])
    return


@app.cell(hide_code=True)
def _(mo, year_max, year_min):
    mo.md(f"""
    #### Data funnel

    Effect of the cleaning and preparation filters on the number of unique
    codes (`code` + `code_type`) and total records (sum of `quantity`) across
    the 10 study cities, restricted to the study window
    **{year_min}–{year_max}**.

    - **Raw**: records from the 10 study cities with a parseable `CIDCIAP`
      field (regex `^[A-Z]+\\(...\\)`).
    - **Cleaned**: after dropping records with missing year, week or age
      group.
    - **Prepared**: after additionally dropping non-indicative codes and
      ICD-10 Z-chapter codes.
    """)
    return


@app.cell
def _(
    CODE_TYPES_MAPPING,
    IBGE_CITY_MAP,
    RAW_DATA_FILE,
    pl,
    year_max,
    year_min,
):
    ibge_to_city = {str(ibge_code): data["city"] for ibge_code, data in IBGE_CITY_MAP.items()}

    raw_normalized_df = (
        pl.scan_parquet(str(RAW_DATA_FILE))
        .filter(
            pl.col("CO_MUNICIPIO_IBGE").is_in(list(ibge_to_city.keys()))
            & pl.col("ANO").is_between(year_min, year_max)
        )
        .with_columns(
            pl.col("CO_MUNICIPIO_IBGE").replace_strict(ibge_to_city).alias("city"),
            pl.col("CIDCIAP").str.extract(r"^([A-Z]+)\(", 1).alias("code_type_raw"),
            pl.col("CIDCIAP").str.extract(r"\(([^)]+)\)", 1).alias("code_extracted"),
        )
        .filter(pl.col("code_type_raw").is_not_null() & pl.col("code_extracted").is_not_null())
        .with_columns(
            pl.when(pl.col("code_type_raw").eq("AB"))
            .then(pl.col("code_extracted").str.replace(r"-\d+", ""))
            .otherwise(pl.col("code_extracted"))
            .alias("code"),
        )
        .with_columns(
            pl.col("code_type_raw")
            .replace_strict(CODE_TYPES_MAPPING, default=None)
            .alias("code_type"),
        )
        .filter(pl.col("code_type").is_not_null())
        .select(["city", "code", "code_type", "QT"])
        .collect()
    )
    return (raw_normalized_df,)


@app.cell
def _(load_all_cities, pl, year_max, year_min):
    cleaned_full_df = load_all_cities(from_="cleaned").filter(
        pl.col("year").is_between(year_min, year_max)
    )
    prepared_full_df = load_all_cities(from_="prepared").filter(
        pl.col("year").is_between(year_min, year_max)
    )
    return cleaned_full_df, prepared_full_df


@app.cell
def _(cleaned_full_df, pl, prepared_full_df, raw_normalized_df):
    def _stage_stats_per_city(dataframe, quantity_column):
        return dataframe.group_by("city").agg(
            pl.struct("code", "code_type").n_unique().alias("unique_codes"),
            pl.col(quantity_column).sum().alias("records"),
        )

    def _stage_stats_overall(dataframe, quantity_column):
        return dataframe.select(
            pl.struct("code", "code_type").n_unique().alias("unique_codes"),
            pl.col(quantity_column).sum().alias("records"),
        )

    stage_order = ["Raw", "Cleaned", "Prepared"]

    funnel_per_city = pl.concat(
        [
            _stage_stats_per_city(raw_normalized_df, "QT").with_columns(stage=pl.lit("Raw")),
            _stage_stats_per_city(cleaned_full_df, "quantity").with_columns(
                stage=pl.lit("Cleaned")
            ),
            _stage_stats_per_city(prepared_full_df, "quantity").with_columns(
                stage=pl.lit("Prepared")
            ),
        ]
    ).select(["stage", "city", "unique_codes", "records"])

    funnel_overall = pl.concat(
        [
            _stage_stats_overall(raw_normalized_df, "QT").with_columns(stage=pl.lit("Raw")),
            _stage_stats_overall(cleaned_full_df, "quantity").with_columns(
                stage=pl.lit("Cleaned")
            ),
            _stage_stats_overall(prepared_full_df, "quantity").with_columns(
                stage=pl.lit("Prepared")
            ),
        ]
    ).select(["stage", "unique_codes", "records"])
    return funnel_overall, funnel_per_city, stage_order


@app.cell
def _(funnel_overall):
    funnel_overall
    return


@app.cell
def _(funnel_per_city):
    funnel_per_city
    return


@app.cell
def _(funnel_overall, go, make_subplots, mo, stage_order):
    _overall_by_stage = {row["stage"]: row for row in funnel_overall.iter_rows(named=True)}
    _ordered_stages = [stage for stage in stage_order if stage in _overall_by_stage]
    _unique_codes_values = [_overall_by_stage[stage]["unique_codes"] for stage in _ordered_stages]
    _records_values = [_overall_by_stage[stage]["records"] for stage in _ordered_stages]

    _grayscale_palette = ["#BDBDBD", "#707070", "#2E2E2E"]
    _stage_fill = _grayscale_palette[: len(_ordered_stages)]
    _stage_text_color = ["#1A1A1A", "#FFFFFF", "#FFFFFF"][: len(_ordered_stages)]

    figure_funnel_overall = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Unique codes (code + code_type)", "Records (sum of quantity)"),
        horizontal_spacing=0.18,
    )

    figure_funnel_overall.add_trace(
        go.Funnel(
            y=_ordered_stages,
            x=_unique_codes_values,
            texttemplate="%{value:,}<br>%{percentInitial:.2%}",
            textposition="inside",
            textfont=dict(size=16, color=_stage_text_color),
            marker=dict(color=_stage_fill),
            connector=dict(line=dict(color="#888888", width=1)),
            hovertemplate="<b>%{y}</b><br>Unique codes: %{x:,}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    figure_funnel_overall.add_trace(
        go.Funnel(
            y=_ordered_stages,
            x=_records_values,
            texttemplate="%{value:,}<br>%{percentInitial:.2%}",
            textposition="inside",
            textfont=dict(size=16, color=_stage_text_color),
            marker=dict(color=_stage_fill),
            connector=dict(line=dict(color="#888888", width=1)),
            hovertemplate="<b>%{y}</b><br>Records: %{x:,}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    figure_funnel_overall.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        width=1100,
        height=480,
        margin=dict(l=20, r=20, t=60, b=30),
        font=dict(size=15, color="#1A1A1A"),
    )

    figure_funnel_overall.write_image("data_funnel_overall.pdf")
    mo.ui.plotly(figure_funnel_overall)
    return


@app.cell
def _(funnel_per_city, go, make_subplots, mo, pl, stage_order):
    _city_order = (
        funnel_per_city.filter(pl.col("stage") == "Raw")
        .sort("unique_codes", descending=False)["city"]
        .to_list()
    )

    _stage_colors = {
        "Raw": "#BDBDBD",
        "Cleaned": "#707070",
        "Prepared": "#2E2E2E",
    }

    figure_funnel_per_city = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Unique codes by city", "Records by city"),
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )

    _raw_lookup = {
        row["city"]: row
        for row in funnel_per_city.filter(pl.col("stage") == "Raw").iter_rows(named=True)
    }

    for _stage in stage_order:
        _subset = funnel_per_city.filter(pl.col("stage") == _stage)
        _by_city = {row["city"]: row for row in _subset.iter_rows(named=True)}

        _codes_values = [_by_city.get(city, {}).get("unique_codes", 0) for city in _city_order]
        _records_values = [_by_city.get(city, {}).get("records", 0) for city in _city_order]

        _codes_pct = [
            (
                f"{value / _raw_lookup[city]['unique_codes']:.2%}"
                if _raw_lookup.get(city, {}).get("unique_codes")
                else ""
            )
            for value, city in zip(_codes_values, _city_order)
        ]
        _records_pct = [
            (
                f"{value / _raw_lookup[city]['records']:.2%}"
                if _raw_lookup.get(city, {}).get("records")
                else ""
            )
            for value, city in zip(_records_values, _city_order)
        ]

        figure_funnel_per_city.add_trace(
            go.Bar(
                y=_city_order,
                x=_codes_values,
                name=_stage,
                orientation="h",
                marker=dict(color=_stage_colors[_stage]),
                text=_codes_pct,
                textposition="outside",
                textfont=dict(size=11),
                legendgroup=_stage,
                showlegend=True,
                hovertemplate=(f"<b>%{{y}}</b><br>{_stage}: %{{x:,}} unique codes<extra></extra>"),
            ),
            row=1,
            col=1,
        )

        figure_funnel_per_city.add_trace(
            go.Bar(
                y=_city_order,
                x=_records_values,
                name=_stage,
                orientation="h",
                marker=dict(color=_stage_colors[_stage]),
                text=_records_pct,
                textposition="outside",
                textfont=dict(size=11),
                legendgroup=_stage,
                showlegend=False,
                hovertemplate=f"<b>%{{y}}</b><br>{_stage}: %{{x:,}} records<extra></extra>",
            ),
            row=1,
            col=2,
        )

    figure_funnel_per_city.update_layout(
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=1200,
        height=620,
        margin=dict(l=20, r=40, t=60, b=40),
        font=dict(size=14),
        legend=dict(
            title_text="Stage",
            bgcolor="white",
            bordercolor="#DDDDDD",
            borderwidth=1,
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.08,
        ),
    )
    figure_funnel_per_city.update_xaxes(
        title_text="Unique codes",
        row=1,
        col=1,
        gridcolor="#EEEEEE",
        zeroline=False,
    )
    figure_funnel_per_city.update_xaxes(
        title_text="Records",
        row=1,
        col=2,
        gridcolor="#EEEEEE",
        zeroline=False,
    )
    figure_funnel_per_city.update_yaxes(
        autorange="reversed",
        row=1,
        col=1,
    )

    figure_funnel_per_city.write_image("data_funnel_per_city.pdf")
    mo.ui.plotly(figure_funnel_per_city)
    return


if __name__ == "__main__":
    app.run()
