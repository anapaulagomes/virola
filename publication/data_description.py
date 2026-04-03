import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import pathlib
    from collections import defaultdict

    import marimo as mo
    import plotly.graph_objects as go
    import polars as pl

    from vir.config import INDICATIVE_CODES_FILE, INTERIM_DATA_DIR
    from vir.features import IBGE_CITY_MAP
    from vir.viz import CODE_TYPE_COLOR, TABLEAU_COLORS

    return (
        CODE_TYPE_COLOR,
        IBGE_CITY_MAP,
        INDICATIVE_CODES_FILE,
        INTERIM_DATA_DIR,
        TABLEAU_COLORS,
        defaultdict,
        go,
        json,
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
def _(IBGE_CITY_MAP, TABLEAU_COLORS, pl):
    REGION_TRANSLATIONS = {
        "Norte": "North",
        "Nordeste": "Northeast",
        "Centro-Oeste": "Central-West",
        "Sudeste": "Southeast",
        "Sul": "South",
    }
    POPULATION_RANGE_TRANSLATIONS = {
        "250 - 500 mil": "250K-500K",
        "> 500 mil": "> 500K",
    }
    REGION_COLORS = dict(zip(REGION_TRANSLATIONS.values(), TABLEAU_COLORS[:5]))
    POPULATION_SIZES = {"250K-500K": 11, "> 500K": 22}

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
                legendgrouptitle_text="Region",
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
    for _population_label, _size in POPULATION_SIZES.items():
        figure.add_trace(
            go.Scattergeo(
                lon=[None],
                lat=[None],
                mode="markers",
                name=_population_label,
                legendgroup="population",
                legendgrouptitle_text="Population",
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
        lataxis_range=[-36, 6],
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
            font=dict(size=15),
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
    ### Unique codes

    Indicative vs non-indicative
    """)
    return


@app.cell
def _(CODE_TYPE_COLOR, INDICATIVE_CODES_FILE, INTERIM_DATA_DIR, go, mo, pl):
    _parquet_file = (
        INTERIM_DATA_DIR / "cleaned_base_aesop_raw_extracao_01042026_all_cities.parquet"
    )

    _unique_codes = pl.read_parquet(_parquet_file).select(["code", "code_type"]).unique()

    _indicative_lookup = pl.read_csv(INDICATIVE_CODES_FILE).select(
        ["code", "code_type", "indicative"]
    )

    _counts = (
        _unique_codes.join(_indicative_lookup, on=["code", "code_type"], how="left")
        .with_columns(pl.col("indicative").fill_null("unlabeled"))
        .with_columns(pl.col("code_type").str.to_uppercase().alias("code_type_label"))
        .group_by(["code_type_label", "indicative"])
        .len("count")
        .sort(["code_type_label", "indicative"], descending=[False, True])
    )

    _fig = go.Figure()

    # (status, pattern_shape, opacity)
    _indicative_styles = [("yes", "", 1.0), ("no", "/", 1.0), ("unlabeled", "", 0.35)]

    # Data traces — hidden from legend, both dimensions encoded visually
    for _code_type, _color in CODE_TYPE_COLOR.items():
        for _indicative_status, _pattern_shape, _opacity in _indicative_styles:
            _subset = _counts.filter(
                (pl.col("code_type_label") == _code_type)
                & (pl.col("indicative") == _indicative_status)
            )
            if len(_subset) == 0:
                continue
            _fig.add_trace(
                go.Bar(
                    x=[_code_type],
                    y=_subset["count"].to_list(),
                    opacity=_opacity,
                    marker=dict(
                        color=_color,
                        pattern=dict(shape=_pattern_shape, fgcolor="white", size=6),
                    ),
                    showlegend=False,
                )
            )

    # Legend box 1: Terminology (colors)
    for _code_type, _color in CODE_TYPE_COLOR.items():
        _fig.add_trace(
            go.Bar(
                x=[None],
                y=[None],
                name=_code_type,
                marker=dict(color=_color),
                legendgroup="terminology",
                legendgrouptitle_text="Terminology",
            )
        )

    # Legend box 2: Indicative status (textures/opacity)
    for _label, _pattern_shape, _opacity in [
        ("Indicative", "", 1.0),
        ("Non-indicative", "/", 1.0),
        ("Unlabeled", "", 0.35),
    ]:
        _fig.add_trace(
            go.Bar(
                x=[None],
                y=[None],
                name=_label,
                opacity=_opacity,
                marker=dict(
                    color="#888888",
                    pattern=dict(shape=_pattern_shape, fgcolor="white", size=6),
                ),
                legendgroup="indicative_status",
                legendgrouptitle_text="Indicative status",
            )
        )

    _fig.update_layout(
        barmode="stack",
        barnorm="percent",
        # title="Unique codes by type and indicative status",
        xaxis_title="Code type",
        yaxis_title="% of unique codes",
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

    mo.ui.plotly(_fig)
    return


@app.cell
def _(CODE_TYPE_COLOR, INDICATIVE_CODES_FILE, INTERIM_DATA_DIR, go, mo, pl):
    _parquet_file = (
        INTERIM_DATA_DIR / "cleaned_base_aesop_raw_extracao_01042026_all_cities.parquet"
    )

    _record_counts = pl.read_parquet(_parquet_file).group_by(["code", "code_type"]).len("quantity")

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

    _freq_indicative_styles = [("yes", "", 1.0), ("no", "/", 1.0), ("unlabeled", "", 0.35)]

    for _code_type, _color in CODE_TYPE_COLOR.items():
        for _indicative_status, _pattern_shape, _opacity in _freq_indicative_styles:
            _subset = _freq_counts.filter(
                (pl.col("code_type_label") == _code_type)
                & (pl.col("indicative") == _indicative_status)
            )
            if len(_subset) == 0:
                continue
            _fig_freq.add_trace(
                go.Bar(
                    x=[_code_type],
                    y=_subset["quantity"].to_list(),
                    opacity=_opacity,
                    marker=dict(
                        color=_color,
                        pattern=dict(shape=_pattern_shape, fgcolor="white", size=6),
                    ),
                    showlegend=False,
                )
            )

    for _code_type, _color in CODE_TYPE_COLOR.items():
        _fig_freq.add_trace(
            go.Bar(
                x=[None],
                y=[None],
                name=_code_type,
                marker=dict(color=_color),
                legendgroup="terminology",
                legendgrouptitle_text="Terminology",
            )
        )

    for _label, _pattern_shape, _opacity in [
        ("Indicative", "", 1.0),
        ("Non-indicative", "/", 1.0),
        ("Unlabeled", "", 0.35),
    ]:
        _fig_freq.add_trace(
            go.Bar(
                x=[None],
                y=[None],
                name=_label,
                opacity=_opacity,
                marker=dict(
                    color="#888888",
                    pattern=dict(shape=_pattern_shape, fgcolor="white", size=6),
                ),
                legendgroup="indicative_status",
                legendgrouptitle_text="Indicative status",
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


if __name__ == "__main__":
    app.run()
