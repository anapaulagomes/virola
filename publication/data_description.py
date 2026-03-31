import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import pathlib
    from collections import defaultdict

    import marimo as mo
    import plotly.graph_objects as go
    import polars as pl

    from vir.features import IBGE_CITY_MAP
    from vir.viz import TABLEAU_COLORS

    return (
        IBGE_CITY_MAP,
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


if __name__ == "__main__":
    app.run()
