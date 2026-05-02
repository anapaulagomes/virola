import marimo

__generated_with = "0.19.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import plotly.express as px
    import polars as pl

    from publication.commons import POPULATION_RANGE_TRANSLATIONS, REGION_TRANSLATIONS
    from vir.db import fetch_cluster_profiles_for_runs, fetch_runs
    from vir.features import CITY_NAMES
    from vir.viz import TABLEAU_COLORS

    return (
        CITY_NAMES,
        POPULATION_RANGE_TRANSLATIONS,
        REGION_TRANSLATIONS,
        TABLEAU_COLORS,
        fetch_cluster_profiles_for_runs,
        fetch_runs,
        mo,
        pl,
        px,
    )


@app.cell
def _(REGION_TRANSLATIONS, TABLEAU_COLORS):
    REGION_ORDER = ["North", "Northeast", "Central-West", "South", "Southeast"]
    POPULATION_ORDER = ["250K-500K", "> 500K"]
    REGION_COLORS = dict(zip(REGION_TRANSLATIONS.values(), TABLEAU_COLORS[:5]))
    return POPULATION_ORDER, REGION_COLORS, REGION_ORDER


@app.cell
def _(
    CITY_NAMES,
    POPULATION_RANGE_TRANSLATIONS,
    REGION_TRANSLATIONS,
    fetch_cluster_profiles_for_runs,
    fetch_runs,
    pl,
):
    _latest_runs = (
        fetch_runs()
        .filter(pl.col("city").is_in(CITY_NAMES))
        .group_by("city")
        .first()
        .select(["run_id", "city", "region", "population_range"])
    )
    _profiles = fetch_cluster_profiles_for_runs(_latest_runs["run_id"].to_list())
    cluster_sizes = (
        _profiles.join(_latest_runs, on="run_id")
        .filter(pl.col("n_codes").is_not_null())
        .with_columns(
            pl.col("region").replace(REGION_TRANSLATIONS),
            pl.col("population_range").replace(POPULATION_RANGE_TRANSLATIONS),
        )
        .select(["city", "region", "population_range", "cluster_id", "n_codes"])
    )
    return (cluster_sizes,)


@app.cell
def _(cluster_sizes):
    cluster_sizes["city"].unique()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Overview
    """)
    return


@app.cell
def _(cluster_sizes, pl):
    partition_overview = (
        cluster_sizes.group_by("city")
        .agg(
            (pl.col("n_codes") == 1).sum().alias("singletons"),
            pl.col("n_codes").quantile(0.25, interpolation="nearest").cast(pl.Int64).alias("q25"),
            pl.col("n_codes").quantile(0.50, interpolation="nearest").cast(pl.Int64).alias("q50"),
            pl.col("n_codes").quantile(0.75, interpolation="nearest").cast(pl.Int64).alias("q75"),
            pl.col("n_codes").max().alias("residual"),
        )
        .sort("city")
        .with_columns(pl.col("city").cast(pl.String))
    )
    partition_overview
    return


@app.cell
def _(POPULATION_ORDER, REGION_COLORS, cluster_sizes, mo, pl, px):
    _city_region_mapping = dict(cluster_sizes.select(["city", "region"]).unique().iter_rows())
    _city_color_mapping = {
        city: REGION_COLORS[region] for city, region in _city_region_mapping.items()
    }
    _ecdf_figures = []
    for _population_range_label in POPULATION_ORDER:
        _subset = cluster_sizes.filter(pl.col("population_range") == _population_range_label)
        _fig = px.ecdf(
            _subset,
            x="n_codes",
            log_x=True,
            color="city",
            color_discrete_map=_city_color_mapping,
            labels={"n_codes": "Cluster size (number of codes)", "city": "City"},
            title=_population_range_label,
        )
        _fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=400,
        )
        _ecdf_figures.append(mo.ui.plotly(_fig))
    mo.hstack(_ecdf_figures)


@app.cell
def _(POPULATION_ORDER, REGION_COLORS, REGION_ORDER, cluster_sizes, mo, px):
    _fig = px.ecdf(
        cluster_sizes,
        x="n_codes",
        log_x=True,
        color="region",
        color_discrete_map=REGION_COLORS,
        facet_row="population_range",
        facet_col="region",
        facet_row_spacing=0.1,
        category_orders={
            "region": REGION_ORDER,
            "population_range": POPULATION_ORDER,
        },
        labels={"n_codes": "Cluster size"},
        hover_name="city",
    )

    _fig.update_xaxes(matches=None)
    for _trace in _fig.data:
        _xaxis_key = "xaxis" + _trace.xaxis[1:]
        if _trace.hovertext is not None and len(_trace.hovertext) > 0:
            _fig.layout[_xaxis_key].update(
                title=dict(text=_trace.hovertext[0], font=dict(size=10))
            )

    _fig.update_layout(
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=500,
    )
    _fig.for_each_annotation(
        lambda annotation: annotation.update(text=annotation.text.split("=")[-1])
    )
    mo.ui.plotly(_fig)
    return


if __name__ == "__main__":
    app.run()
