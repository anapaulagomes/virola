import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    from datetime import datetime

    import marimo as mo
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import plotly.express as px
    import polars as pl
    from plotly.subplots import make_subplots

    from vir.config import INTERIM_DATA_DIR

    return INTERIM_DATA_DIR, cm, datetime, make_subplots, mcolors, mo, pl, px


@app.cell
def _(INTERIM_DATA_DIR, mo):
    _all_options = [
        str(_file.resolve()) for _file in INTERIM_DATA_DIR.glob("**/cleaned_*.parquet")
    ]

    files_dropdown = mo.ui.dropdown(
        options=_all_options,
        label="Choose dataset (record-level or diagnosis-level)",
        searchable=True,
    )
    files_dropdown
    return (files_dropdown,)


@app.cell
def _(datetime):
    def week_year_to_date(year: int, week: int) -> datetime:
        # ISO week: Monday as first day
        return datetime.strptime(f"{year}-W{week}-1", "%G-W%V-%u")

    return (week_year_to_date,)


@app.cell
def _(files_dropdown, pl, week_year_to_date):
    df = pl.read_parquet(files_dropdown.value)
    df = df.with_columns(
        pl.struct(["year", "week"])
        .map_elements(
            lambda row: week_year_to_date(row["year"], row["week"]), return_dtype=pl.Datetime
        )
        .alias("date")
    )
    df
    return (df,)


@app.cell
def _(df, pl):
    count = df.group_by("code_type", "code").len().sort(by="len")
    count = count.join(count, on=["code", "code_type"])
    quantile75 = count["len"].quantile(0.75)
    count.filter(pl.col("len").gt(quantile75))
    return


@app.cell
def _(cm, df, mcolors, pl, px):
    def plot_code(code, code_type):
        counts = (
            df.filter(pl.col("code").eq(code) & pl.col("code_type").eq(code_type))
            .group_by(["year", "week"])
            .agg(pl.col("quantity").sum().alias("count"))
            .sort(["year", "week"])
        )

        # Generate a color gradient from light → dark (e.g. Blues)
        cmap = cm.get_cmap("Blues", 11)  # 11 years from 2015–2025
        years = list(range(2015, 2026))
        year_colors = {year: mcolors.rgb2hex(cmap(i)[:3]) for i, year in enumerate(years)}

        fig = px.line(
            counts,
            x="week",
            y="count",
            color="year",
            markers=True,
            title=f"Frequência semanal: {code}",
            color_discrete_map=year_colors,
        )

        # Axis setup
        fig.update_layout(
            xaxis=dict(title="Semana", tickmode="linear", dtick=1, range=[1, 53]),
            yaxis=dict(title="Quantidade"),
            legend_title_text="Ano",  # show legend with title
        )

        return fig

    return (plot_code,)


@app.cell
def _(make_subplots, plot_code):
    _fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[
            "Hipertensão",
            "Diabetes",
            "Nasofaringite aguda [resfriado comum]",
            "Gripe",
        ],
    )

    _fig1 = plot_code("ABP005", "AB")
    _fig2 = plot_code("ABP006", "AB")
    _fig3 = plot_code("J00", "CID")
    _fig4 = plot_code("R80", "CIAP")

    for trace in _fig1.data:
        _fig.add_trace(trace, row=1, col=1)

    for trace in _fig2.data:
        _fig.add_trace(trace, row=1, col=2)

    for trace in _fig3.data:
        _fig.add_trace(trace, row=2, col=1)

    for trace in _fig4.data:
        _fig.add_trace(trace, row=2, col=2)

    return


@app.cell
def _(make_subplots, plot_code):
    _fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Sífilis",
            "HIV",
            "Dengue",
            "Asma",
        ],
    )

    _fig1 = plot_code("X70", "CIAP")
    _fig2 = plot_code("B24", "CID")
    _fig3 = plot_code("ABP019", "AB")
    _fig4 = plot_code("ABP009", "AB")

    for _trace in _fig1.data:
        _fig.add_trace(_trace, row=1, col=1)

    for _trace in _fig2.data:
        _trace.showlegend = False
        _fig.add_trace(_trace, row=1, col=2)

    for _trace in _fig3.data:
        _trace.showlegend = False
        _fig.add_trace(_trace, row=2, col=1)

    for _trace in _fig4.data:
        _trace.showlegend = False
        _fig.add_trace(_trace, row=2, col=2)

    _fig.update_layout(
        height=800,
        width=1000,
        title_text="Indicadores por Semana",
        legend_title_text="Ano",
        showlegend=True,
    )
    return


@app.cell
def _(make_subplots, plot_code):
    _fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[
            "Dengue (CIAP)",
            "Dengue (CID)",
            "Chikungunya (CID)",
            "Zika",
            "Oropouche (CID)",
            "Unspecified viral hemorrhagic fevers (CID)",
        ],
    )

    """
    CIAP;A77;Dengue and other viral diseases
    CID 10;A90;Dengue
    CID 10;A91;Dengue hemorrhagic fever
    ABP;ABP019;Severe dengue
    CID;A972;Dengue grave
    CID;A979;Unspecified dengue
    CID 10;A920;Chikungunya
    CID 10;A92;Other mosquito-borne viral fevers
    CID 10;A928;Other specified mosquito-borne viral fevers
    CID 10;A929;Mosquito-borne viral fever, unspecified
    CID 10;A99;Unspecified viral hemorrhagic fevers
    CID 10;A98;Other viral hemorrhagic fevers
    CID;A925;Zika
    CID;A93;Other arboviral diseases
    CID;A988;Other viral hemorrhagic fevers
    CID;A93;Oropouche fever
    """

    _fig1 = plot_code("A77", "CIAP")
    _fig2 = plot_code("A90", "CID")
    _fig3 = plot_code("A920", "CID")
    _fig4 = plot_code("A925", "CID")
    _fig5 = plot_code("A93", "CID")
    _fig6 = plot_code("A99", "CID")

    for _trace in _fig1.data:
        _trace.showlegend = False
        _fig.add_trace(_trace, row=1, col=1)

    for _trace in _fig2.data:
        # _trace.showlegend = False
        _fig.add_trace(_trace, row=1, col=2)

    for _trace in _fig3.data:
        _trace.showlegend = False
        _fig.add_trace(_trace, row=2, col=1)

    for _trace in _fig4.data:
        _trace.showlegend = False
        _fig.add_trace(_trace, row=2, col=2)

    for _trace in _fig5.data:
        _trace.showlegend = False
        _fig.add_trace(_trace, row=3, col=1)

    for _trace in _fig6.data:
        _trace.showlegend = False
        _fig.add_trace(_trace, row=3, col=2)

    _fig.update_layout(
        height=800,
        width=1000,
        title_text="Indicadores por Semana",
        legend_title_text="Ano",
        showlegend=True,
    )
    _fig
    return


if __name__ == "__main__":
    app.run()
