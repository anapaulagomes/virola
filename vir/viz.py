import plotly.express as px
import polars as pl

TABLEAU_COLORS = [
    "#4E79A7",  # Blue
    "#F28E2B",  # Orange
    "#E15759",  # Red
    "#76B7B2",  # Cyan
    "#59A14F",  # Green
    "#EDC948",  # Yellow
    "#B07AA1",  # Purple
    "#FF9DA7",  # Pink
    "#9C755F",  # Brown
    "#BAB0AC",  # Gray
]


def plot_null_vs_not_null(df, verbose=False):
    _proportion = []
    for _col, _valid, _null in zip(df.columns, df.count(), df.null_count()):
        _total = _valid[0] + _null[0]
        _not_null = _valid[0] / _total
        _nulls = _null[0] / _total
        _proportion.append({"column": _col, "status": "not-null", "proportion": _not_null})
        _proportion.append({"column": _col, "status": "null", "proportion": _nulls})
        if verbose:
            print(f"{_col} not-nulls {_valid[0]} nulls {_null[0]}")

    _fig = px.bar(
        pl.DataFrame(_proportion),
        x="column",
        y="proportion",
        color="status",
        barmode="stack",
        title="Null vs non-null values per column",
    )
    _fig.update_layout(yaxis_tickformat=".0%")
    return _fig


def plot_categories(df, column, x_categoryorder=None, title=None):
    _count = df[column].value_counts()
    _fig = px.bar(_count, x=column, y="count", title=title)
    if x_categoryorder:
        _fig.update_xaxes(categoryorder="array", categoryarray=x_categoryorder)
    return _fig


def scatter_plot_year_per_week(
    df, year_column, week_column, color_column=None, color_schema="viridis", title=None
):
    _year_week = df.group_by([year_column, week_column]).len("count")
    _params = {}
    if color_column:
        _params = {"color": color_column, "color_continuous_scale": color_schema}
    _fig = px.scatter(_year_week, x=week_column, y="count", title=title, **_params)
    _fig.update_layout(xaxis=dict(tickmode="linear", dtick=1))
    return _fig
