import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import plotly.express as px
    import polars as pl

    return mo, pl, px


@app.cell
def _():
    from vir.db import fetch_run_multi_k, fetch_runs

    return fetch_run_multi_k, fetch_runs


@app.cell
def _(mo):
    mo.md(r"""
    # Runs Overview

    Compare the latest SNF run per city. Toggle the selector to include older runs.
    """)
    return


@app.cell
def _(fetch_runs, mo):
    all_runs = fetch_runs()

    mo.stop(
        all_runs.is_empty(),
        mo.callout(mo.md("No runs found. Run `vir model snf` first."), kind="warn"),
    )

    latest_per_city = (
        all_runs.sort("created_at", descending=True).group_by("city").first().sort("city")
    )

    show_all_toggle = mo.ui.checkbox(label="Show all runs (not just latest per city)", value=False)
    show_all_toggle
    return all_runs, latest_per_city, show_all_toggle


@app.cell
def _(all_runs, latest_per_city, show_all_toggle):
    runs_df = all_runs if show_all_toggle.value else latest_per_city
    return (runs_df,)


@app.cell
def _(mo, runs_df):
    display_cols = [
        "run_id",
        "city",
        "region",
        "population_range",
        "year_min",
        "year_max",
        "snf_k",
        "n_clusters",
        "silhouette_score",
        "affinity_zscore",
        "n_codes_common",
        "n_codes_clinical",
        "n_codes_temporal",
        "n_codes_demographic",
        "n_missing_temporal",
        "n_missing_demographic",
        "cluster_largest",
        "cluster_median",
        "n_singletons",
        "pct_singletons",
        "duration_min",
        "notes",
    ]
    available_cols = [c for c in display_cols if c in runs_df.columns]
    mo.ui.table(runs_df.select(available_cols))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Silhouette score by city
    """)
    return


@app.cell
def _(px, runs_df):
    _fig = px.bar(
        runs_df,
        x="city",
        y="silhouette_score",
        color="region",
        hover_data=["run_id", "n_clusters", "snf_k", "affinity_zscore"],
        title="Silhouette Score per City (latest run)",
        labels={"silhouette_score": "Silhouette", "city": "City"},
    )
    _fig.update_layout(xaxis_tickangle=-30)
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Corpus size by city
    """)
    return


@app.cell
def _(mo, px, runs_df):
    mo.stop("n_codes_common" not in runs_df.columns or runs_df["n_codes_common"].is_null().all())

    _fig = px.bar(
        runs_df,
        x="city",
        y=["n_codes_clinical", "n_codes_temporal", "n_codes_demographic", "n_codes_common"],
        barmode="group",
        title="Code Coverage per View by City",
        labels={"value": "# Codes", "variable": "View", "city": "City"},
    )
    _fig.update_layout(xaxis_tickangle=-30)
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Cluster quality by city
    """)
    return


@app.cell
def _(mo, px, runs_df):
    mo.stop("cluster_largest" not in runs_df.columns or runs_df["cluster_largest"].is_null().all())

    _fig = px.scatter(
        runs_df,
        x="cluster_median",
        y="cluster_largest",
        size="n_clusters",
        color="city",
        hover_data=["run_id", "n_singletons", "pct_singletons", "silhouette_score"],
        title="Cluster Distribution Quality by City",
        labels={"cluster_median": "Median Cluster Size", "cluster_largest": "Largest Cluster"},
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Multi-K comparison
    """)
    return


@app.cell
def _(all_runs, mo):
    city_options = all_runs.sort("city")["city"].drop_nulls().unique(maintain_order=True).to_list()
    multi_k_city = mo.ui.dropdown(
        options=city_options, label="City", value=city_options[0] if city_options else None
    )
    multi_k_city
    return (multi_k_city,)


@app.cell
def _(all_runs, fetch_run_multi_k, mo, multi_k_city, pl, px):
    mo.stop(not multi_k_city.value)

    _city_runs = (
        all_runs.filter(pl.col("city") == multi_k_city.value)
        .sort("created_at", descending=True)
        .select(["run_id", "created_at", "notes", "n_clusters"])
    )

    mo.stop(
        _city_runs.is_empty(),
        mo.callout(mo.md(f"No runs found for {multi_k_city.value}."), kind="warn"),
    )

    _frames = []
    for _row in _city_runs.iter_rows(named=True):
        _mk = fetch_run_multi_k(_row["run_id"])
        if _mk.is_empty():
            continue
        _label = f"#{_row['run_id']}"
        if _row["notes"]:
            _label += f" – {_row['notes']}"
        _frames.append(_mk.with_columns(pl.lit(_label).alias("run")))

    mo.stop(
        not _frames,
        mo.callout(
            mo.md(
                f"No multi-K data for any {multi_k_city.value} run. Re-run the pipeline to populate."
            ),
            kind="warn",
        ),
    )

    _combined = pl.concat(_frames)

    _fig = px.line(
        _combined,
        x="k",
        y="silhouette",
        color="run",
        markers=True,
        title=f"Silhouette vs K — {multi_k_city.value}",
        labels={"k": "Number of clusters (K)", "silhouette": "Silhouette Score", "run": "Run"},
    )
    _fig
    return


if __name__ == "__main__":
    app.run()
