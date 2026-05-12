import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import plotly.express as px
    import polars as pl

    return mo, pl, px


@app.cell
def _():
    from vir.adapters.datasus.features import CITY_NAMES
    from vir.db import fetch_run_leiden, fetch_run_multi_k, fetch_runs

    return CITY_NAMES, fetch_run_leiden, fetch_run_multi_k, fetch_runs


@app.cell
def _(mo):
    mo.md(r"""
    # Runs Overview

    Compare the latest SNF run per city. Toggle the selector to include older runs.
    """)
    return


@app.cell
def _(CITY_NAMES, fetch_runs, mo, pl):
    all_runs = fetch_runs().filter(pl.col("city").is_in(CITY_NAMES))

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per city analysis
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Silhouette score by city
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
    ### Corpus size
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
    ### Cluster stats

    Biggest cluster, number/percentage of singletons and silhouette score.
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
        title="Cluster Stats",
        labels={"cluster_median": "Median Cluster Size", "cluster_largest": "Largest Cluster"},
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ### Multi-K comparison
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


@app.cell
def _(all_runs, fetch_run_multi_k, mo, multi_k_city, pl, px):
    mo.stop(not multi_k_city.value)

    _city_runs = (
        all_runs.filter(pl.col("city") == multi_k_city.value)
        .sort("created_at", descending=True)
        .select(["run_id", "created_at", "notes", "n_clusters"])
    )

    mo.stop(_city_runs.is_empty())

    _frames = []
    for _row in _city_runs.iter_rows(named=True):
        _mk = fetch_run_multi_k(_row["run_id"])
        if _mk.is_empty() or "affinity_zscore" not in _mk.columns:
            continue
        if _mk["affinity_zscore"].is_null().all():
            continue
        _label = f"#{_row['run_id']}"
        if _row["notes"]:
            _label += f" – {_row['notes']}"
        _frames.append(_mk.with_columns(pl.lit(_label).alias("run")))

    mo.stop(
        not _frames,
        mo.callout(
            mo.md(
                f"No affinity Z-score data for {multi_k_city.value}. Re-run the pipeline to populate."
            ),
            kind="warn",
        ),
    )

    _combined_zaff = pl.concat(_frames)

    _fig = px.line(
        _combined_zaff,
        x="k",
        y="affinity_zscore",
        color="run",
        markers=True,
        title=f"Affinity Z-score vs K — {multi_k_city.value}",
        labels={
            "k": "Number of clusters (K)",
            "affinity_zscore": "Affinity Z-score",
            "run": "Run",
        },
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ### Leiden community detection sweep
    """)
    return


@app.cell
def _(all_runs, fetch_run_leiden, mo, multi_k_city, pl, px):
    mo.stop(not multi_k_city.value)

    _city_runs_leiden = (
        all_runs.filter(pl.col("city") == multi_k_city.value)
        .sort("created_at", descending=True)
        .select(["run_id", "created_at", "notes"])
    )

    mo.stop(_city_runs_leiden.is_empty())

    _frames_leiden = []
    for _row in _city_runs_leiden.iter_rows(named=True):
        _ld = fetch_run_leiden(_row["run_id"])
        if _ld.is_empty():
            continue
        _label = f"#{_row['run_id']}"
        if _row["notes"]:
            _label += f" – {_row['notes']}"
        _frames_leiden.append(_ld.with_columns(pl.lit(_label).alias("run")))

    mo.stop(
        not _frames_leiden,
        mo.callout(
            mo.md(f"No Leiden data for {multi_k_city.value}. Re-run the pipeline to populate."),
            kind="warn",
        ),
    )

    _combined_leiden = pl.concat(_frames_leiden)

    _fig = px.line(
        _combined_leiden,
        x="resolution",
        y="n_communities",
        color="run",
        markers=True,
        log_x=True,
        title=f"Leiden: communities vs resolution — {multi_k_city.value}",
        labels={
            "resolution": "Resolution (γ, log scale)",
            "n_communities": "Number of communities",
            "run": "Run",
        },
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multi-city comparison
    """)
    return


@app.cell
def _(latest_per_city, mo):
    _run_options = {
        f"{row['city']} (#{row['run_id']})": row["run_id"]
        for row in latest_per_city.iter_rows(named=True)
    }
    run_id_to_city = {row["run_id"]: row["city"] for row in latest_per_city.iter_rows(named=True)}

    cross_city_runs = mo.ui.multiselect(
        options=_run_options,
        value=list(_run_options.keys()),
        label="Cities",
    )
    cross_city_runs
    return cross_city_runs, run_id_to_city


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Affinity Z-score vs K
    """)
    return


@app.cell
def _(cross_city_runs, fetch_run_multi_k, mo, pl, px, run_id_to_city):
    mo.stop(not cross_city_runs.value)

    _frames = []
    for _run_id in cross_city_runs.value:
        _mk = fetch_run_multi_k(_run_id)
        if _mk.is_empty() or "affinity_zscore" not in _mk.columns:
            continue
        if _mk["affinity_zscore"].is_null().all():
            continue
        _city = run_id_to_city[_run_id]
        _frames.append(_mk.with_columns(pl.lit(_city).alias("city")))

    mo.stop(
        not _frames,
        mo.callout(
            mo.md("No affinity Z-score data for the selected runs."),
            kind="warn",
        ),
    )

    zscore_combined = pl.concat(_frames)

    _fig = px.line(
        zscore_combined,
        x="k",
        y="affinity_zscore",
        color="city",
        markers=True,
        title="Affinity Z-score vs K",
        labels={"k": "K", "affinity_zscore": "Affinity Z-score", "city": "City"},
    )
    _fig.add_vline(x=200, line_dash="dash", line_color="gray", annotation_text="K=200")
    _fig
    return (zscore_combined,)


@app.cell
def _(zscore_combined):
    zscore_combined
    return


@app.cell
def _(pl, zscore_combined):
    zscore_combined.group_by("k").agg(pl.col("affinity_zscore").median()).sort("k")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Leiden vs K
    """)
    return


@app.cell
def _(cross_city_runs, fetch_run_leiden, mo, pl, px, run_id_to_city):
    mo.stop(not cross_city_runs.value)

    _frames = []
    for _run_id in cross_city_runs.value:
        _ld = fetch_run_leiden(_run_id)
        if _ld.is_empty():
            continue
        _city = run_id_to_city[_run_id]
        _frames.append(_ld.with_columns(pl.lit(_city).alias("city")))

    mo.stop(
        not _frames,
        mo.callout(
            mo.md("No Leiden data for the selected runs."),
            kind="warn",
        ),
    )

    leiden_combined = pl.concat(_frames)

    _fig = px.line(
        leiden_combined,
        x="resolution",
        y="n_communities",
        color="city",
        markers=True,
        log_x=True,
        title="Leiden: communities vs resolution",
        labels={
            "resolution": "Resolution (γ, log scale)",
            "n_communities": "Number of communities",
            "city": "City",
        },
    )
    _fig.add_hline(y=200, line_dash="dash", line_color="gray", annotation_text="K=200")
    _fig
    return (leiden_combined,)


@app.cell
def _(leiden_combined):
    leiden_combined
    return


@app.cell
def _(leiden_combined, pl):
    leiden_combined.filter(pl.col("resolution").eq(0.5000000000000000000)).with_columns(
        pl.col("n_communities").min().alias("min"),
        pl.col("n_communities").median().alias("median"),
        pl.col("n_communities").max().alias("max"),
    )
    return


@app.cell
def _(leiden_combined, pl):
    leiden_combined.filter(pl.col("resolution").eq(0.5000000000000000000)).with_columns(
        pl.col("n_communities").median()
    )
    return


if __name__ == "__main__":
    app.run()
