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
    from vir.adapters import ADAPTERS
    from vir.db import fetch_run_leiden, fetch_run_multi_k, fetch_runs

    return ADAPTERS, fetch_run_leiden, fetch_run_multi_k, fetch_runs


@app.cell
def _(mo):
    args = mo.cli_args() if mo.app_meta().mode == "script" else {}
    return (args,)


@app.cell
def _(ADAPTERS, mo):
    adapter = mo.ui.dropdown(
        options=list(ADAPTERS.keys()),
        label="Choose adapter",
        searchable=True,
        value="datasus",
    )
    adapter
    return (adapter,)


@app.cell
def _(adapter, args):
    adapter_name = args.get("adapter", adapter.value)
    return (adapter_name,)


@app.cell
def _(mo):
    mo.md(r"""
    # Runs Overview

    Compare the latest SNF run per stratum for the selected adapter. Toggle the
    selector to include older runs.
    """)
    return


@app.cell
def _(adapter_name, fetch_runs, mo):
    all_runs = fetch_runs(adapter=adapter_name)

    mo.stop(
        all_runs.is_empty(),
        mo.callout(
            mo.md(f"No runs found for adapter `{adapter_name}`. Run `vir pipeline run` first."),
            kind="warn",
        ),
    )

    latest_per_city = (
        all_runs.sort("created_at", descending=True).group_by("city").first().sort("city")
    )

    show_all_toggle = mo.ui.checkbox(
        label="Show all runs (not just latest per stratum)", value=False
    )
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
        "adapter",
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
    ## Per stratum analysis
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Silhouette score by stratum
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
        title="Silhouette Score per Stratum (latest run)",
        labels={"silhouette_score": "Silhouette", "city": "Stratum"},
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
        title="Code Coverage per View by Stratum",
        labels={"value": "# Codes", "variable": "View", "city": "Stratum"},
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
    stratum_options = (
        all_runs.sort("city")["city"].drop_nulls().unique(maintain_order=True).to_list()
    )
    multi_k_stratum = mo.ui.dropdown(
        options=stratum_options,
        label="Stratum",
        value=stratum_options[0] if stratum_options else None,
    )
    multi_k_stratum
    return (multi_k_stratum,)


@app.cell
def _(all_runs, fetch_run_multi_k, mo, multi_k_stratum, pl, px):
    mo.stop(not multi_k_stratum.value)

    _stratum_runs = (
        all_runs.filter(pl.col("city").eq(multi_k_stratum.value))
        .sort("created_at", descending=True)
        .select(["run_id", "created_at", "notes", "n_clusters"])
    )

    mo.stop(
        _stratum_runs.is_empty(),
        mo.callout(mo.md(f"No runs found for {multi_k_stratum.value}."), kind="warn"),
    )

    _frames = []
    for _row in _stratum_runs.iter_rows(named=True):
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
                f"No multi-K data for any {multi_k_stratum.value} run. "
                "Re-run the pipeline to populate."
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
        title=f"Silhouette vs K — {multi_k_stratum.value}",
        labels={"k": "Number of clusters (K)", "silhouette": "Silhouette Score", "run": "Run"},
    )
    _fig
    return


@app.cell
def _(all_runs, fetch_run_multi_k, mo, multi_k_stratum, pl, px):
    mo.stop(not multi_k_stratum.value)

    _stratum_runs = (
        all_runs.filter(pl.col("city").eq(multi_k_stratum.value))
        .sort("created_at", descending=True)
        .select(["run_id", "created_at", "notes", "n_clusters"])
    )

    mo.stop(_stratum_runs.is_empty())

    _frames = []
    for _row in _stratum_runs.iter_rows(named=True):
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
                f"No affinity Z-score data for {multi_k_stratum.value}. "
                "Re-run the pipeline to populate."
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
        title=f"Affinity Z-score vs K — {multi_k_stratum.value}",
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
def _(all_runs, fetch_run_leiden, mo, multi_k_stratum, pl, px):
    mo.stop(not multi_k_stratum.value)

    _stratum_runs_leiden = (
        all_runs.filter(pl.col("city").eq(multi_k_stratum.value))
        .sort("created_at", descending=True)
        .select(["run_id", "created_at", "notes"])
    )

    mo.stop(_stratum_runs_leiden.is_empty())

    _frames_leiden = []
    for _row in _stratum_runs_leiden.iter_rows(named=True):
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
            mo.md(f"No Leiden data for {multi_k_stratum.value}. Re-run the pipeline to populate."),
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
        title=f"Leiden: communities vs resolution — {multi_k_stratum.value}",
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
    ## Multi-stratum comparison
    """)
    return


@app.cell
def _(latest_per_city, mo):
    _run_options = {
        f"{row['city']} (#{row['run_id']})": row["run_id"]
        for row in latest_per_city.iter_rows(named=True)
    }
    run_id_to_stratum = {
        row["run_id"]: row["city"] for row in latest_per_city.iter_rows(named=True)
    }

    cross_stratum_runs = mo.ui.multiselect(
        options=_run_options,
        value=list(_run_options.keys()),
        label="Strata",
    )
    cross_stratum_runs
    return cross_stratum_runs, run_id_to_stratum


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Affinity Z-score vs K
    """)
    return


@app.cell
def _(cross_stratum_runs, fetch_run_multi_k, mo, pl, px, run_id_to_stratum):
    mo.stop(not cross_stratum_runs.value)

    _frames = []
    for _run_id in cross_stratum_runs.value:
        _mk = fetch_run_multi_k(_run_id)
        if _mk.is_empty() or "affinity_zscore" not in _mk.columns:
            continue
        if _mk["affinity_zscore"].is_null().all():
            continue
        _stratum = run_id_to_stratum[_run_id]
        _frames.append(_mk.with_columns(pl.lit(_stratum).alias("stratum")))

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
        color="stratum",
        markers=True,
        title="Affinity Z-score vs K",
        labels={"k": "K", "affinity_zscore": "Affinity Z-score", "stratum": "Stratum"},
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
def _(cross_stratum_runs, fetch_run_leiden, mo, pl, px, run_id_to_stratum):
    mo.stop(not cross_stratum_runs.value)

    _frames = []
    for _run_id in cross_stratum_runs.value:
        _ld = fetch_run_leiden(_run_id)
        if _ld.is_empty():
            continue
        _stratum = run_id_to_stratum[_run_id]
        _frames.append(_ld.with_columns(pl.lit(_stratum).alias("stratum")))

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
        color="stratum",
        markers=True,
        log_x=True,
        title="Leiden: communities vs resolution",
        labels={
            "resolution": "Resolution (γ, log scale)",
            "n_communities": "Number of communities",
            "stratum": "Stratum",
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
    leiden_combined.filter(pl.col("resolution").eq(0.5)).with_columns(
        pl.col("n_communities").min().alias("min"),
        pl.col("n_communities").median().alias("median"),
        pl.col("n_communities").max().alias("max"),
    )
    return


@app.cell
def _(leiden_combined, pl):
    leiden_combined.filter(pl.col("resolution").eq(0.5)).with_columns(
        pl.col("n_communities").median()
    )
    return


if __name__ == "__main__":
    app.run()
