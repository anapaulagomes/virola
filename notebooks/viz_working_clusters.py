import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import plotly.express as px
    import plotly.io as plotly_io
    import polars as pl
    from dotenv import load_dotenv

    plotly_io.templates.default = "plotly_white"
    return load_dotenv, mo, pl, px


@app.cell
def _():
    from vir.adapters import ADAPTERS
    from vir.config import DATA_DIR
    from vir.db import fetch_cluster_assignments, fetch_cluster_profiles, fetch_runs
    from vir.osi import build_osi_payload, osi_payload_as_json

    return (
        ADAPTERS,
        DATA_DIR,
        build_osi_payload,
        fetch_cluster_assignments,
        fetch_cluster_profiles,
        fetch_runs,
        osi_payload_as_json,
    )


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
def _(ADAPTERS, adapter, args):
    adapter_name = args.get("adapter", adapter.value)
    active_adapter = ADAPTERS[adapter_name]
    viz_palettes = active_adapter.viz_palettes
    SEX_COLOR = viz_palettes["sex"]
    AGE_GROUP_COLOR = viz_palettes["age_group"]
    code_type_color_lowercase = viz_palettes["code_type"]
    return AGE_GROUP_COLOR, SEX_COLOR, adapter_name, code_type_color_lowercase


@app.cell
def _(load_dotenv):
    _ = load_dotenv()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Syndrome Discovery

    Workspace for epidemiologists to inspect candidate clusters discovered by
    the SNF pipeline and select the ones that should feed downstream syndromic
    surveillance systems.
    """)
    return


@app.cell
def _(fetch_runs, mo):
    runs_dataframe = fetch_runs()
    if runs_dataframe.is_empty():
        run_options = {}
    else:
        run_options = {
            f"#{r['run_id']} — {r['city'] or r['dataset_name']} "
            f"({r['year_min']}–{r['year_max']}) "
            f"K={r['snf_k']} n={r['n_clusters']}"
            + (f" [{r['notes']}]" if r["notes"] else "")
            + (f" ⏱ {r['duration_min']}min" if r["duration_min"] else ""): r["run_id"]
            for r in runs_dataframe.iter_rows(named=True)
        }

    run_dropdown = mo.ui.dropdown(
        options=run_options,
        label="Select run",
        value=list(run_options.keys())[0] if run_options else None,
    )
    mo.right(run_dropdown)
    return run_dropdown, run_options, runs_dataframe


@app.cell
def _(mo, run_options):
    mo.stop(
        not run_options,
        mo.callout(mo.md("No runs found. Run `vir model snf` first."), kind="warn"),
    )
    return


@app.cell
def _(run_dropdown):
    selected_run_id = run_dropdown.value
    return (selected_run_id,)


@app.cell
def _(fetch_cluster_assignments, selected_run_id):
    assignments = fetch_cluster_assignments(selected_run_id)
    cluster_labels = sorted(assignments["cluster_id"].unique().to_list())
    return assignments, cluster_labels


@app.cell
def _(fetch_cluster_profiles, selected_run_id):
    cluster_descriptions = fetch_cluster_profiles(selected_run_id).select(
        ["cluster_id", "llm_short_description"]
    )
    return (cluster_descriptions,)


@app.cell
def _(DATA_DIR):
    DATA_DIR
    return


@app.cell
def _(DATA_DIR, pl, selected_run_id):
    run_dataset_path = DATA_DIR / "runs" / f"{selected_run_id}.parquet"
    if run_dataset_path.exists():
        run_dataset = pl.read_parquet(run_dataset_path)
    else:
        run_dataset = pl.DataFrame()
    return run_dataset, run_dataset_path


@app.cell
def _(mo, pl, run_dataset, run_dataset_path, runs_dataframe, selected_run_id):
    selected_run_row = runs_dataframe.filter(pl.col("run_id").eq(selected_run_id))
    selected_dataset_name = (
        selected_run_row["dataset_name"][0] if not selected_run_row.is_empty() else None
    )
    selected_stratum = selected_run_row["city"][0] if not selected_run_row.is_empty() else None
    rerun_target = selected_stratum or selected_dataset_name or "<stratum>"
    mo.stop(
        run_dataset.is_empty(),
        mo.callout(
            mo.md(
                f"Per-record parquet sidecar not found for run **{selected_run_id}**.\n\n"
                f"- Expected at `{run_dataset_path}`\n"
                f"- Cause: `vir model snf` writes this sidecar at the end of `save_run`. "
                f"If the SNF step crashed or was interrupted after the run row was "
                f"inserted but before the parquet write, the row stays orphaned.\n\n"
                f"Recover by re-running:\n\n"
                f"```bash\nvir pipeline run {rerun_target} --year-min … --year-max …\n```\n\n"
                f"or delete the orphan run: `vir results delete --id {selected_run_id}`."
            ),
            kind="warn",
        ),
    )
    return


@app.cell
def _(mo):
    terminology_search = mo.ui.text(
        placeholder="Search by code or description (e.g. dengue, J10, influenza)...",
        label="Search",
        full_width=True,
    )
    terminology_search
    return (terminology_search,)


@app.cell
def _(assignments, cluster_labels, pl, terminology_search):
    _term = terminology_search.value.strip().lower()
    if _term:
        _matching = (
            assignments.filter(
                pl.col("code").str.to_lowercase().str.contains(_term)
                | pl.col("description").str.to_lowercase().str.contains(_term)
            )["cluster_id"]
            .unique()
            .to_list()
        )
        filtered_cluster_labels = sorted(_matching)
    else:
        filtered_cluster_labels = cluster_labels
    return (filtered_cluster_labels,)


@app.cell
def _(filtered_cluster_labels, mo):
    cluster_dropdown = mo.ui.dropdown(
        options=filtered_cluster_labels,
        value=filtered_cluster_labels[0] if filtered_cluster_labels else None,
        label="Cluster ID",
    )
    cluster_dropdown
    return (cluster_dropdown,)


@app.cell
def _(assignments, cluster_dropdown, pl):
    cluster_assignments = assignments.filter(pl.col("cluster_id") == cluster_dropdown.value)
    return (cluster_assignments,)


@app.cell
def _(cluster_assignments, run_dataset):
    cluster_records = run_dataset.join(
        cluster_assignments.select(
            ["terminology_id", "code", "code_type", "description"]
        ).unique(),
        on=["terminology_id", "code", "code_type"],
        how="inner",
    )
    return (cluster_records,)


@app.cell
def _(cluster_dropdown, cluster_records, mo):
    _unique_codes = cluster_records["terminology_id"].n_unique()
    _total_cases = int(cluster_records["quantity"].sum() or 0)

    _cards = [
        mo.stat(label="Cluster", value=cluster_dropdown.value, bordered=True),
        mo.stat(label="Unique codes", value=_unique_codes),
        mo.stat(label="Total cases", value=f"{_total_cases:,}"),
    ]

    mo.vstack(
        [
            mo.md("## Cluster Detail"),
            mo.hstack(_cards, widths="equal", align="center"),
        ]
    )
    return


@app.cell
def _(SEX_COLOR, cluster_records, pl, px):
    _sex_dataframe = (
        cluster_records.group_by("sex").agg(pl.col("quantity").sum().alias("cases")).sort("sex")
    )
    sex_chart = px.bar(
        _sex_dataframe,
        x="sex",
        y="cases",
        color="sex",
        labels={"cases": "Cases", "sex": "Sex"},
        color_discrete_map=SEX_COLOR,
    )
    return (sex_chart,)


@app.cell
def _(AGE_GROUP_COLOR, cluster_records, pl, px):
    _age_order = list(AGE_GROUP_COLOR.keys())
    _age_dataframe = (
        cluster_records.group_by("age_group")
        .agg(pl.col("quantity").sum().alias("cases"))
        .with_columns(
            pl.col("age_group").cast(pl.Enum(_age_order)).alias("age_group"),
        )
        .sort("age_group")
    )
    age_chart = px.bar(
        _age_dataframe,
        x="age_group",
        y="cases",
        color="age_group",
        labels={"cases": "Cases", "age_group": "Age group"},
        color_discrete_map=AGE_GROUP_COLOR,
        category_orders={"age_group": _age_order},
    )
    return (age_chart,)


@app.cell
def _(age_chart, mo, sex_chart):
    sex_chart.update_layout(showlegend=False)
    age_chart.update_layout(showlegend=False)

    mo.vstack(
        [
            mo.md("### Demographics"),
            mo.hstack(
                [mo.ui.plotly(sex_chart), mo.ui.plotly(age_chart)],
                widths="equal",
            ),
        ]
    )
    return


@app.cell
def _(cluster_records, pl):
    member_summary = (
        cluster_records.group_by(["terminology_id", "code", "code_type", "description"])
        .agg(pl.col("quantity").sum().alias("total_cases"))
        .sort("total_cases", descending=True)
    )
    return (member_summary,)


@app.cell
def _(code_type_color_lowercase, member_summary, px):
    treemap = px.treemap(
        member_summary,
        path=["code_type", "code"],
        values="total_cases",
        color="code_type",
        color_discrete_map=code_type_color_lowercase,
        hover_data=["description"],
    )
    return (treemap,)


@app.cell
def _(mo, treemap):
    mo.vstack(
        [
            mo.md("### Code Composition"),
            mo.ui.plotly(treemap),
        ]
    )
    return


@app.cell
def _(member_summary, mo):
    member_table = mo.ui.table(
        member_summary.select(["code", "code_type", "description", "total_cases"]),
        selection="multi",
        label="Tick the codes that should be included in the exported syndrome.",
    )

    mo.vstack(
        [
            mo.md("### Member Codes"),
            mo.md(
                "Select the rows that compose the syndrome candidate; the export "
                "below will only include checked codes."
            ),
            member_table,
        ]
    )
    return (member_table,)


@app.cell
def _(cluster_records, pl):
    timeline_dataframe = cluster_records.with_columns(
        (
            pl.col("year").cast(pl.Utf8)
            + "-"
            + pl.col("week").cast(pl.Int64).cast(pl.Utf8).str.zfill(2)
            + "-1"
        )
        .str.to_date(format="%G-%V-%u")
        .alias("week_start"),
        pl.concat_str(["code", "code_type"], separator=":").alias("member"),
    )
    return (timeline_dataframe,)


@app.cell
def _(pl, px, timeline_dataframe):
    _aggregated = (
        timeline_dataframe.group_by("week_start")
        .agg(pl.col("quantity").sum().alias("cases"))
        .sort("week_start")
    )
    total_timeline_chart = px.line(
        _aggregated,
        x="week_start",
        y="cases",
        labels={"week_start": "Week", "cases": "Cases"},
    )
    return (total_timeline_chart,)


@app.cell
def _(pl, px, timeline_dataframe):
    _per_member = (
        timeline_dataframe.group_by(["member", "week_start"])
        .agg(pl.col("quantity").sum().alias("cases"))
        .sort(["member", "week_start"])
    )
    per_member_timeline_chart = px.line(
        _per_member,
        x="week_start",
        y="cases",
        color="member",
        labels={"week_start": "Week", "cases": "Cases", "member": "Member"},
    )
    return (per_member_timeline_chart,)


@app.cell
def _(cluster_records, pl, px):
    _epi_week_dataframe = (
        cluster_records.group_by("week")
        .agg(pl.col("quantity").sum().alias("cases"))
        .join(
            pl.DataFrame({"week": list(range(1, 53))}, schema={"week": pl.Int8}),
            on="week",
            how="right",
        )
        .with_columns(pl.col("cases").fill_null(0))
        .sort("week")
    )
    epi_week_chart = px.bar(
        _epi_week_dataframe,
        x="week",
        y="cases",
        labels={"week": "Epidemiological week", "cases": "Cases"},
    )
    return (epi_week_chart,)


@app.cell
def _(epi_week_chart, mo, per_member_timeline_chart, total_timeline_chart):
    epi_week_chart.update_xaxes(tickmode="linear", dtick=4)

    mo.vstack(
        [
            mo.md("### Temporal"),
            mo.ui.tabs(
                {
                    "Total over time": mo.ui.plotly(total_timeline_chart),
                    "Per member over time": mo.ui.plotly(per_member_timeline_chart),
                    "Epidemiological week (1–52)": mo.ui.plotly(epi_week_chart),
                }
            ),
        ]
    )
    return


@app.cell
def _(adapter_name, mo):
    osi_supported = adapter_name == "datasus"
    export_button = (
        mo.ui.run_button(label="Export selected codes to OSI (JSON)") if osi_supported else None
    )
    if osi_supported:
        export_button
    else:
        mo.callout(
            mo.md(
                f"OSI export is currently only wired up for the DATASUS adapter; "
                f"the active adapter is `{adapter_name}`."
            ),
            kind="info",
        )
    return export_button, osi_supported


@app.cell
def _(
    build_osi_payload,
    cluster_descriptions,
    cluster_dropdown,
    export_button,
    member_table,
    mo,
    osi_payload_as_json,
    osi_supported,
    pl,
    runs_dataframe,
    selected_run_id,
):
    mo.stop(not osi_supported)

    if not export_button.value:
        _export_content = mo.md(
            "*Tick member codes above and click the export button to preview the OSI payload.*"
        )
    else:
        _selection = member_table.value
        if _selection is None or _selection.is_empty():
            _export_content = mo.callout(
                mo.md("Select at least one member code before exporting."), kind="warn"
            )
        else:
            _description_row = cluster_descriptions.filter(
                pl.col("cluster_id") == cluster_dropdown.value
            )
            _llm_short_description = (
                _description_row["llm_short_description"][0]
                if not _description_row.is_empty()
                else None
            )
            _dataset_name = runs_dataframe.filter(pl.col("run_id") == selected_run_id)[
                "dataset_name"
            ][0]
            _title = (
                _llm_short_description
                or f"Cluster {cluster_dropdown.value} — run {selected_run_id}"
            )
            _description = (
                f"Dataset: {_dataset_name} · "
                f"Cluster: {cluster_dropdown.value} · "
                f"Run: {selected_run_id}"
            )
            _payload = build_osi_payload(
                members=_selection.iter_rows(named=True),
                title=_title,
                description=_description,
            )
            _export_content = mo.vstack(
                [
                    mo.callout(
                        mo.md(
                            f"Prepared OSI payload with **{_selection.height}** member code(s)."
                        ),
                        kind="success",
                    ),
                    mo.ui.code_editor(
                        value=osi_payload_as_json(_payload),
                        language="json",
                        disabled=True,
                    ),
                ]
            )

    mo.vstack(
        [
            mo.md("### Export to OSI"),
            _export_content,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
