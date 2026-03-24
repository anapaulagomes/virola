import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json

    import marimo as mo
    import numpy as np
    import plotly.express as px
    import polars as pl
    from dotenv import load_dotenv

    return json, load_dotenv, mo, np, pl, px


@app.cell
def _():
    from vir.db import (
        fetch_cluster_assignments,
        fetch_cluster_profiles,
        fetch_runs,
        upsert_cluster_profile,
    )

    return (
        fetch_cluster_assignments,
        fetch_cluster_profiles,
        fetch_runs,
        upsert_cluster_profile,
    )


@app.cell
def _(load_dotenv):
    _ = load_dotenv()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # SNF Cluster Explorer

    Browse syndromes discovered by the SNF pipeline.
    Select a run, then explore individual clusters with their profiles,
    temporal signatures, demographic distributions and explanations.
    """)
    return


@app.cell
def _(fetch_runs, mo):
    runs_df = fetch_runs()
    if runs_df.is_empty():
        run_options = {}
    else:
        run_options = {
            f"#{r['run_id']} — {r['city'] or r['dataset_name']} "
            f"({r['year_min']}–{r['year_max']}) "
            f"K={r['snf_k']} n={r['n_clusters']}"
            + (f" [{r['notes']}]" if r["notes"] else "")
            + (f" ⏱ {r['duration_min']}min" if r["duration_min"] else ""): r["run_id"]
            for r in runs_df.iter_rows(named=True)
        }

    run_dropdown = mo.ui.dropdown(
        options=run_options,
        label="Select run",
        value=list(run_options.keys())[0] if run_options else None,
    )
    run_dropdown
    return run_dropdown, run_options, runs_df


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
def _(fetch_cluster_assignments, fetch_cluster_profiles, selected_run_id):
    assignments = fetch_cluster_assignments(selected_run_id)
    profiles = fetch_cluster_profiles(selected_run_id)
    cluster_labels = sorted(assignments["cluster_id"].unique().to_list())
    number_of_clusters = len(cluster_labels)
    return assignments, cluster_labels, number_of_clusters, profiles


@app.cell
def _(assignments, mo, number_of_clusters, pl, px, runs_df, selected_run_id):
    _run = runs_df.filter(pl.col("run_id") == selected_run_id)

    _cluster_sizes = (
        assignments.group_by("cluster_id")
        .agg(pl.len().alias("n_codes"))
        .sort("n_codes", descending=True)
    )
    _median_size = _cluster_sizes["n_codes"].median()
    _total_codes = assignments.height

    _cards = [
        mo.stat(label="Clusters", value=number_of_clusters, bordered=True),
        mo.stat(label="Total codes", value=_total_codes),
        mo.stat(label="Median cluster size", value=f"{_median_size:.0f} codes"),
        mo.stat(
            label="Silhouette",
            value=f"{_run['silhouette_score'][0]:.3f}" if _run["silhouette_score"][0] else "—",
        ),
        mo.stat(
            label="Affinity Z-score",
            value=f"{_run['affinity_zscore'][0]:.1f}" if _run["affinity_zscore"][0] else "—",
        ),
        mo.stat(
            label="Duration",
            value=f"{_run['duration_min'][0]} min" if _run["duration_min"][0] else "—",
        ),
    ]

    _hist = px.histogram(
        _cluster_sizes,
        x="n_codes",
        nbins=50,
        title="Cluster size distribution",
        labels={"n_codes": "Codes per cluster"},
    )

    mo.vstack(
        [
            mo.md("## Run Overview"),
            mo.hstack(_cards, widths="equal", align="center"),
            _hist,
        ]
    )
    return


@app.cell
def _(json, mo, np, pl, profiles, px, selected_run_id, upsert_cluster_profile):
    def _build_features(profiles_df):
        """Build temporal+demographic feature matrix from stored JSON profiles."""
        rows = []
        for row in profiles_df.iter_rows(named=True):
            temp_map = {}
            if row["temporal_signature"]:
                for item in json.loads(row["temporal_signature"]):
                    temp_map[item["week"]] = item["total"]
            temp_vec = [float(temp_map.get(w, 0)) for w in range(1, 53)]
            total_t = sum(temp_vec) or 1.0
            temp_norm = [v / total_t for v in temp_vec]

            demo_map = {}
            if row["demographic_profile"]:
                for item in json.loads(row["demographic_profile"]):
                    demo_map[item["sex_age"]] = item["proportion"]

            rows.append(
                {
                    "cluster_id": row["cluster_id"],
                    "n_codes": row["n_codes"] or 1,
                    "total_cases": row["total_cases"] or 0,
                    "label": row["llm_short_description"] or f"Cluster {row['cluster_id']}",
                    "temp_norm": temp_norm,
                    "demo_map": demo_map,
                }
            )

        all_demo_keys = sorted(set(k for r in rows for k in r["demo_map"]))
        X = []
        for r in rows:
            demo_vec = [r["demo_map"].get(k, 0.0) for k in all_demo_keys]
            X.append(r["temp_norm"] + demo_vec)

        return np.array(X, dtype=np.float32), rows

    _umap_fig = None

    if not profiles.is_empty():
        _has_coords = profiles["umap_x"].null_count() < profiles.height

        if _has_coords:
            _umap_df = profiles.select(
                [
                    "cluster_id",
                    "umap_x",
                    "umap_y",
                    "n_codes",
                    "total_cases",
                    pl.col("llm_short_description")
                    .fill_null(
                        pl.concat_str([pl.lit("Cluster "), pl.col("cluster_id").cast(pl.Utf8)])
                    )
                    .alias("label"),
                ]
            )
        else:
            _X, _rows = _build_features(profiles)
            if len(_rows) >= 4:
                import umap as _umap_lib

                _reducer = _umap_lib.UMAP(
                    n_components=2,
                    random_state=42,
                    n_neighbors=min(15, len(_rows) - 1),
                )
                _embedding = _reducer.fit_transform(_X)
                _umap_df = pl.DataFrame(
                    {
                        "cluster_id": [r["cluster_id"] for r in _rows],
                        "umap_x": _embedding[:, 0].tolist(),
                        "umap_y": _embedding[:, 1].tolist(),
                        "n_codes": [r["n_codes"] for r in _rows],
                        "total_cases": [r["total_cases"] for r in _rows],
                        "label": [r["label"] for r in _rows],
                    }
                )
                # Persist coordinates for future loads
                for _row in _umap_df.iter_rows(named=True):
                    upsert_cluster_profile(
                        selected_run_id,
                        _row["cluster_id"],
                        umap_x=float(_row["umap_x"]),
                        umap_y=float(_row["umap_y"]),
                    )
            else:
                _umap_df = pl.DataFrame()

        if not _umap_df.is_empty():

            def _top_codes_text(tc_json, n=5):
                if not tc_json:
                    return ""
                codes = json.loads(tc_json)
                return "<br>".join(
                    f"{c['code']} ({c['code_type']}): {c.get('description', '')}"
                    for c in codes[:n]
                )

            _desc_df = profiles.with_columns(
                pl.col("top_codes")
                .map_elements(_top_codes_text, return_dtype=pl.Utf8)
                .alias("top_codes_text"),
                pl.col("total_cases").fill_null(0),
            ).select(["cluster_id", "top_codes_text", "total_cases"])

            _umap_df = _umap_df.join(_desc_df, on="cluster_id", how="left").with_columns(
                pl.col("top_codes_text").fill_null("")
            )

            _umap_fig = px.scatter(
                _umap_df,
                x="umap_x",
                y="umap_y",
                # color="cluster_id",
                size="n_codes",
                hover_name="label",
                custom_data=["cluster_id", "n_codes", "total_cases", "top_codes_text"],
                title="UMAP — cluster map (temporal + demographic features)",
                labels={"cluster_id": "Cluster", "n_codes": "Codes", "total_cases": "Cases"},
                color_continuous_scale="Turbo",
            )
            _umap_fig.update_traces(
                hovertemplate=(
                    "<b>%{hovertext}</b><br>"
                    "Cluster %{customdata[0]} · %{customdata[1]} codes · %{customdata[2]:,} cases<br>"
                    "<br>"
                    "<b>Top codes:</b><br>%{customdata[3]}"
                    "<extra></extra>"
                ),
                marker=dict(opacity=0.8, line=dict(width=0)),
            )
            _umap_fig.update_layout(coloraxis_showscale=False)

    mo.vstack(
        [
            mo.md("## Cluster Map (UMAP)"),
            _umap_fig
            if _umap_fig
            else mo.callout(
                mo.md("UMAP requires at least 4 clusters with stored profiles."), kind="info"
            ),
        ]
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
        label="Cluster",
    )
    cluster_dropdown
    return (cluster_dropdown,)


@app.cell
def _(cluster_dropdown, json, pl, profiles):
    _profile = profiles.filter(pl.col("cluster_id") == cluster_dropdown.value)

    if not _profile.is_empty():
        _row = _profile.row(0, named=True)
        profile_top_codes = json.loads(_row["top_codes"]) if _row["top_codes"] else []
        profile_temporal = (
            json.loads(_row["temporal_signature"]) if _row["temporal_signature"] else []
        )
        profile_demographic = (
            json.loads(_row["demographic_profile"]) if _row["demographic_profile"] else []
        )
        profile_explanation = json.loads(_row["explanation"]) if _row["explanation"] else None
        profile_llm_short = _row["llm_short_description"] or ""
        profile_llm_desc = _row["llm_description"] or ""
        profile_n_codes = _row["n_codes"]
        profile_total_cases = _row["total_cases"]
    else:
        profile_top_codes = []
        profile_temporal = []
        profile_demographic = []
        profile_explanation = None
        profile_llm_short = ""
        profile_llm_desc = ""
        profile_n_codes = None
        profile_total_cases = None
    return (
        profile_demographic,
        profile_explanation,
        profile_llm_desc,
        profile_llm_short,
        profile_n_codes,
        profile_temporal,
        profile_top_codes,
        profile_total_cases,
    )


@app.cell
def _(
    assignments,
    cluster_dropdown,
    mo,
    pl,
    profile_explanation,
    profile_llm_desc,
    profile_llm_short,
    profile_n_codes,
    profile_total_cases,
):
    _n_codes = (
        profile_n_codes
        or assignments.filter(pl.col("cluster_id") == cluster_dropdown.value).height
    )
    _total = profile_total_cases or 0

    _driver_card = mo.md("*Run `vir results explain` to see view drivers.*")
    if profile_explanation:
        _vs = profile_explanation.get("view_similarity", {})
        _driver = profile_explanation.get("primary_driver", "—")
        _central = profile_explanation.get("central_code", {})
        _driver_card = mo.vstack(
            [
                mo.md(f"**Primary driver:** `{_driver}`"),
                mo.md(
                    f"Clinical `{_vs.get('clinical', 0):.3f}` · "
                    f"Demographic `{_vs.get('demographic', 0):.3f}` · "
                    f"Temporal `{_vs.get('temporal', 0):.3f}`"
                ),
                mo.md(
                    f"**Central code:** {_central.get('code', '')} [{_central.get('code_type', '')}] "
                    f"{_central.get('description', '')}"
                ),
            ]
        )

    _llm_card = (
        mo.stat(label="Syndrome (LLM)", value=profile_llm_short, caption=profile_llm_desc)
        if profile_llm_short
        else mo.md("*Run `vir results llm-descriptions` to generate.*")
    )

    _cards = [
        mo.stat(label="Cluster", value=cluster_dropdown.value, bordered=True),
        mo.stat(label="Unique codes", value=_n_codes),
        mo.stat(label="Total cases", value=f"{_total:,}"),
        _llm_card,
    ]

    mo.vstack(
        [
            mo.md("## Cluster Detail"),
            mo.hstack(_cards, widths="equal", align="center"),
            mo.md("### Explanation"),
            _driver_card,
        ]
    )
    return


@app.cell
def _(mo, pl, profile_temporal, px):
    if profile_temporal:
        _temp_df = pl.DataFrame(profile_temporal)
        _fig = px.bar(
            _temp_df,
            x="week",
            y="total",
            title="Weekly case counts (all years aggregated)",
            labels={"week": "Epidemiological week", "total": "Cases"},
        )
    else:
        _fig = None

    mo.vstack(
        [
            mo.md("### Temporal Signature"),
            _fig if _fig else mo.md("*No temporal data — run `vir results profiles`.*"),
        ]
    )
    return


@app.cell
def _(mo, pl, profile_demographic, px):
    if profile_demographic:
        _demo_df = pl.DataFrame(profile_demographic).with_columns(
            [
                pl.col("sex_age").str.split("-").list.get(0).alias("sex"),
                pl.col("sex_age").str.split("-").list.get(1).alias("age_group"),
            ]
        )

        _sex_df = _demo_df.group_by("sex").agg(pl.col("proportion").sum()).sort("sex")
        _age_df = _demo_df.group_by("age_group").agg(pl.col("proportion").sum()).sort("age_group")

        _fig_sex = px.bar(
            _sex_df, x="sex", y="proportion", title="Sex", labels={"proportion": "Proportion"}
        )
        _fig_sex.update_layout(yaxis_tickformat=".0%")

        _fig_age = px.bar(
            _age_df,
            x="age_group",
            y="proportion",
            title="Age group",
            labels={"proportion": "Proportion"},
        )
        _fig_age.update_layout(yaxis_tickformat=".0%")

        _demo_content = mo.hstack([_fig_sex, _fig_age])
    else:
        _demo_content = mo.md("*No demographic data — run `vir results profiles`.*")

    mo.vstack(
        [
            mo.md("### Demographics"),
            _demo_content,
        ]
    )
    return


@app.cell
def _(assignments, cluster_dropdown, mo, pl, profile_top_codes, px):
    _cluster_assignments = assignments.filter(pl.col("cluster_id") == cluster_dropdown.value)

    # Treemap from top_codes (DB)
    if profile_top_codes:
        _top_df = pl.DataFrame(profile_top_codes).join(
            _cluster_assignments.select(["code", "code_type", "description"]),
            on=["code", "code_type"],
            how="left",
        )
        _treemap = px.treemap(
            _top_df,
            path=["code_type", "code"],
            values="total",
            hover_data=["description"],
            title="Code composition (top codes)",
        )
    else:
        _treemap = px.treemap(
            _cluster_assignments,
            path=["code_type", "code"],
            title="Code composition (all codes, no case counts)",
        )

    # Member table: all codes from assignments, annotated with case counts from top_codes
    if profile_top_codes:
        _top_lookup = {(r["code"], r["code_type"]): r["total"] for r in profile_top_codes}
        _member_table = _cluster_assignments.with_columns(
            pl.struct(["code", "code_type"])
            .map_elements(
                lambda x: _top_lookup.get((x["code"], x["code_type"])),
                return_dtype=pl.Int64,
            )
            .alias("total_cases")
        ).sort("total_cases", descending=True, nulls_last=True)
    else:
        _member_table = _cluster_assignments.with_columns(
            pl.lit(None).cast(pl.Int64).alias("total_cases")
        )

    mo.vstack(
        [
            mo.md("### Code Composition"),
            _treemap,
            mo.md("### Member Codes"),
            mo.ui.table(
                _member_table.select(
                    ["cluster_id", "code", "code_type", "description", "total_cases"]
                ),
                selection=None,
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
