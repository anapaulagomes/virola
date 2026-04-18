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
    from vir.config import DATA_DIR, REFERENCES_DIR
    from vir.db import fetch_cluster_assignments, fetch_runs
    from vir.features import CODE_TYPES_MAPPING
    from vir.viz import AGE_GROUP_COLOR, SEX_COLOR

    return (
        AGE_GROUP_COLOR,
        CODE_TYPES_MAPPING,
        DATA_DIR,
        REFERENCES_DIR,
        SEX_COLOR,
        fetch_cluster_assignments,
        fetch_runs,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Cross-City Cluster Consistency

    For each cluster in city A, find the most similar cluster in city B using
    Jaccard similarity on `terminology_id` sets.

    **Goal**: identify clusters that are reproducible across cities — these are
    the strongest candidates for genuine syndromic signals, independent of
    local data quirks.

    **Calibration**: arboviroses (dengue/chikungunya) serves as the anchor
    syndrome — its cross-city Jaccard sets the threshold τ for "consistent."
    """)
    return


@app.cell
def _(fetch_runs, mo, pl):
    runs_df = fetch_runs()

    mo.stop(
        runs_df.is_empty(),
        mo.callout(mo.md("No runs found. Run `vir model snf` first."), kind="warn"),
    )

    latest_per_city = (
        runs_df.filter(pl.col("city").is_not_null())
        .sort("created_at", descending=True)
        .group_by("city")
        .first()
        .sort("city")
    )

    city_options = latest_per_city["city"].to_list()

    city_selector = mo.ui.multiselect(
        options=city_options,
        value=city_options[:2] if len(city_options) >= 2 else city_options,
        label="Cities to compare (select 2 or more)",
    )
    city_selector
    return city_selector, latest_per_city


@app.cell
def _(city_selector, fetch_cluster_assignments, latest_per_city, mo, pl):
    mo.stop(
        len(city_selector.value) < 2,
        mo.callout(mo.md("Select at least 2 cities."), kind="warn"),
    )

    selected_cities = city_selector.value

    city_run_map = {
        row["city"]: row["run_id"]
        for row in latest_per_city.iter_rows(named=True)
        if row["city"] in selected_cities
    }

    frames = []
    for city_name, run_id in city_run_map.items():
        df_city = fetch_cluster_assignments(run_id).with_columns(
            pl.lit(city_name).alias("city"),
            pl.lit(run_id).alias("run_id"),
        )
        frames.append(df_city)

    all_assignments = pl.concat(frames)

    mo.md(f"""
    **Loaded assignments for {len(selected_cities)} cities**

    | City | Run ID | Clusters | Codes |
    |------|--------|----------|-------|
    {
        "\n".join(
            [
                f"| {row['city']} | {city_run_map[row['city']]} | {row['n_clusters']} | {row['n_codes']} |"
                for row in (
                    all_assignments.group_by("city")
                    .agg(
                        pl.col("cluster_id").n_unique().alias("n_clusters"),
                        pl.col("terminology_id").n_unique().alias("n_codes"),
                    )
                    .sort("city")
                    .iter_rows(named=True)
                )
            ]
        )
    }
    """)
    return all_assignments, city_run_map, selected_cities


@app.cell
def _(DATA_DIR, city_run_map, mo, pl):
    run_datasets = {}
    missing_datasets = []
    for _city, _run_id in city_run_map.items():
        _path = DATA_DIR / "runs" / f"{_run_id}.parquet"
        if _path.exists():
            run_datasets[_city] = pl.read_parquet(_path)
        else:
            missing_datasets.append(_city)

    if missing_datasets:
        mo.callout(
            mo.md(
                f"Run datasets not found for: {', '.join(missing_datasets)}. "
                "Re-run the pipeline so per-record data is available for profiles."
            ),
            kind="warn",
        )
    return (run_datasets,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Jaccard Computation

    For every pair of cities, compute the best-match Jaccard for each cluster:

    ```
    Jaccard(A_i, B_j) = |A_i ∩ B_j| / |A_i ∪ B_j|
    best_match(A_i, B) = max_j Jaccard(A_i, B_j)
    ```
    """)
    return


@app.cell
def _(all_assignments, pl, selected_cities):
    def compute_jaccard_best_matches(
        assignments: pl.DataFrame,
        city_a: str,
        city_b: str,
    ) -> pl.DataFrame:
        a = (
            assignments.filter(pl.col("city") == city_a)
            .select(["cluster_id", "terminology_id"])
            .rename({"cluster_id": "cluster_a"})
        )

        b = (
            assignments.filter(pl.col("city") == city_b)
            .select(["cluster_id", "terminology_id"])
            .rename({"cluster_id": "cluster_b"})
        )

        intersection = (
            a.join(b, on="terminology_id", how="inner")
            .group_by(["cluster_a", "cluster_b"])
            .agg(pl.len().alias("intersection"))
        )

        size_a = a.group_by("cluster_a").agg(pl.len().alias("size_a"))
        size_b = b.group_by("cluster_b").agg(pl.len().alias("size_b"))

        result = (
            intersection.join(size_a, on="cluster_a")
            .join(size_b, on="cluster_b")
            .with_columns(
                (
                    pl.col("intersection")
                    / (pl.col("size_a") + pl.col("size_b") - pl.col("intersection"))
                ).alias("jaccard")
            )
        )

        return (
            result.sort("jaccard", descending=True)
            .group_by("cluster_a")
            .first()
            .with_columns(
                pl.lit(city_a).alias("city_a"),
                pl.lit(city_b).alias("city_b"),
            )
            .sort("cluster_a")
        )

    pair_results = []
    cities = selected_cities
    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            ab = compute_jaccard_best_matches(all_assignments, cities[i], cities[j])
            ba = compute_jaccard_best_matches(all_assignments, cities[j], cities[i])
            pair_results.extend([ab, ba])

    jaccard_df = pl.concat(pair_results)
    jaccard_df
    return compute_jaccard_best_matches, jaccard_df


@app.cell
def _(mo):
    mo.md(r"""
    ## Arbovirus Calibration

    Arboviroses (dengue/chikungunya) is the anchor syndrome.
    Its cross-city Jaccard sets τ: *"consistent in the same sense that
    dengue/chikungunya is consistent across cities."*
    """)
    return


@app.cell
def _(CODE_TYPES_MAPPING, REFERENCES_DIR, all_assignments, pl):
    _aesop_dir = REFERENCES_DIR / "aesop"

    anchor_arbovirus = (
        pl.read_csv(
            _aesop_dir / "code_list_arbovirus_apr2024.csv",
            separator=";",
            encoding="utf-8-sig",
        )
        .rename({"Type": "raw_type", "Code": "code"})
        .with_columns(
            pl.col("raw_type").replace(CODE_TYPES_MAPPING).alias("code_type"),
            pl.col("code").str.strip_chars(),
        )
        .select("code", "code_type")
        .unique()
    )

    arbo_hits = all_assignments.join(anchor_arbovirus, on=["code", "code_type"], how="inner")

    arbo_cluster_per_city = (
        arbo_hits.group_by(["city", "cluster_id"])
        .agg(pl.len().alias("ref_codes"))
        .sort("ref_codes", descending=True)
        .group_by("city")
        .first()
        .rename({"cluster_id": "arbo_cluster"})
    )

    arbo_cluster_per_city
    return (arbo_cluster_per_city,)


@app.cell
def _(
    all_assignments,
    arbo_cluster_per_city,
    compute_jaccard_best_matches,
    pl,
    selected_cities,
):
    arbo_jaccards = []

    for _i in range(len(selected_cities)):
        for _j in range(_i + 1, len(selected_cities)):
            _ca, _cb = selected_cities[_i], selected_cities[_j]

            _arbo_a = arbo_cluster_per_city.filter(pl.col("city") == _ca)
            _arbo_b = arbo_cluster_per_city.filter(pl.col("city") == _cb)

            if _arbo_a.is_empty() or _arbo_b.is_empty():
                continue

            _cluster_a = _arbo_a["arbo_cluster"][0]
            _best = compute_jaccard_best_matches(all_assignments, _ca, _cb)
            _row = _best.filter(pl.col("cluster_a") == _cluster_a)

            if not _row.is_empty():
                arbo_jaccards.append(
                    {
                        "city_a": _ca,
                        "city_b": _cb,
                        "cluster_a": int(_cluster_a),
                        "cluster_b": int(_row["cluster_b"][0]),
                        "jaccard": float(_row["jaccard"][0]),
                    }
                )

    arbo_jaccard_df = pl.DataFrame(arbo_jaccards) if arbo_jaccards else pl.DataFrame()
    arbo_jaccard_df
    return (arbo_jaccard_df,)


@app.cell
def _(arbo_jaccard_df, mo):
    if arbo_jaccard_df.is_empty():
        tau = 0.2
        mo.callout(mo.md(f"No arbovirus matches found. Using default τ = {tau}"), kind="warn")
    else:
        tau = float(arbo_jaccard_df["jaccard"].median())
        mo.callout(
            mo.md(f"""
            **Arbovirus anchor Jaccard**

            Median across city pairs: **τ = {tau:.3f}**
            Range: {arbo_jaccard_df["jaccard"].min():.3f} – {arbo_jaccard_df["jaccard"].max():.3f}

            Clusters with best-match Jaccard ≥ τ are *consistent in the same sense
            that arboviroses is consistent across cities.*
            """),
            kind="success",
        )
    return (tau,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Distribution of Best-Match Jaccard
    """)
    return


@app.cell
def _(jaccard_df, mo, px, tau):
    _fig = px.histogram(
        jaccard_df,
        x="jaccard",
        facet_col="city_a",
        nbins=40,
        title="Best-match Jaccard distribution per city",
        labels={"jaccard": "Jaccard (best match in paired city)"},
    )
    _fig.add_vline(
        x=tau,
        line_dash="dash",
        line_color="red",
        annotation_text=f"τ = {tau:.3f} (arbovirus anchor)",
    )
    _fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    _fig.update_layout(height=350)

    _pct_above = jaccard_df.filter(jaccard_df["jaccard"] >= tau).height / jaccard_df.height * 100

    mo.vstack(
        [
            _fig,
            mo.stat(
                label=f"Clusters with Jaccard ≥ τ ({tau:.3f})",
                value=f"{_pct_above:.1f}%",
                caption="across all city pairs",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Consistent Clusters — Full View

    Clusters where the best match in **every** paired city exceeds τ.
    """)
    return


@app.cell
def _(all_assignments, jaccard_df, mo, pl, tau):
    consistency = (
        jaccard_df.group_by(["city_a", "cluster_a"])
        .agg(
            pl.col("jaccard").min().alias("min_jaccard"),
            pl.col("jaccard").median().alias("median_jaccard"),
            pl.len().alias("n_pairs"),
        )
        .with_columns((pl.col("min_jaccard") >= tau).alias("consistent"))
        .sort("median_jaccard", descending=True)
    )

    n_consistent = consistency.filter(pl.col("consistent")).height
    n_total = consistency.height

    consistent_clusters = (
        consistency.filter(pl.col("consistent"))
        .join(
            all_assignments.group_by(["city", "cluster_id"])
            .agg(pl.len().alias("n_codes"))
            .rename({"city": "city_a", "cluster_id": "cluster_a"}),
            on=["city_a", "cluster_a"],
            how="left",
        )
        .sort(["city_a", "median_jaccard"], descending=[False, True])
    )

    mo.vstack(
        [
            mo.stat(
                label="Consistent clusters",
                value=f"{n_consistent} / {n_total}",
                caption=f"min Jaccard ≥ τ ({tau:.3f}) across all paired cities",
                bordered=True,
            ),
            mo.ui.table(
                consistent_clusters.select(
                    ["city_a", "cluster_a", "n_codes", "min_jaccard", "median_jaccard", "n_pairs"]
                ),
                selection=None,
            ),
        ]
    )
    return (consistent_clusters,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Cluster Inspector
    """)
    return


@app.cell
def _(consistent_clusters, mo):
    if consistent_clusters.is_empty():
        cluster_selector = mo.md("*No consistent clusters found with current τ.*")
    else:
        _opts = {
            f"{row['city_a']} — cluster {row['cluster_a']} ({row['n_codes']} codes, median J={row['median_jaccard']:.3f})": (
                row["city_a"],
                row["cluster_a"],
            )
            for row in consistent_clusters.iter_rows(named=True)
        }
        cluster_selector = mo.ui.dropdown(
            options=_opts,
            label="Inspect cluster",
            value=list(_opts.keys())[0],
        )
    cluster_selector
    return (cluster_selector,)


@app.cell
def _(
    all_assignments,
    cluster_selector,
    consistent_clusters,
    jaccard_df,
    mo,
    pl,
    px,
):
    mo.stop(consistent_clusters.is_empty())
    mo.stop(cluster_selector.value is None)

    _selected_city, _selected_cluster = cluster_selector.value

    _pairs = jaccard_df.filter(
        (pl.col("city_a") == _selected_city) & (pl.col("cluster_a") == _selected_cluster)
    )
    _fig_jaccard = px.bar(
        _pairs,
        x="city_b",
        y="jaccard",
        title="Best-match Jaccard across cities",
        labels={"jaccard": "Jaccard", "city_b": "City"},
        color="jaccard",
        color_continuous_scale="Greens",
    )
    _fig_jaccard.update_layout(coloraxis_showscale=False, height=300)

    _codes = (
        all_assignments.filter(
            (pl.col("city") == _selected_city) & (pl.col("cluster_id") == _selected_cluster)
        )
        .select(["code", "code_type", "description"])
        .sort(["code_type", "code"])
    )

    mo.vstack(
        [
            _fig_jaccard,
            mo.md(
                f"**Codes — {_selected_city} cluster {_selected_cluster}** ({len(_codes)} codes)"
            ),
            mo.ui.table(_codes, selection=None),
        ]
    )
    return


@app.cell
def _(
    AGE_GROUP_COLOR,
    SEX_COLOR,
    all_assignments,
    cluster_selector,
    consistent_clusters,
    mo,
    pl,
    px,
    run_datasets,
):
    mo.stop(consistent_clusters.is_empty())
    mo.stop(cluster_selector.value is None)

    _selected_city, _selected_cluster = cluster_selector.value

    mo.stop(
        _selected_city not in run_datasets,
        mo.callout(mo.md(f"No run dataset for {_selected_city}."), kind="warn"),
    )

    _members = (
        all_assignments.filter(
            (pl.col("city") == _selected_city) & (pl.col("cluster_id") == _selected_cluster)
        )
        .select(["terminology_id", "code", "code_type"])
        .unique()
    )

    _records = run_datasets[_selected_city].join(
        _members,
        on=["terminology_id", "code", "code_type"],
        how="inner",
    )

    _sex_df = _records.group_by("sex").agg(pl.col("quantity").sum().alias("cases")).sort("sex")
    _fig_sex = px.bar(
        _sex_df,
        x="sex",
        y="cases",
        color="sex",
        color_discrete_map=SEX_COLOR,
        title="Sex",
        labels={"cases": "Cases", "sex": ""},
    )
    _fig_sex.update_layout(showlegend=False, height=280)

    _age_df = (
        _records.group_by("age_group")
        .agg(pl.col("quantity").sum().alias("cases"))
        .sort("age_group")
    )
    _fig_age = px.bar(
        _age_df,
        x="age_group",
        y="cases",
        color="age_group",
        color_discrete_map=AGE_GROUP_COLOR,
        title="Age group",
        labels={"cases": "Cases", "age_group": ""},
    )
    _fig_age.update_layout(showlegend=False, height=280)

    mo.vstack(
        [
            mo.md("### Demographics"),
            mo.hstack([_fig_sex, _fig_age], widths="equal"),
        ]
    )
    return


@app.cell
def _(
    all_assignments,
    cluster_selector,
    consistent_clusters,
    mo,
    pl,
    px,
    run_datasets,
):
    mo.stop(consistent_clusters.is_empty())
    mo.stop(cluster_selector.value is None)

    _selected_city, _selected_cluster = cluster_selector.value

    mo.stop(_selected_city not in run_datasets)

    _members = (
        all_assignments.filter(
            (pl.col("city") == _selected_city) & (pl.col("cluster_id") == _selected_cluster)
        )
        .select(["terminology_id", "code", "code_type"])
        .unique()
    )

    _records = run_datasets[_selected_city].join(
        _members,
        on=["terminology_id", "code", "code_type"],
        how="inner",
    )

    # Epi week profile
    _epi_df = (
        _records.group_by("week")
        .agg(pl.col("quantity").sum().alias("cases"))
        .join(
            pl.DataFrame({"week": list(range(1, 53))}, schema={"week": pl.Int8}),
            on="week",
            how="right",
        )
        .with_columns(pl.col("cases").fill_null(0))
        .sort("week")
    )
    _fig_epi = px.bar(
        _epi_df,
        x="week",
        y="cases",
        title="Epi week profile (all years aggregated)",
        labels={"week": "Epidemiological week", "cases": "Cases"},
    )
    _fig_epi.update_xaxes(tickmode="linear", dtick=4)
    _fig_epi.update_layout(height=300)

    # Chronological timeline
    _timeline_df = (
        _records.with_columns(
            (
                pl.col("year").cast(pl.Utf8)
                + "-"
                + pl.col("week").cast(pl.Int64).cast(pl.Utf8).str.zfill(2)
                + "-1"
            )
            .str.to_date(format="%G-%V-%u")
            .alias("week_start")
        )
        .group_by("week_start")
        .agg(pl.col("quantity").sum().alias("cases"))
        .sort("week_start")
    )
    _fig_timeline = px.line(
        _timeline_df,
        x="week_start",
        y="cases",
        title="Timeline",
        labels={"week_start": "Week", "cases": "Cases"},
    )
    _fig_timeline.update_layout(height=300)

    mo.vstack(
        [
            mo.md("### Temporal"),
            mo.ui.tabs(
                {
                    "Epi week (1–52)": _fig_epi,
                    "Timeline": _fig_timeline,
                }
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
