import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import json

    import igraph as ig
    import marimo as mo
    import plotly.express as px
    import polars as pl

    return ig, json, mo, pl, px


@app.cell
def _():
    from publication.commons import POPULATION_RANGE_TRANSLATIONS, REGION_TRANSLATIONS
    from vir.config import DATA_DIR, REFERENCES_DIR
    from vir.db import fetch_cluster_assignments, fetch_runs
    from vir.features import CITY_NAMES, CODE_TYPES_MAPPING
    from vir.viz import AGE_GROUP_COLOR, SEX_COLOR, TABLEAU_COLORS

    return (
        AGE_GROUP_COLOR,
        CITY_NAMES,
        CODE_TYPES_MAPPING,
        DATA_DIR,
        POPULATION_RANGE_TRANSLATIONS,
        REFERENCES_DIR,
        REGION_TRANSLATIONS,
        SEX_COLOR,
        TABLEAU_COLORS,
        fetch_cluster_assignments,
        fetch_runs,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Cross-City Cluster Consistency

    For each cluster in city A, find the most similar cluster in city B using
    Jaccard similarity on `terminology_id` sets.

    **Goal**: identify clusters that are reproducible across cities. They are the strongest candidates for genuine syndromic signals, independent of local data quirks.

    **Calibration**: arboviroses (dengue/chikungunya) serves as the anchor syndrome given its cross-city Jaccard sets the threshold τ for "consistent."
    """)
    return


@app.cell
def _(CITY_NAMES, fetch_runs, mo):
    runs_df = fetch_runs()

    mo.stop(
        runs_df.is_empty(),
        mo.callout(mo.md("No runs found. Run `vir model snf` first."), kind="warn"),
    )

    city_selector = mo.ui.multiselect(
        options=CITY_NAMES,
        value=CITY_NAMES,
        label="Cities to compare (select 2 or more)",
    )
    city_selector
    return city_selector, runs_df


@app.cell
def _(pl, runs_df):
    latest_per_city = (
        runs_df.filter(pl.col("city").is_not_null())
        .sort("created_at", descending=True)
        .group_by("city")
        .first()
        .sort("city")
    )
    return (latest_per_city,)


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
def _(pl):
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
            size_a.join(intersection, on="cluster_a", how="left")
            .join(size_b, on="cluster_b", how="left")
            .with_columns(
                pl.col("intersection").fill_null(0),
                pl.col("size_b").fill_null(0),
            )
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

    return (compute_jaccard_best_matches,)


@app.cell
def _(all_assignments, compute_jaccard_best_matches, pl, selected_cities):
    pair_results = []
    for i in range(len(selected_cities)):
        for j in range(i + 1, len(selected_cities)):
            ab = compute_jaccard_best_matches(
                all_assignments, selected_cities[i], selected_cities[j]
            )
            ba = compute_jaccard_best_matches(
                all_assignments, selected_cities[j], selected_cities[i]
            )
            pair_results.extend([ab, ba])

    jaccard_df = pl.concat(pair_results)
    jaccard_df
    return (jaccard_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Arbovirus Calibration

    Arboviroses (dengue/chikungunya) is the anchor syndrome.
    Its cross-city Jaccard sets τ:

    > consistent in the same sense that dengue/chikungunya is consistent across cities.
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
def _(arbo_cluster_per_city, arbo_jaccard_df, pl):
    _min_ref_codes = 3

    _anchor_cities = arbo_cluster_per_city.filter(pl.col("ref_codes") >= _min_ref_codes)[
        "city"
    ].to_list()

    arbo_jaccard_df_filtered = arbo_jaccard_df.filter(
        pl.col("city_a").is_in(_anchor_cities) & pl.col("city_b").is_in(_anchor_cities)
    )
    return (arbo_jaccard_df_filtered,)


@app.cell
def _(arbo_jaccard_df_filtered, mo):
    tau = 0.2
    _callout_content = mo.md(f"No arbovirus matches found. Using default τ = {tau}")
    _callout_kind = "warn"

    if not arbo_jaccard_df_filtered.is_empty():
        tau = float(arbo_jaccard_df_filtered["jaccard"].median())
        _callout_content = mo.md(f"""
        **Arbovirus anchor Jaccard**

        Median across city pairs: **τ = {tau:.3f}**
        Range: {arbo_jaccard_df_filtered["jaccard"].min():.3f} – {arbo_jaccard_df_filtered["jaccard"].max():.3f}

        Clusters with best-match Jaccard ≥ τ are *consistent in the same sense
        that arboviroses is consistent across cities.*
        """)
        _callout_kind = "success"

    mo.callout(
        _callout_content,
        kind=_callout_kind,
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
    )
    _fig.add_vline(
        x=tau,
        line_dash="dash",
        line_color="red",
        annotation_text=f"τ = {tau:.3f}",  # arbovirus anchor
    )
    _fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    _fig.update_layout(
        height=350,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

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
def _(
    DATA_DIR,
    POPULATION_RANGE_TRANSLATIONS,
    REGION_TRANSLATIONS,
    TABLEAU_COLORS,
    jaccard_df,
    pl,
    px,
    runs_df,
    tau,
):
    _region_order = ["North", "Northeast", "Central-West", "South", "Southeast"]
    _population_order = ["250K-500K", "> 500K"]
    _region_colors = dict(zip(_region_order, TABLEAU_COLORS[:5]))

    _city_metadata = (
        runs_df.select(["city", "region", "population_range"])
        .unique()
        .with_columns(
            pl.col("region").replace(REGION_TRANSLATIONS),
            pl.col("population_range").replace(POPULATION_RANGE_TRANSLATIONS),
        )
    )

    _jaccard_with_metadata = jaccard_df.join(
        _city_metadata, left_on="city_a", right_on="city", how="left"
    )

    _fig_faceted = px.histogram(
        _jaccard_with_metadata,
        x="jaccard",
        color="region",
        color_discrete_map=_region_colors,
        facet_row="population_range",
        facet_col="region",
        facet_row_spacing=0.2,
        category_orders={
            "region": _region_order,
            "population_range": _population_order,
        },
        nbins=40,
        hover_name="city_a",
    )
    _fig_faceted.add_vline(
        x=tau,
        line_dash="dash",
        line_color="red",
        annotation_text=f"τ = {tau:.3f}",
    )
    _fig_faceted.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    _fig_faceted.update_layout(
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=500,
        margin=dict(b=110),
    )
    _fig_faceted.for_each_xaxis(lambda axis: axis.update(title_text=""))

    _cities_per_facet: dict[tuple[str, str], list[str]] = {}
    for _city_row in _city_metadata.iter_rows(named=True):
        _facet_key = (_city_row["region"], _city_row["population_range"])
        _cities_per_facet.setdefault(_facet_key, []).append(_city_row["city"])

    for _population_index, _population in enumerate(_population_order, start=1):
        for _region_index, _region in enumerate(_region_order, start=1):
            _cities_in_facet = _cities_per_facet.get((_region, _population))
            if not _cities_in_facet:
                continue
            _fig_faceted.add_annotation(
                row=_population_index,
                col=_region_index,
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=-0.1,
                text="<br>".join(sorted(_cities_in_facet)),
                showarrow=False,
                align="center",
                xanchor="center",
                yanchor="top",
                font=dict(size=12, color="black"),
            )

    _fig_faceted.write_image(DATA_DIR / "paper" / "jaccard_distribution.pdf")
    _fig_faceted
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Consistent Clusters

    Clusters where the best match in **every** paired city exceeds τ.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    A cluster from city *A* is **consistent** when it has a recognisable counterpart in other cities — formally, when its best-match Jaccard with at least **K** other cities is ≥ τ.

    The strictness is governed by **K**:

    - **K = 1** — the cluster need only resemble *one* counterpart elsewhere (loose — high coverage, low specificity).
    - **K = N** (all selected cities) — the cluster must resemble a counterpart in *every* other city (strict — low coverage, high specificity).

    The default uses the strictest criterion (K = N), which can yield **zero consistent clusters**: real-world coding noise means even well-behaved syndromes — including arboviroses — can fail to find a match in one or two cities. **Use the K slider further down** to relax the criterion and inspect how the count of consistent clusters trades off against breadth of cross-city evidence.
    """)
    return


@app.cell
def _(consistent_clusters, mo, n_consistent, n_total, tau):
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
    return


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


@app.cell
def _(selected_cities):
    len(selected_cities)
    return


@app.cell
def _(mo, selected_cities):
    k_slider = mo.ui.slider(
        start=1,
        stop=len(selected_cities),
        value=len(selected_cities),
        label="Minimum cities with Jaccard ≥ τ (K)",
        show_value=True,
    )

    mo.vstack(
        [
            mo.md(r"""
        ## Consistent Clusters

        A cluster is *consistent* if its best match in at least **K** other cities has Jaccard ≥ τ.

        K = max (total cities) means *appears everywhere*;
        K = 1 means *appears in at least one other city*

        """),
            k_slider,
        ]
    )
    return (k_slider,)


@app.cell
def _(arbo_cluster_per_city):
    arbo_cluster_per_city.select(["city", "arbo_cluster", "ref_codes"]).sort("ref_codes")
    return


@app.cell
def _(all_assignments, jaccard_df, k_slider, mo, pl, tau):
    consistency = (
        jaccard_df.with_columns((pl.col("jaccard") >= tau).alias("passes"))
        .group_by(["city_a", "cluster_a"])
        .agg(
            pl.col("passes").sum().alias("n_cities_passing"),
            pl.col("jaccard").median().alias("median_jaccard"),
            pl.col("jaccard").min().alias("min_jaccard"),
            pl.len().alias("n_pairs"),
        )
        .with_columns(
            # +1 to include the own city
            ((pl.col("n_cities_passing") + 1) >= k_slider.value).alias("consistent")
        )
        .sort("n_cities_passing", descending=True)
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
        .sort(["n_cities_passing", "median_jaccard"], descending=[True, True])
    )

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(
                        label="Consistent clusters",
                        value=f"{n_consistent} / {n_total}",
                        caption=f"Jaccard ≥ τ in ≥ {k_slider.value} cities",
                        bordered=True,
                    ),
                    mo.stat(
                        label="Unique syndromes (est.)",
                        value=f"~{n_consistent // len(set(consistency['city_a'].to_list()))}",
                        caption="consistent clusters ÷ n cities",
                    ),
                ]
            ),
            mo.ui.table(
                consistent_clusters.select(
                    [
                        "city_a",
                        "cluster_a",
                        "n_codes",
                        "n_cities_passing",
                        "median_jaccard",
                        "min_jaccard",
                    ]
                ),
                selection=None,
            ),
        ]
    )
    return consistency, consistent_clusters, n_consistent, n_total


@app.cell
def _(component_membership, mo, mutual_edges, pl, px, selected_cities, tau):
    _full_component_ids = (
        component_membership.group_by("component_id")
        .agg(pl.col("city").n_unique().alias("n_cities"))
        .filter(pl.col("n_cities") == len(selected_cities))
        .sort("component_id")["component_id"]
        .to_list()
    )

    mo.stop(
        len(_full_component_ids) == 0,
        mo.callout(mo.md("No components span all cities — nothing to plot."), kind="warn"),
    )

    _component_edges = (
        component_membership.join(
            component_membership.rename({"city": "city_b", "cluster_id": "cluster_b"}),
            on="component_id",
        )
        .filter(pl.col("city") != pl.col("city_b"))
        .join(
            mutual_edges.select(["city_a", "cluster_a", "city_b", "cluster_b", "min_jaccard"]),
            left_on=["city", "cluster_id", "city_b", "cluster_b"],
            right_on=["city_a", "cluster_a", "city_b", "cluster_b"],
            how="inner",
        )
    )

    _heatmap_long = (
        _component_edges.filter(pl.col("component_id").is_in(_full_component_ids))
        .group_by(["component_id", "city"])
        .agg(pl.col("min_jaccard").min().alias("median_jaccard"))
        .with_columns(pl.col("component_id").cast(pl.Utf8))
    )

    _component_order = (
        _heatmap_long.group_by("component_id")
        .agg(pl.col("median_jaccard").median().alias("overall_median"))
        .sort("overall_median", descending=True)["component_id"]
        .to_list()
    )

    _pivot = (
        _heatmap_long.pivot(on="component_id", index="city", values="median_jaccard")
        .fill_null(0)
        .sort("city")
    )

    _city_labels = _pivot["city"].to_list()
    _component_labels = [col for col in _component_order if col in _pivot.columns]
    _matrix = _pivot.select(_component_labels).to_numpy()

    _fig = px.imshow(
        _matrix,
        x=_component_labels,
        y=_city_labels,
        color_continuous_scale="Blues",
        zmin=tau,
        zmax=1.0,
        labels={"color": "Jaccard", "x": "Component ID", "y": "City"},
        # text_auto=".2f",
    )
    _fig.update_layout(
        xaxis_title="Component ID",
        yaxis_title="City",
        height=400,
        width=1400,
        margin=dict(l=120, r=120, t=60, b=80),
    )
    _fig.update_coloraxes(
        colorbar=dict(
            lenmode="fraction",
            len=1.0,
            thickness=20,
        ),
        cmin=0.2,
        cmax=1,
    )
    _fig.write_image("cross_city_heatmap.pdf", format="pdf", scale=2)
    _fig
    return


@app.cell
def _(jaccard_df, mo, pl, tau):
    # mutual best-match: edge only exists if A→B and B→A both have Jaccard ≥ τ
    _forward = jaccard_df.filter(pl.col("jaccard") >= tau).select(
        ["city_a", "cluster_a", "city_b", "cluster_b", "jaccard"]
    )
    _backward = _forward.rename(
        {
            "city_a": "city_b",
            "cluster_a": "cluster_b",
            "city_b": "city_a",
            "cluster_b": "cluster_a",
            "jaccard": "jaccard_rev",
        }
    )
    mutual_edges = _forward.join(
        _backward, on=["city_a", "cluster_a", "city_b", "cluster_b"], how="inner"
    ).with_columns(pl.min_horizontal("jaccard", "jaccard_rev").alias("min_jaccard"))

    mo.stat(
        label="Mutual edges",
        value=f"{len(mutual_edges):,}",
        caption=f"pairs where both directions ≥ τ ({tau:.3f})",
        bordered=True,
    )
    return (mutual_edges,)


@app.cell
def _(mutual_edges, pl):
    # Build node index: (city, cluster_id) → int
    all_nodes = (
        mutual_edges.select(
            pl.col("city_a").alias("city"),
            pl.col("cluster_a").cast(pl.Int64).alias("cluster_id"),
        )
        .vstack(
            mutual_edges.select(
                pl.col("city_b").alias("city"),
                pl.col("cluster_b").cast(pl.Int64).alias("cluster_id"),
            )
        )
        .unique()
        .sort(["city", "cluster_id"])
        .with_row_index("node_id")
    )

    node_to_id = {
        (row["city"], row["cluster_id"]): row["node_id"] for row in all_nodes.iter_rows(named=True)
    }

    edges_indexed = [
        (
            node_to_id[(row["city_a"], row["cluster_a"])],
            node_to_id[(row["city_b"], row["cluster_b"])],
        )
        for row in mutual_edges.iter_rows(named=True)
    ]
    return all_nodes, edges_indexed


@app.cell
def _(all_nodes, edges_indexed, ig):
    _graph = ig.Graph(n=len(all_nodes), edges=edges_indexed)
    components = _graph.connected_components()
    return (components,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Component Anatomy

    For each component, derive its *core* (codes present in every city's cluster), its *union* (codes present in any), and a per-city profile.

    The summary table below is the unit of expert validation; details arein the drill-down tabs at the end.
    """)
    return


@app.cell
def _(pl, run_datasets):
    code_volume_per_city = (
        pl.concat(
            [
                df.group_by(["code", "code_type"])
                .agg(pl.col("quantity").sum().alias("total_records"))
                .with_columns(pl.lit(city).alias("city"))
                for city, df in run_datasets.items()
            ]
        )
        if run_datasets
        else pl.DataFrame(
            schema={
                "code": pl.Utf8,
                "code_type": pl.Utf8,
                "total_records": pl.Int64,
                "city": pl.Utf8,
            }
        )
    )
    return (code_volume_per_city,)


@app.cell
def _(all_nodes, components, k_slider, pl):
    _membership_rows = []
    for _component_id, _member_node_ids in enumerate(components):
        _member_df = all_nodes.filter(pl.col("node_id").is_in(_member_node_ids))
        if _member_df["city"].n_unique() < k_slider.value:
            continue
        for _row in _member_df.iter_rows(named=True):
            _membership_rows.append(
                {
                    "component_id": _component_id,
                    "city": _row["city"],
                    "cluster_id": int(_row["cluster_id"]),
                }
            )
    component_membership = (
        pl.DataFrame(_membership_rows)
        if _membership_rows
        else pl.DataFrame(
            schema={
                "component_id": pl.Int64,
                "city": pl.Utf8,
                "cluster_id": pl.Int64,
            }
        )
    )
    return (component_membership,)


@app.cell
def _(all_assignments, code_volume_per_city, component_membership, pl):
    component_codes = (
        component_membership.join(
            all_assignments.with_columns(pl.col("cluster_id").cast(pl.Int64)),
            on=["city", "cluster_id"],
            how="inner",
        )
        .select(["component_id", "city", "code", "code_type", "description"])
        .join(
            code_volume_per_city,
            on=["city", "code", "code_type"],
            how="left",
        )
        .with_columns(pl.col("total_records").fill_null(0))
    )
    return (component_codes,)


@app.cell
def _(component_codes, pl):
    component_code_presence = (
        component_codes.group_by(["component_id", "code", "code_type", "description"])
        .agg(
            pl.col("city").n_unique().alias("n_cities_with_code"),
            pl.col("total_records").sum().alias("total_records_all_cities"),
        )
        .join(
            component_codes.group_by("component_id").agg(
                pl.col("city").n_unique().alias("n_cities_in_component")
            ),
            on="component_id",
        )
        .with_columns(
            (pl.col("n_cities_with_code") == pl.col("n_cities_in_component")).alias("is_core"),
            (pl.col("n_cities_with_code") == 1).alias("is_exclusive_to_one_city"),
        )
        .sort(
            ["component_id", "n_cities_with_code", "total_records_all_cities"],
            descending=[False, True, True],
        )
    )
    return (component_code_presence,)


@app.cell
def _(component_code_presence, component_codes, pl):
    component_per_city = (
        component_codes.join(
            component_code_presence.select(
                ["component_id", "code", "code_type", "n_cities_with_code"]
            ),
            on=["component_id", "code", "code_type"],
        )
        .group_by(["component_id", "city"])
        .agg(
            pl.col("code").n_unique().alias("n_codes"),
            pl.col("total_records").sum().alias("city_volume"),
            (pl.col("n_cities_with_code") == 1).sum().alias("n_exclusive_codes"),
        )
        .sort(["component_id", "city"])
    )
    return (component_per_city,)


@app.cell
def _(component_code_presence, component_per_city, pl):
    _cluster_size_stats = component_per_city.group_by("component_id").agg(
        pl.col("n_codes").min().alias("n_codes_min"),
        pl.col("n_codes").max().alias("n_codes_max"),
        pl.col("n_codes").median().alias("n_codes_median"),
    )
    _set_size_stats = component_code_presence.group_by("component_id").agg(
        pl.col("is_core").sum().alias("n_core_codes"),
        pl.col("is_exclusive_to_one_city").sum().alias("n_exclusive_codes"),
        pl.len().alias("n_union_codes"),
    )
    _volume_stats = component_code_presence.group_by("component_id").agg(
        pl.col("total_records_all_cities").min().alias("code_volume_min"),
        pl.col("total_records_all_cities").median().alias("code_volume_median"),
    )
    component_summary = _cluster_size_stats.join(_set_size_stats, on="component_id").join(
        _volume_stats, on="component_id"
    )
    return (component_summary,)


@app.cell
def _(
    all_nodes,
    component_code_presence,
    component_per_city,
    components,
    k_slider,
    pl,
):
    component_rows = []
    for _component_id, _member_node_ids in enumerate(components):
        _member_df = all_nodes.filter(pl.col("node_id").is_in(_member_node_ids))
        _cities_in_component = sorted(_member_df["city"].unique().to_list())
        if len(_cities_in_component) < k_slider.value:
            continue

        _rep_city = component_per_city.filter(pl.col("component_id") == _component_id).sort(
            "city_volume", descending=True
        )["city"][0]
        _rep_cluster = int(_member_df.filter(pl.col("city") == _rep_city)["cluster_id"][0])

        _core_descriptions = (
            component_code_presence.filter(
                (pl.col("component_id") == _component_id) & pl.col("is_core")
            )
            .sort("total_records_all_cities", descending=True)["description"]
            .to_list()
        )
        _union_descriptions = (
            component_code_presence.filter(pl.col("component_id") == _component_id)
            .sort(
                ["n_cities_with_code", "total_records_all_cities"],
                descending=[True, True],
            )["description"]
            .to_list()
        )

        component_rows.append(
            {
                "component_id": _component_id,
                "n_cities": len(_cities_in_component),
                "n_clusters": len(_member_node_ids),
                "cities": ", ".join(_cities_in_component),
                "rep_city": _rep_city,
                "rep_cluster": _rep_cluster,
                "core_codes": " | ".join(_core_descriptions)
                if _core_descriptions
                else "(empty core — no code present in every city)",
                "all_codes": " | ".join(_union_descriptions),
            }
        )
    return (component_rows,)


@app.cell
def _(
    all_nodes,
    component_rows,
    component_summary,
    k_slider,
    mo,
    mutual_edges,
    pl,
):
    syndromes_df = (
        pl.DataFrame(component_rows).join(component_summary, on="component_id")
        if component_rows
        else pl.DataFrame()
    )

    _n_syndromes = len(syndromes_df)
    mo.vstack(
        [
            mo.md(
                "## Syndrome Graph — Connected Components\n\n"
                "Each row is a candidate syndrome — clusters that mutually "
                "recognise each other across cities. Read `core_codes` to "
                "judge whether the cluster names a coherent syndrome; use "
                "`n_codes_median` and `code_volume_median` as red flags "
                "(very small / very rare clusters are unstable). The "
                "drill-down tabs below expose per-city and per-code detail."
            ),
            mo.hstack(
                [
                    mo.stat(
                        label="Candidate syndromes",
                        value=str(_n_syndromes),
                        caption=f"components spanning ≥ {k_slider.value} cities",
                        bordered=True,
                    ),
                    mo.stat(
                        label="Mutual edges",
                        value=str(len(mutual_edges)),
                        caption=f"on {len(all_nodes)} (city, cluster) nodes",
                    ),
                ]
            ),
            mo.ui.table(
                syndromes_df.select(
                    [
                        "component_id",
                        "n_codes_median",
                        "n_core_codes",
                        "code_volume_median",
                        "rep_city",
                        "core_codes",
                    ]
                )
                if not syndromes_df.is_empty()
                else syndromes_df,
                selection=None,
            ),
        ]
    )
    return


@app.cell
def _(component_code_presence, component_per_city, mo):
    mo.vstack(
        [
            mo.md(
                "### Drill-down\n\nFilter by `component_id` to inspect a "
                "single component in detail."
            ),
            mo.ui.tabs(
                {
                    "Per-City": mo.ui.table(component_per_city, selection=None, page_size=20),
                    "All codes": mo.ui.table(
                        component_code_presence, selection=None, page_size=20
                    ),
                }
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Export results
    """)
    return


@app.cell
def _(DATA_DIR, consistency, n_total, pl, px, selected_cities):
    _k_values = list(range(1, len(selected_cities) + 1))
    _sensitivity_rows = []
    for _k in _k_values:
        _n_consistent_k = consistency.filter(pl.col("n_cities_passing") >= _k - 1).height
        _sensitivity_rows.append(
            {
                "k": _k,
                "n_consistent": _n_consistent_k,
                "pct_consistent": _n_consistent_k / n_total * 100 if n_total > 0 else 0.0,
            }
        )
    _sensitivity_df = pl.DataFrame(_sensitivity_rows)

    (DATA_DIR / "paper").mkdir(parents=True, exist_ok=True)
    _sensitivity_df.write_csv(DATA_DIR / "paper" / "sensitivity_k.csv")

    _fig = px.line(
        _sensitivity_df,
        x="k",
        y="pct_consistent",
        markers=True,
        labels={
            "k": "Minimum cities required (K)",
            "pct_consistent": "Consistent clusters (%)",
        },
    )
    _fig.add_vline(
        x=len(selected_cities),
        line_dash="dot",
        line_color="gray",
        annotation_text=f"K = {len(selected_cities)}",
    )
    _fig.update_layout(
        plot_bgcolor="white",
        yaxis=dict(showgrid=True, gridcolor="rgb(235, 235, 235)"),
        xaxis=dict(showgrid=False),
    )
    _fig.update_xaxes(showline=True, linecolor="black", linewidth=1)
    _fig.update_yaxes(showline=True, linecolor="black", linewidth=1)
    _fig.write_image("sensitivity_k.pdf", scale=2)
    _fig
    return


@app.cell
def _(DATA_DIR, all_assignments, pl):
    _cluster_sizes = all_assignments.group_by(["city", "cluster_id"]).agg(
        pl.len().alias("cluster_size")
    )

    _partition_overview = (
        _cluster_sizes.group_by("city")
        .agg(
            (pl.col("cluster_size") == 1).sum().alias("singletons"),
            pl.col("cluster_size")
            .quantile(0.25, interpolation="nearest")
            .cast(pl.Int64)
            .alias("q25"),
            pl.col("cluster_size")
            .quantile(0.50, interpolation="nearest")
            .cast(pl.Int64)
            .alias("q50"),
            pl.col("cluster_size")
            .quantile(0.75, interpolation="nearest")
            .cast(pl.Int64)
            .alias("q75"),
            pl.col("cluster_size").max().alias("residual"),
        )
        .sort("city")
    )

    (DATA_DIR / "paper").mkdir(parents=True, exist_ok=True)
    _partition_overview.write_csv(DATA_DIR / "paper" / "partition_overview.csv")
    _partition_overview
    return


@app.cell
def _(
    DATA_DIR,
    component_membership,
    json,
    mo,
    n_consistent,
    n_total,
    pl,
    selected_cities,
    tau,
):
    _n_full_span_components = (
        component_membership.group_by("component_id")
        .agg(pl.col("city").n_unique().alias("n_cities"))
        .filter(pl.col("n_cities") == len(selected_cities))
        .height
    )

    _paper_numbers = {
        "n_total": n_total,
        "n_consistent": n_consistent,
        "pct_consistent": round(n_consistent / n_total * 100, 1) if n_total > 0 else 0.0,
        "n_cities": len(selected_cities),
        "k": len(selected_cities) - 1,
        "tau": round(tau, 3),
        "n_full_span_components": _n_full_span_components,
    }

    (DATA_DIR / "paper").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "paper" / "paper_numbers.json").write_text(json.dumps(_paper_numbers, indent=2))

    mo.callout(
        mo.md(f"""
        **Paper numbers saved** → `data/paper/paper_numbers.json`

        | Metric | Value |
        |--------|-------|
        | Total clusters | {n_total} |
        | Consistent (K = {len(selected_cities) - 1}, τ = {tau:.3f}) | {n_consistent} |
        | % consistent | {_paper_numbers["pct_consistent"]}% |
        | Components spanning all {len(selected_cities)} cities | {_n_full_span_components} |
        """),
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()
