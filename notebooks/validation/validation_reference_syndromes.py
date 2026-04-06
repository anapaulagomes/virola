import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import plotly.express as px
    import polars as pl
    from dotenv import load_dotenv
    from sklearn.metrics import normalized_mutual_info_score

    return load_dotenv, normalized_mutual_info_score, pl, px


@app.cell
def _():
    from vir.config import REFERENCES_DIR
    from vir.db import fetch_cluster_assignments, fetch_runs
    from vir.features import CODE_TYPES_MAPPING

    return REFERENCES_DIR, fetch_cluster_assignments, fetch_runs, CODE_TYPES_MAPPING


@app.cell
def _(load_dotenv):
    _ = load_dotenv()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Reference Syndrome Recovery

    Tests whether the SNF clustering naturally recovers two published syndrome
    definitions from AESOP/Fiocruz (peer-reviewed, operationally used by the
    Brazilian national health surveillance system), without those definitions
    being used as model inputs:

    - **Arboviroses**: dengue, chikungunya, zika and related
    - **URI / Respiratory**: influenza, acute upper respiratory infections and related

    **Logic**: if both reference syndromes emerge as coherent clusters, the remaining
    clusters inherit epidemiological credibility. This is the primary result of the paper.

    **Metrics**:
    - *Best-cluster recall*: fraction of reference codes in the single most concentrated cluster (Manning et al. 2008)
    - *Aggregated recall*: fraction of reference codes in any cluster above the precision threshold — accounts for valid sub-syndrome fragmentation (proposed; inspired by BCubed completeness, Amigó et al. 2009)
    - *Cluster precision*: fraction of a cluster that belongs to the reference definition (Manning et al. 2008)
    - *NMI*: normalised mutual information between reference membership and cluster labels (Strehl & Ghosh 2002)
    """)
    return


@app.cell
def _(REFERENCES_DIR, CODE_TYPES_MAPPING, pl):
    _AESOP_DIR = REFERENCES_DIR / "aesop"

    def _load_anchor(filename: str) -> pl.DataFrame:
        return (
            pl.read_csv(_AESOP_DIR / filename, separator=";", encoding="utf-8-sig")
            .rename({"Type": "raw_type", "Code": "code"})
            .with_columns(
                pl.col("raw_type").replace(CODE_TYPES_MAPPING).alias("code_type"),
                pl.col("code").str.strip_chars(),
            )
            .select("code", "code_type")
            .unique()
        )

    anchor_arbovirus = _load_anchor("code_list_arbovirus_apr2024.csv")
    anchor_uri = _load_anchor("code_list_uri_apr2024.csv")
    return anchor_arbovirus, anchor_uri


@app.cell
def _(fetch_runs, mo):
    _runs = fetch_runs()
    mo.stop(
        _runs.is_empty(),
        mo.callout(mo.md("No saved runs found. Run the SNF pipeline first."), kind="warn"),
    )
    _opts = {
        f"[{r['run_id']}] {r['dataset_name']} {r['year_min']}–{r['year_max']} ({r['n_clusters']} clusters)": r[
            "run_id"
        ]
        for r in _runs.iter_rows(named=True)
    }
    run_dropdown = mo.ui.dropdown(
        options=_opts, label="Select SNF run", value=list(_opts.keys())[0]
    )
    run_dropdown
    return (run_dropdown,)


@app.cell
def _(fetch_cluster_assignments, mo, run_dropdown):
    mo.stop(run_dropdown.value is None)
    assignments = fetch_cluster_assignments(run_dropdown.value)
    print(
        f"Run {run_dropdown.value}: {len(assignments)} codes "
        f"across {assignments['cluster_id'].n_unique()} clusters"
    )
    return (assignments,)


@app.cell
def _(anchor_arbovirus, anchor_uri, assignments, mo, pl):
    def _match_anchor(anchor: pl.DataFrame) -> pl.DataFrame:
        return anchor.join(
            assignments.select("terminology_id", "code", "code_type", "cluster_id"),
            on=["code", "code_type"],
            how="inner",
        )

    arbovirus_hits = _match_anchor(anchor_arbovirus)
    uri_hits = _match_anchor(anchor_uri)

    _coverage = pl.DataFrame(
        {
            "Reference syndrome": ["Arboviroses", "URI / Respiratory"],
            "Defined codes": [len(anchor_arbovirus), len(anchor_uri)],
            "Found in run": [len(arbovirus_hits), len(uri_hits)],
        }
    ).with_columns(
        (pl.col("Found in run") / pl.col("Defined codes") * 100).round(1).alias("Coverage (%)")
    )

    mo.vstack(
        [
            mo.md("## Reference Syndrome Coverage in Selected Run"),
            mo.ui.table(_coverage, selection=None),
        ]
    )
    return arbovirus_hits, uri_hits


@app.cell
def _(mo):
    precision_threshold = mo.ui.slider(
        start=5,
        stop=50,
        step=5,
        value=20,
        label="Precision threshold for coherent cluster (%)",
    )
    mo.vstack(
        [
            mo.md("""
        ### Precision threshold

        A cluster is considered a **coherent recovery** of the reference syndrome if at least
        this fraction of its codes belongs to the reference definition (cluster precision ≥ τ).
        Lowering the threshold counts fragmented or diluted clusters as valid recoveries.
        """),
            precision_threshold,
        ]
    )
    return (precision_threshold,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Arboviroses — Cluster Recovery

    How do arbovirus reference codes distribute across clusters?
    The best cluster is the primary candidate; fragmentation analysis shows
    whether remaining codes landed in coherent sub-syndrome clusters.
    """)
    return


@app.cell
def _(
    anchor_arbovirus,
    arbovirus_hits,
    assignments,
    mo,
    normalized_mutual_info_score,
    pl,
    px,
):
    mo.stop(
        len(arbovirus_hits) == 0,
        mo.callout(mo.md("⚠️ No arbovirus codes found in this run."), kind="warn"),
    )

    _cluster_counts = (
        arbovirus_hits.group_by("cluster_id")
        .agg(pl.len().alias("reference_codes"))
        .sort("reference_codes", descending=True)
    )
    arbo_dominant_cluster = int(_cluster_counts["cluster_id"][0])
    _n_in_dominant = int(_cluster_counts["reference_codes"][0])
    _n_anchor_found = len(arbovirus_hits)
    _n_cluster_size = int(assignments.filter(pl.col("cluster_id") == arbo_dominant_cluster).height)

    arbo_purity = _n_in_dominant / _n_anchor_found
    _arbo_precision = _n_in_dominant / _n_cluster_size
    _is_reference = (
        assignments["terminology_id"]
        .is_in(arbovirus_hits["terminology_id"])
        .cast(pl.Int8)
        .to_numpy()
    )
    arbo_nmi = float(
        normalized_mutual_info_score(_is_reference, assignments["cluster_id"].to_numpy())
    )

    _pass = arbo_purity >= 0.5
    _fig = px.bar(
        _cluster_counts.sort("cluster_id"),
        x="cluster_id",
        y="reference_codes",
        title="Arboviroses reference codes per cluster",
        labels={"cluster_id": "Cluster", "reference_codes": "Reference codes in cluster"},
    )
    _fig.add_vline(x=arbo_dominant_cluster, line_dash="dash", annotation_text="dominant")

    mo.vstack(
        [
            _fig,
            mo.callout(
                mo.md(f"""
            **Arboviroses best-cluster recall: {"✅ PASS" if _pass else "❌ FAIL"}**

            Reference codes found in run: `{_n_anchor_found}` / {len(anchor_arbovirus)} defined
            Dominant cluster: **{arbo_dominant_cluster}** ({_n_cluster_size} codes total)
            Best-cluster recall: `{arbo_purity:.2%}` | Cluster precision: `{_arbo_precision:.2%}` | NMI: `{arbo_nmi:.4f}`
            """),
                kind="success" if _pass else "danger",
            ),
        ]
    )
    return arbo_nmi, arbo_purity


@app.cell
def _(arbovirus_hits, assignments, mo, pl, precision_threshold, px):
    _cluster_sizes = assignments.group_by("cluster_id").agg(pl.len().alias("cluster_size"))

    _frag = (
        arbovirus_hits.group_by("cluster_id")
        .agg(pl.len().alias("reference_codes"))
        .join(_cluster_sizes, on="cluster_id")
        .with_columns(
            (pl.col("reference_codes") / pl.col("cluster_size") * 100)
            .round(1)
            .alias("precision_%"),
            (
                pl.col("reference_codes") / pl.col("cluster_size")
                >= precision_threshold.value / 100
            ).alias("coherent"),
        )
        .sort("reference_codes", descending=True)
    )

    arbo_coherent_clusters = _frag.filter(pl.col("coherent"))["cluster_id"].to_list()
    _n_recovered = int(_frag.filter(pl.col("coherent"))["reference_codes"].sum())
    _n_anchor_found = len(arbovirus_hits)
    arbo_recovered_purity = _n_recovered / _n_anchor_found if _n_anchor_found > 0 else 0.0

    # Stacked bar: reference vs other codes, only clusters that have reference codes
    _anchor_bar = (
        arbovirus_hits.group_by("cluster_id")
        .agg(pl.len().alias("count"))
        .with_columns(pl.lit("reference").alias("type"))
    )
    _other_bar = (
        assignments.join(arbovirus_hits.select("terminology_id"), on="terminology_id", how="anti")
        .filter(pl.col("cluster_id").is_in(_frag["cluster_id"].to_list()))
        .group_by("cluster_id")
        .agg(pl.len().alias("count"))
        .with_columns(pl.lit("other").alias("type"))
    )
    _bar_df = pl.concat([_anchor_bar, _other_bar]).sort("cluster_id")

    _fig = px.bar(
        _bar_df,
        x="cluster_id",
        y="count",
        color="type",
        color_discrete_map={"reference": "#2ca02c", "other": "#cccccc"},
        title="Arboviroses: reference vs other codes in clusters containing reference codes",
        labels={"cluster_id": "Cluster", "count": "Codes", "type": ""},
    )

    mo.vstack(
        [
            mo.md(
                f"### Fragmentation analysis (precision threshold: {precision_threshold.value}%)"
            ),
            mo.md(
                f"Aggregated recall (multi-cluster): `{arbo_recovered_purity:.2%}` — "
                f"{_n_recovered}/{_n_anchor_found} reference codes in coherent clusters"
            ),
            mo.ui.table(_frag, selection=None),
            _fig,
        ]
    )
    return arbo_coherent_clusters, arbo_recovered_purity


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. URI / Respiratory — Cluster Recovery

    Distribution of URI reference codes across clusters.
    """)
    return


@app.cell
def _(
    anchor_uri,
    assignments,
    mo,
    normalized_mutual_info_score,
    pl,
    px,
    uri_hits,
):
    mo.stop(
        len(uri_hits) == 0,
        mo.callout(mo.md("⚠️ No URI codes found in this run."), kind="warn"),
    )

    _cluster_counts = (
        uri_hits.group_by("cluster_id")
        .agg(pl.len().alias("reference_codes"))
        .sort("reference_codes", descending=True)
    )
    uri_dominant_cluster = int(_cluster_counts["cluster_id"][0])
    _n_in_dominant = int(_cluster_counts["reference_codes"][0])
    _n_anchor_found = len(uri_hits)
    _n_cluster_size = int(assignments.filter(pl.col("cluster_id") == uri_dominant_cluster).height)

    uri_purity = _n_in_dominant / _n_anchor_found
    _uri_precision = _n_in_dominant / _n_cluster_size
    _is_reference = (
        assignments["terminology_id"].is_in(uri_hits["terminology_id"]).cast(pl.Int8).to_numpy()
    )
    uri_nmi = float(
        normalized_mutual_info_score(_is_reference, assignments["cluster_id"].to_numpy())
    )

    _pass = uri_purity >= 0.5
    _fig = px.bar(
        _cluster_counts.sort("cluster_id"),
        x="cluster_id",
        y="reference_codes",
        title="URI / Respiratory reference codes per cluster",
        labels={"cluster_id": "Cluster", "reference_codes": "Reference codes in cluster"},
    )
    _fig.add_vline(x=uri_dominant_cluster, line_dash="dash", annotation_text="dominant")

    mo.vstack(
        [
            _fig,
            mo.callout(
                mo.md(f"""
            **URI best-cluster recall: {"✅ PASS" if _pass else "❌ FAIL"}**

            Reference codes found in run: `{_n_anchor_found}` / {len(anchor_uri)} defined
            Dominant cluster: **{uri_dominant_cluster}** ({_n_cluster_size} codes total)
            Best-cluster recall: `{uri_purity:.2%}` | Cluster precision: `{_uri_precision:.2%}` | NMI: `{uri_nmi:.4f}`
            """),
                kind="success" if _pass else "danger",
            ),
        ]
    )
    return uri_nmi, uri_purity


@app.cell
def _(assignments, mo, pl, precision_threshold, px, uri_hits):
    _cluster_sizes = assignments.group_by("cluster_id").agg(pl.len().alias("cluster_size"))

    _frag = (
        uri_hits.group_by("cluster_id")
        .agg(pl.len().alias("reference_codes"))
        .join(_cluster_sizes, on="cluster_id")
        .with_columns(
            (pl.col("reference_codes") / pl.col("cluster_size") * 100)
            .round(1)
            .alias("precision_%"),
            (
                pl.col("reference_codes") / pl.col("cluster_size")
                >= precision_threshold.value / 100
            ).alias("coherent"),
        )
        .sort("reference_codes", descending=True)
    )

    uri_coherent_clusters = _frag.filter(pl.col("coherent"))["cluster_id"].to_list()
    _n_recovered = int(_frag.filter(pl.col("coherent"))["reference_codes"].sum())
    _n_anchor_found = len(uri_hits)
    uri_recovered_purity = _n_recovered / _n_anchor_found if _n_anchor_found > 0 else 0.0

    _anchor_bar = (
        uri_hits.group_by("cluster_id")
        .agg(pl.len().alias("count"))
        .with_columns(pl.lit("reference").alias("type"))
    )
    _other_bar = (
        assignments.join(uri_hits.select("terminology_id"), on="terminology_id", how="anti")
        .filter(pl.col("cluster_id").is_in(_frag["cluster_id"].to_list()))
        .group_by("cluster_id")
        .agg(pl.len().alias("count"))
        .with_columns(pl.lit("other").alias("type"))
    )
    _bar_df = pl.concat([_anchor_bar, _other_bar]).sort("cluster_id")

    _fig = px.bar(
        _bar_df,
        x="cluster_id",
        y="count",
        color="type",
        color_discrete_map={"reference": "#1f77b4", "other": "#cccccc"},
        title="URI / Respiratory: reference vs other codes in clusters containing reference codes",
        labels={"cluster_id": "Cluster", "count": "Codes", "type": ""},
    )

    mo.vstack(
        [
            mo.md(
                f"### Fragmentation analysis (precision threshold: {precision_threshold.value}%)"
            ),
            mo.md(
                f"Aggregated recall (multi-cluster): `{uri_recovered_purity:.2%}` — "
                f"{_n_recovered}/{_n_anchor_found} reference codes in coherent clusters"
            ),
            mo.ui.table(_frag, selection=None),
            _fig,
        ]
    )
    return uri_coherent_clusters, uri_recovered_purity


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Geometric Coherence

    What non-reference codes share the coherent clusters with each syndrome?
    Epidemiologically coherent co-travelers confirm the clustering captures real signal.
    All clusters above the precision threshold are shown, not only the best one.
    """)
    return


@app.cell
def _(
    anchor_arbovirus,
    anchor_uri,
    arbo_coherent_clusters,
    assignments,
    mo,
    pl,
    uri_coherent_clusters,
):
    _arbo_reference_codes = set(anchor_arbovirus["code"].to_list())
    _uri_reference_codes = set(anchor_uri["code"].to_list())

    def _show_cluster(cluster_id: int, reference_codes: set, label: str):
        _df = (
            assignments.filter(pl.col("cluster_id") == cluster_id)
            .with_columns(pl.col("code").is_in(list(reference_codes)).alias("is_reference"))
            .select("code", "code_type", "description", "is_reference")
            .sort("is_reference", descending=True)
        )
        return mo.vstack(
            [
                mo.md(
                    f"**{label} — cluster {cluster_id}** "
                    f"({len(_df)} codes, {_df['is_reference'].sum()} reference)"
                ),
                mo.ui.table(_df, selection=None),
            ]
        )

    _arbo_panels = [
        _show_cluster(c, _arbo_reference_codes, "Arboviroses")
        for c in sorted(arbo_coherent_clusters)
    ]
    _uri_panels = [
        _show_cluster(c, _uri_reference_codes, "URI / Respiratory")
        for c in sorted(uri_coherent_clusters)
    ]

    mo.vstack(
        [
            mo.md("### Arboviroses coherent clusters"),
            *_arbo_panels,
            mo.md("### URI / Respiratory coherent clusters"),
            *_uri_panels,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary
    """)
    return


@app.cell
def _(
    arbo_nmi,
    arbo_purity,
    arbo_recovered_purity,
    mo,
    uri_nmi,
    uri_purity,
    uri_recovered_purity,
):
    _arbo_pass = arbo_recovered_purity >= 0.5
    _uri_pass = uri_recovered_purity >= 0.5
    _both_pass = _arbo_pass and _uri_pass

    _lines = (
        f"**Arboviroses**: best-cluster recall `{arbo_purity:.2%}` → aggregated recall `{arbo_recovered_purity:.2%}` | NMI `{arbo_nmi:.4f}` — "
        f"{'✅ PASS' if _arbo_pass else '❌ FAIL'}  \n"
        f"**URI / Respiratory**: best-cluster recall `{uri_purity:.2%}` → aggregated recall `{uri_recovered_purity:.2%}` | NMI `{uri_nmi:.4f}` — "
        f"{'✅ PASS' if _uri_pass else '❌ FAIL'}"
    )

    if _both_pass:
        _message = f"""
        **Primary result confirmed**

        Both published reference syndromes were recovered naturally by the SNF
        clustering without being used as model inputs.

        {_lines}
        """
    else:
        _message = f"""
        **Partial recovery — review required**

        {_lines}

        At least one reference syndrome was not recovered with aggregated recall ≥ 50%.
        Adjust the precision threshold or inspect the fragmentation bar charts
        to understand the failure mode (coverage gap vs. cluster dilution).
        """

    mo.callout(mo.md(_message), kind="success" if _both_pass else "warn")
    return


if __name__ == "__main__":
    app.run()
