import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import json
    import math
    from collections import defaultdict

    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    from dotenv import load_dotenv
    from plotly.subplots import make_subplots

    return defaultdict, go, json, load_dotenv, make_subplots, math, np, pl, px


@app.cell
def _():
    from vir.adapters.datasus import DatasusAdapter
    from vir.adapters.datasus.features import (
        CITY_NAMES,
        CODE_TYPES_MAPPING,
        IBGE_CITY_MAP,
    )
    from vir.config import PROJ_ROOT, REFERENCES_DIR
    from vir.db import fetch_cluster_assignments, fetch_runs

    return (
        CITY_NAMES,
        CODE_TYPES_MAPPING,
        DatasusAdapter,
        IBGE_CITY_MAP,
        PROJ_ROOT,
        REFERENCES_DIR,
        fetch_cluster_assignments,
        fetch_runs,
    )


@app.cell
def _(DatasusAdapter):
    outputs_dir = DatasusAdapter().outputs_dir
    return (outputs_dir,)


@app.cell
def _(load_dotenv):
    _ = load_dotenv()
    return


@app.cell
def _(DatasusAdapter, fetch_runs, mo):
    _models = sorted(value for value in fetch_runs()["model"].unique().to_list() if value)
    _adapter_default = DatasusAdapter.default_model
    _default_model = (
        _adapter_default if _adapter_default in _models else (_models[0] if _models else None)
    )
    model_selector = mo.ui.dropdown(
        options=_models,
        value=_default_model,
        label="Embedding model",
    )
    model_selector
    return (model_selector,)


@app.cell
def _(mo):
    mo.md(r"""
    # Reference Syndrome Recovery

    Tests whether the SNF clustering naturally recovers two published syndrome
    definitions from AESOP/Fiocruz (peer-reviewed, operationally used by the
    Brazilian national health surveillance system), without those definitions
    being used as model inputs:

    - **Arboviroses**: dengue, chikungunya, zika and related
    - **Influenza Like Illness (ILI)**: influenza, acute upper respiratory infections and related

    **Logic**: if both reference syndromes emerge as coherent clusters, the remaining
    clusters inherit epidemiological credibility. This is the primary result of the paper.

    **Metrics** (reference recovery is counted in matched reference *categories* — see the
    code-matching note below — while cluster precision is counted in corpus *codes*):
    - *Best-cluster recall*: fraction of matched reference categories concentrated in the single best cluster (Manning et al. 2008)
    - *Aggregated recall*: fraction of matched reference categories landing in any cluster above the precision threshold — accounts for valid sub-syndrome fragmentation (proposed; inspired by BCubed completeness, Amigó et al. 2009)
    - *Cluster precision*: fraction of a cluster's codes that belong to the reference definition (Manning et al. 2008)
    """)
    return


@app.cell
def _(CODE_TYPES_MAPPING, REFERENCES_DIR, pl):
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
            .unique(maintain_order=True)
        )

    anchor_arbovirus = _load_anchor("code_list_arbovirus_apr2024.csv")
    anchor_uri = _load_anchor("code_list_uri_apr2024.csv")
    return anchor_arbovirus, anchor_uri


@app.cell
def _(pl):
    def _build_match_key(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            pl.when(pl.col("code_type") == "cid10")
            .then(pl.col("code").str.slice(0, 3))
            .otherwise(pl.col("code"))
            .alias("match_key")
        )

    def match_reference_codes(anchor: pl.DataFrame, run_assignments: pl.DataFrame) -> pl.DataFrame:
        anchor_keys = _build_match_key(anchor).select("match_key", "code_type").unique()
        return _build_match_key(run_assignments).join(
            anchor_keys, on=["match_key", "code_type"], how="inner"
        )

    def count_reference_categories(anchor: pl.DataFrame) -> int:
        return _build_match_key(anchor).select("match_key", "code_type").unique().height

    def count_corpus_categories(run_assignments: pl.DataFrame) -> int:
        return _build_match_key(run_assignments).select("match_key", "code_type").unique().height

    return (
        count_corpus_categories,
        count_reference_categories,
        match_reference_codes,
    )


@app.cell
def _(mo):
    mo.md(r"""
    **Code matching follows the AESOP convention**: CID-10 reference codes are matched at the
    3-character category level (e.g. `A92` recovers `A92`, `A920`, `A928`, `A929`…), while CIAP
    and AB codes are matched exactly. *Coverage* below is therefore the fraction of reference
    **categories** that have at least one matched code in the run.
    """)
    return


@app.cell
def _(fetch_runs, mo, model_selector, pl):
    _runs = fetch_runs().filter(pl.col("model") == model_selector.value)
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
def _(
    anchor_arbovirus,
    anchor_uri,
    assignments,
    count_corpus_categories,
    count_reference_categories,
    match_reference_codes,
    mo,
    pl,
):
    _assignment_columns = assignments.select("terminology_id", "code", "code_type", "cluster_id")
    arbovirus_hits = match_reference_codes(anchor_arbovirus, _assignment_columns)
    uri_hits = match_reference_codes(anchor_uri, _assignment_columns)
    corpus_category_count = count_corpus_categories(_assignment_columns)

    def _categories_found(hits: pl.DataFrame) -> int:
        return hits.select("match_key", "code_type").unique().height

    _coverage = pl.DataFrame(
        {
            "Reference syndrome": ["Arboviroses", "Influenza Like Illness (ILI)"],
            "Defined categories": [
                count_reference_categories(anchor_arbovirus),
                count_reference_categories(anchor_uri),
            ],
            "Categories found": [
                _categories_found(arbovirus_hits),
                _categories_found(uri_hits),
            ],
            "Corpus codes matched": [len(arbovirus_hits), len(uri_hits)],
        }
    ).with_columns(
        (pl.col("Categories found") / pl.col("Defined categories") * 100)
        .round(1)
        .alias("Coverage (%)")
    )

    mo.vstack(
        [
            mo.md("## Reference Syndrome Coverage in Selected Run"),
            mo.ui.table(_coverage, selection=None),
        ]
    )
    return arbovirus_hits, corpus_category_count, uri_hits


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
def _(
    anchor_arbovirus,
    anchor_uri,
    corpus_category_count,
    count_reference_categories,
    mo,
    precision_threshold,
):
    _tau = precision_threshold.value / 100
    _arbo_reference_categories = count_reference_categories(anchor_arbovirus)
    _ili_reference_categories = count_reference_categories(anchor_uri)
    _arbo_base_rate = _arbo_reference_categories / corpus_category_count
    _ili_base_rate = _ili_reference_categories / corpus_category_count
    _arbo_enrichment = _tau / _arbo_base_rate
    _ili_enrichment = _tau / _ili_base_rate
    mo.callout(
        mo.md(
            rf"""**How to read τ = {precision_threshold.value}%**

The corpus contains **{corpus_category_count} distinct code categories**.
If codes were distributed uniformly across clusters, a randomly chosen cluster would contain
roughly {_arbo_base_rate:.1%} arbovirus categories ({_arbo_reference_categories} out of
{corpus_category_count}) and {_ili_base_rate:.1%} ILI categories ({_ili_reference_categories}
out of {corpus_category_count}).

Requiring τ = {precision_threshold.value}% therefore demands a cluster be
**{_arbo_enrichment:.1f}× above chance** to count as a coherent arbovirus recovery, and
**{_ili_enrichment:.1f}× above chance** for ILI.

The asymmetry is expected: arboviroses is a small syndrome ({_arbo_reference_categories}
categories), so even a moderate τ is a high relative bar. ILI is broader ({_ili_reference_categories}
categories), so the same τ is less demanding in relative terms — consistent with ILI codes being
more dispersed across the nosological space. A fixed τ is therefore *more stringent* for rare
syndromes than for common ones, which is a desirable property."""
        ),
        kind="info",
    )
    return


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
    count_reference_categories,
    mo,
    pl,
    px,
):
    mo.stop(
        len(arbovirus_hits) == 0,
        mo.callout(mo.md("⚠️ No arbovirus codes found in this run."), kind="warn"),
    )

    _cluster_counts = (
        arbovirus_hits.group_by("cluster_id")
        .agg(
            pl.struct("match_key", "code_type").n_unique().alias("reference_categories"),
            pl.len().alias("reference_code_hits"),
        )
        .sort(
            ["reference_categories", "reference_code_hits", "cluster_id"],
            descending=[True, True, False],
        )
    )
    _n_matched_categories = arbovirus_hits.select("match_key", "code_type").unique().height
    arbo_dominant_cluster = int(_cluster_counts["cluster_id"][0])
    _n_categories_in_dominant = int(_cluster_counts["reference_categories"][0])
    _n_code_hits_in_dominant = int(_cluster_counts["reference_code_hits"][0])
    _n_cluster_size = int(assignments.filter(pl.col("cluster_id") == arbo_dominant_cluster).height)

    arbo_purity = _n_categories_in_dominant / _n_matched_categories
    _arbo_precision = _n_code_hits_in_dominant / _n_cluster_size

    _pass = arbo_purity >= 0.5
    _fig = px.bar(
        _cluster_counts.sort("cluster_id"),
        x="cluster_id",
        y="reference_categories",
        title="Arboviroses reference categories per cluster",
        labels={
            "cluster_id": "Cluster",
            "reference_categories": "Reference categories in cluster",
        },
    )
    _fig.add_vline(x=arbo_dominant_cluster, line_dash="dash", annotation_text="dominant")

    mo.vstack(
        [
            _fig,
            mo.callout(
                mo.md(f"""
            **Arboviroses best-cluster recall: {"✅ PASS" if _pass else "❌ FAIL"}**

            Reference categories found in run: `{_n_matched_categories}` / {count_reference_categories(anchor_arbovirus)} defined
            Dominant cluster: **{arbo_dominant_cluster}** ({_n_cluster_size} codes total)
            Best-cluster recall: `{arbo_purity:.2%}` | Cluster precision: `{_arbo_precision:.2%}`
            """),
                kind="success" if _pass else "danger",
            ),
        ]
    )
    return (arbo_purity,)


@app.cell
def _(arbovirus_hits, assignments, mo, pl, precision_threshold, px):
    _cluster_sizes = assignments.group_by("cluster_id").agg(pl.len().alias("cluster_size"))

    _frag = (
        arbovirus_hits.group_by("cluster_id")
        .agg(
            pl.struct("match_key", "code_type").n_unique().alias("reference_categories"),
            pl.len().alias("reference_codes"),
        )
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
        .sort("reference_categories", descending=True)
    )

    arbo_coherent_clusters = _frag.filter(pl.col("coherent"))["cluster_id"].to_list()
    _n_matched_categories = arbovirus_hits.select("match_key", "code_type").unique().height
    _n_recovered = (
        arbovirus_hits.filter(pl.col("cluster_id").is_in(arbo_coherent_clusters))
        .select("match_key", "code_type")
        .unique()
        .height
    )
    arbo_recovered_purity = (
        _n_recovered / _n_matched_categories if _n_matched_categories > 0 else 0.0
    )

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
                f"{_n_recovered}/{_n_matched_categories} reference categories in coherent clusters"
            ),
            mo.ui.table(_frag, selection=None),
            _fig,
        ]
    )
    return arbo_coherent_clusters, arbo_recovered_purity


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Influenza Like Illness (ILI) — Cluster Recovery

    Distribution of ILI reference codes across clusters.
    """)
    return


@app.cell
def _(
    anchor_uri,
    assignments,
    count_reference_categories,
    mo,
    pl,
    px,
    uri_hits,
):
    mo.stop(
        len(uri_hits) == 0,
        mo.callout(mo.md("⚠️ No ILI codes found in this run."), kind="warn"),
    )

    _cluster_counts = (
        uri_hits.group_by("cluster_id")
        .agg(
            pl.struct("match_key", "code_type").n_unique().alias("reference_categories"),
            pl.len().alias("reference_code_hits"),
        )
        .sort(
            ["reference_categories", "reference_code_hits", "cluster_id"],
            descending=[True, True, False],
        )
    )
    _n_matched_categories = uri_hits.select("match_key", "code_type").unique().height
    uri_dominant_cluster = int(_cluster_counts["cluster_id"][0])
    _n_categories_in_dominant = int(_cluster_counts["reference_categories"][0])
    _n_code_hits_in_dominant = int(_cluster_counts["reference_code_hits"][0])
    _n_cluster_size = int(assignments.filter(pl.col("cluster_id") == uri_dominant_cluster).height)

    uri_purity = _n_categories_in_dominant / _n_matched_categories
    _uri_precision = _n_code_hits_in_dominant / _n_cluster_size

    _pass = uri_purity >= 0.5
    _fig = px.bar(
        _cluster_counts.sort("cluster_id"),
        x="cluster_id",
        y="reference_categories",
        title="Influenza Like Illness (ILI) reference categories per cluster",
        labels={
            "cluster_id": "Cluster",
            "reference_categories": "Reference categories in cluster",
        },
    )
    _fig.add_vline(x=uri_dominant_cluster, line_dash="dash", annotation_text="dominant")

    mo.vstack(
        [
            _fig,
            mo.callout(
                mo.md(f"""
            **ILI best-cluster recall: {"✅ PASS" if _pass else "❌ FAIL"}**

            Reference categories found in run: `{_n_matched_categories}` / {count_reference_categories(anchor_uri)} defined
            Dominant cluster: **{uri_dominant_cluster}** ({_n_cluster_size} codes total)
            Best-cluster recall: `{uri_purity:.2%}` | Cluster precision: `{_uri_precision:.2%}`
            """),
                kind="success" if _pass else "danger",
            ),
        ]
    )
    return (uri_purity,)


@app.cell
def _(assignments, mo, pl, precision_threshold, px, uri_hits):
    _cluster_sizes = assignments.group_by("cluster_id").agg(pl.len().alias("cluster_size"))

    _frag = (
        uri_hits.group_by("cluster_id")
        .agg(
            pl.struct("match_key", "code_type").n_unique().alias("reference_categories"),
            pl.len().alias("reference_codes"),
        )
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
        .sort("reference_categories", descending=True)
    )

    uri_coherent_clusters = _frag.filter(pl.col("coherent"))["cluster_id"].to_list()
    _n_matched_categories = uri_hits.select("match_key", "code_type").unique().height
    _n_recovered = (
        uri_hits.filter(pl.col("cluster_id").is_in(uri_coherent_clusters))
        .select("match_key", "code_type")
        .unique()
        .height
    )
    uri_recovered_purity = (
        _n_recovered / _n_matched_categories if _n_matched_categories > 0 else 0.0
    )

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
        title="Influenza Like Illness (ILI): reference vs other codes in clusters containing reference codes",
        labels={"cluster_id": "Cluster", "count": "Codes", "type": ""},
    )

    mo.vstack(
        [
            mo.md(
                f"### Fragmentation analysis (precision threshold: {precision_threshold.value}%)"
            ),
            mo.md(
                f"Aggregated recall (multi-cluster): `{uri_recovered_purity:.2%}` — "
                f"{_n_recovered}/{_n_matched_categories} reference categories in coherent clusters"
            ),
            mo.ui.table(_frag, selection=None),
            _fig,
        ]
    )
    return uri_coherent_clusters, uri_recovered_purity


@app.cell
def _(mo):
    mo.md(r"""
    ## Recovery flow — reference syndrome ↔ clusters

    A chord diagram. The coloured arc is the **reference syndrome**; each grey arc is a
    **cluster** that received reference codes. Each ribbon's width is the number of reference
    codes flowing from the syndrome into that cluster. On a cluster arc, the coloured slice is
    the reference portion and the grey remainder is its non-reference co-travelers — so a
    mostly-coloured arc recovered the syndrome cleanly, while a thin coloured slice on a large
    arc signals dilution.
    """)
    return


@app.cell
def _(arbovirus_hits, assignments, go, mo, np, pl, uri_hits):
    _GAP = 0.05

    def _arc_xy(angle_start, angle_end, radius=1.0, n_points=60):
        _angles = np.linspace(angle_start, angle_end, n_points)
        return radius * np.cos(_angles), radius * np.sin(_angles)

    def _bezier_xy(point_start, point_end, n_points=30):
        _t = np.linspace(0.0, 1.0, n_points)
        _x = (1 - _t) ** 2 * point_start[0] + _t**2 * point_end[0]
        _y = (1 - _t) ** 2 * point_start[1] + _t**2 * point_end[1]
        return _x, _y

    def _ribbon_xy(reference_arc, cluster_arc):
        _xa, _ya = _arc_xy(*reference_arc)
        _xb, _yb = _arc_xy(*cluster_arc)
        _bx1, _by1 = _bezier_xy((_xa[-1], _ya[-1]), (_xb[0], _yb[0]))
        _bx2, _by2 = _bezier_xy((_xb[-1], _yb[-1]), (_xa[0], _ya[0]))
        return (
            np.concatenate([_xa, _bx1, _xb, _bx2]),
            np.concatenate([_ya, _by1, _yb, _by2]),
        )

    def chord(hits: pl.DataFrame, label: str, arc_color: str, ribbon_color: str):
        if hits.is_empty():
            return mo.callout(mo.md(f"⚠️ No {label} codes found in this run."), kind="warn")

        _reference_counts = (
            hits.group_by("cluster_id")
            .agg(pl.len().alias("reference"))
            .sort("reference", descending=True)
        )
        _cluster_ids = _reference_counts["cluster_id"].to_list()
        _reference_map = dict(zip(_cluster_ids, _reference_counts["reference"].to_list()))
        _size_rows = (
            assignments.filter(pl.col("cluster_id").is_in(_cluster_ids))
            .group_by("cluster_id")
            .agg(pl.len().alias("size"))
        )
        _size_map = dict(zip(_size_rows["cluster_id"].to_list(), _size_rows["size"].to_list()))

        _reference_total = sum(_reference_map.values())
        _groups = [("Reference", _reference_total)] + [
            (cluster_id, _size_map[cluster_id]) for cluster_id in _cluster_ids
        ]
        _total_size = sum(_size for _, _size in _groups)
        _unit = (2 * np.pi - len(_groups) * _GAP) / _total_size

        _group_arc = {}
        _angle = np.pi / 2
        for _name, _size in _groups:
            _group_arc[_name] = (_angle, _angle + _size * _unit)
            _angle += _size * _unit + _GAP

        _reference_sub = {}
        _cursor = _group_arc["Reference"][0]
        for _cluster_id in _cluster_ids:
            _width = _reference_map[_cluster_id] * _unit
            _reference_sub[_cluster_id] = (_cursor, _cursor + _width)
            _cursor += _width

        _figure = go.Figure()

        for _cluster_id in _cluster_ids:
            _cluster_start = _group_arc[_cluster_id][0]
            _connected = (_cluster_start, _cluster_start + _reference_map[_cluster_id] * _unit)
            _x, _y = _ribbon_xy(_reference_sub[_cluster_id], _connected)
            _precision = _reference_map[_cluster_id] / _size_map[_cluster_id]
            _figure.add_trace(
                go.Scatter(
                    x=_x,
                    y=_y,
                    mode="lines",
                    fill="toself",
                    fillcolor=ribbon_color,
                    line=dict(color=ribbon_color, width=0.5),
                    hoveron="fills",
                    hoverinfo="text",
                    text=(
                        f"Cluster {_cluster_id}<br>"
                        f"Reference codes: {_reference_map[_cluster_id]} / {_size_map[_cluster_id]}<br>"
                        f"Cluster precision: {_precision:.0%}"
                    ),
                    showlegend=False,
                )
            )

        _reference_x, _reference_y = _arc_xy(*_group_arc["Reference"])
        _figure.add_trace(
            go.Scatter(
                x=_reference_x,
                y=_reference_y,
                mode="lines",
                line=dict(color=arc_color, width=16),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        _label_x, _label_y, _label_text, _label_size = [], [], [], []
        for _cluster_id in _cluster_ids:
            _cluster_start, _cluster_end = _group_arc[_cluster_id]
            _grey_x, _grey_y = _arc_xy(_cluster_start, _cluster_end)
            _figure.add_trace(
                go.Scatter(
                    x=_grey_x,
                    y=_grey_y,
                    mode="lines",
                    line=dict(color="#cccccc", width=16),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            _reference_end = _cluster_start + _reference_map[_cluster_id] * _unit
            _overlay_x, _overlay_y = _arc_xy(_cluster_start, _reference_end)
            _figure.add_trace(
                go.Scatter(
                    x=_overlay_x,
                    y=_overlay_y,
                    mode="lines",
                    line=dict(color=arc_color, width=16),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            _mid = (_cluster_start + _cluster_end) / 2
            _label_x.append(1.16 * np.cos(_mid))
            _label_y.append(1.16 * np.sin(_mid))
            _label_text.append(str(_cluster_id))
            _label_size.append(10)

        _reference_mid = (_group_arc["Reference"][0] + _group_arc["Reference"][1]) / 2
        _label_x.append(1.22 * np.cos(_reference_mid))
        _label_y.append(1.22 * np.sin(_reference_mid))
        _label_text.append(label)
        _label_size.append(13)

        _figure.add_trace(
            go.Scatter(
                x=_label_x,
                y=_label_y,
                mode="text",
                text=_label_text,
                textfont=dict(size=_label_size),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        _figure.update_layout(
            width=560,
            height=560,
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(visible=False, range=[-1.4, 1.4]),
            yaxis=dict(visible=False, range=[-1.4, 1.4], scaleanchor="x", scaleratio=1),
        )
        return _figure

    mo.vstack(
        [
            mo.vstack([chord(arbovirus_hits, "Arboviroses", "#2ca02c", "rgba(44,160,44,0.45)")]),
            mo.vstack(
                [
                    chord(
                        uri_hits,
                        "Influenza Like Illness (ILI)",
                        "#1f77b4",
                        "rgba(31,119,180,0.45)",
                    )
                ]
            ),
        ]
    )
    return (chord,)


@app.cell
def _(arbovirus_hits, chord, outputs_dir, uri_hits):
    chord(arbovirus_hits, "Arboviroses", "#2ca02c", "rgba(44,160,44,0.45)").write_image(
        outputs_dir / "chord_arbo.pdf"
    )
    chord(
        uri_hits, "Influenza Like Illness (ILI)", "#1f77b4", "rgba(31,119,180,0.45)"
    ).write_image(outputs_dir / "chord_uri.pdf")
    return


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
    arbo_coherent_clusters,
    arbovirus_hits,
    assignments,
    mo,
    pl,
    uri_coherent_clusters,
    uri_hits,
):
    _arbo_reference_codes = set(arbovirus_hits["code"].to_list())
    _uri_reference_codes = set(uri_hits["code"].to_list())

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
        _show_cluster(c, _uri_reference_codes, "Influenza Like Illness (ILI)")
        for c in sorted(uri_coherent_clusters)
    ]

    mo.vstack(
        [
            mo.md("### Arboviroses coherent clusters"),
            *_arbo_panels,
            mo.md("### Influenza Like Illness (ILI) coherent clusters"),
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
    arbo_purity,
    arbo_recovered_purity,
    mo,
    uri_purity,
    uri_recovered_purity,
):
    _arbo_pass = arbo_recovered_purity >= 0.5
    _uri_pass = uri_recovered_purity >= 0.5
    _both_pass = _arbo_pass and _uri_pass

    _lines = (
        f"**Arboviroses**: best-cluster recall `{arbo_purity:.2%}` → aggregated recall `{arbo_recovered_purity:.2%}` — "
        f"{'✅ PASS' if _arbo_pass else '❌ FAIL'}  \n"
        f"**Influenza Like Illness (ILI)**: best-cluster recall `{uri_purity:.2%}` → aggregated recall `{uri_recovered_purity:.2%}` — "
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


@app.cell
def _(mo):
    mo.md(r"""
    ## Cross-city summary
    """)
    return


@app.cell
def _(CITY_NAMES, fetch_runs, mo, model_selector, pl):
    _all_runs = fetch_runs().filter(
        pl.col("city").is_in(CITY_NAMES) & (pl.col("model") == model_selector.value)
    )
    _run_options = {
        f"{row['city']} (#{row['run_id']})": row["run_id"]
        for row in (
            _all_runs.sort("created_at", descending=True)
            .group_by("city")
            .first()
            .sort("city")
            .iter_rows(named=True)
        )
    }
    cross_city_run_select = mo.ui.multiselect(
        options=_run_options,
        value=list(_run_options.keys()),
        label="Runs para summary cross-city",
    )
    cross_city_run_select
    return (cross_city_run_select,)


@app.cell
def _(
    anchor_arbovirus,
    anchor_uri,
    count_reference_categories,
    cross_city_run_select,
    fetch_cluster_assignments,
    fetch_runs,
    match_reference_codes,
    mo,
    outputs_dir,
    pl,
):
    mo.stop(
        not cross_city_run_select.value,
        mo.callout(mo.md("Select at least one run."), kind="warn"),
    )

    def _compute_syndrome_metrics(
        anchor: pl.DataFrame,
        hits: pl.DataFrame,
        all_assignments: pl.DataFrame,
        precision_threshold: float = 0.20,
    ):
        if len(hits) == 0:
            return None
        _n_matched_categories = hits.select("match_key", "code_type").unique().height
        _coverage = _n_matched_categories / count_reference_categories(anchor)
        _cluster_sizes = all_assignments.group_by("cluster_id").agg(pl.len().alias("cluster_size"))
        _cluster_counts = (
            hits.group_by("cluster_id")
            .agg(
                pl.struct("match_key", "code_type").n_unique().alias("reference_categories"),
                pl.len().alias("reference_code_hits"),
            )
            .join(_cluster_sizes, on="cluster_id")
            .with_columns(
                (pl.col("reference_code_hits") / pl.col("cluster_size")).alias("precision"),
            )
            .sort(
                ["reference_categories", "precision", "cluster_id"],
                descending=[True, True, False],
            )
        )
        _dominant_cluster_id = int(_cluster_counts["cluster_id"][0])
        _n_categories_in_dominant = int(_cluster_counts["reference_categories"][0])
        _n_code_hits_in_dominant = int(_cluster_counts["reference_code_hits"][0])
        _n_cluster_size = int(_cluster_counts["cluster_size"][0])
        _frag = _cluster_counts.with_columns(
            (pl.col("precision") >= precision_threshold).alias("coherent"),
        )
        _coherent_clusters = _frag.filter(pl.col("coherent"))
        _n_coherent_clusters = _coherent_clusters.height
        _n_recovered = (
            hits.filter(pl.col("cluster_id").is_in(_coherent_clusters["cluster_id"].to_list()))
            .select("match_key", "code_type")
            .unique()
            .height
        )
        return {
            "coverage": _coverage,
            "best_cluster_recall": _n_categories_in_dominant / _n_matched_categories,
            "n_coherent_clusters": _n_coherent_clusters,
            "aggregated_recall": _n_recovered / _n_matched_categories,
            "cluster_precision": _n_code_hits_in_dominant / _n_cluster_size,
        }

    _run_id_to_city = {
        row["run_id"]: row["city"]
        for row in fetch_runs()
        .filter(pl.col("run_id").is_in(cross_city_run_select.value))
        .select(["run_id", "city"])
        .iter_rows(named=True)
    }

    _rows = []
    for _run_id in cross_city_run_select.value:
        _city_name = _run_id_to_city.get(_run_id, f"run_{_run_id}")
        _run_assignments = fetch_cluster_assignments(_run_id).select(
            ["terminology_id", "code", "code_type", "cluster_id"]
        )
        if _run_assignments.is_empty():
            continue
        for _syndrome_name, _anchor in [
            ("arboviroses", anchor_arbovirus),
            ("uri", anchor_uri),
        ]:
            _hits = match_reference_codes(_anchor, _run_assignments)
            _metrics = _compute_syndrome_metrics(_anchor, _hits, _run_assignments)
            if _metrics is None:
                continue
            _rows.append({"city": _city_name, "syndrome": _syndrome_name, **_metrics})

    validation_summary = pl.DataFrame(_rows)
    validation_summary.write_csv(outputs_dir / "validation_summary.csv")
    validation_summary
    return (validation_summary,)


@app.cell
def _(PROJ_ROOT, defaultdict, json):
    _STATE_TO_REGION = {
        "AC": "North",
        "AP": "North",
        "AM": "North",
        "PA": "North",
        "RO": "North",
        "RR": "North",
        "TO": "North",
        "AL": "Northeast",
        "BA": "Northeast",
        "CE": "Northeast",
        "MA": "Northeast",
        "PB": "Northeast",
        "PE": "Northeast",
        "PI": "Northeast",
        "RN": "Northeast",
        "SE": "Northeast",
        "ES": "Southeast",
        "MG": "Southeast",
        "RJ": "Southeast",
        "SP": "Southeast",
        "PR": "South",
        "RS": "South",
        "SC": "South",
        "DF": "Central-West",
        "GO": "Central-West",
        "MS": "Central-West",
        "MT": "Central-West",
    }

    with open(PROJ_ROOT / "publication" / "brazil-states.geojson") as _file:
        _geojson = json.load(_file)

    brazil_state_lons, brazil_state_lats = [], []
    _edge_contributors = defaultdict(list)

    for _feature in _geojson["features"]:
        _state_code = _feature["properties"]["sigla"]
        _region = _STATE_TO_REGION[_state_code]
        _geometry = _feature["geometry"]
        _rings = (
            _geometry["coordinates"]
            if _geometry["type"] == "Polygon"
            else [ring for poly in _geometry["coordinates"] for ring in poly]
        )
        for _ring in _rings:
            for _coordinate in _ring:
                brazil_state_lons.append(_coordinate[0])
                brazil_state_lats.append(_coordinate[1])
            brazil_state_lons.append(None)
            brazil_state_lats.append(None)

            for _index in range(len(_ring) - 1):
                _edge_start, _edge_end = tuple(_ring[_index]), tuple(_ring[_index + 1])
                _edge = (min(_edge_start, _edge_end), max(_edge_start, _edge_end))
                _edge_contributors[_edge].append((_state_code, _region))

    brazil_region_lons, brazil_region_lats = [], []
    for _edge, _contributors in _edge_contributors.items():
        if len(_contributors) == 1 or (
            len(_contributors) == 2 and _contributors[0][1] != _contributors[1][1]
        ):
            brazil_region_lons.extend([_edge[0][0], _edge[1][0], None])
            brazil_region_lats.extend([_edge[0][1], _edge[1][1], None])
    return (
        brazil_region_lats,
        brazil_region_lons,
        brazil_state_lats,
        brazil_state_lons,
    )


@app.cell
def _(
    IBGE_CITY_MAP,
    brazil_region_lats,
    brazil_region_lons,
    brazil_state_lats,
    brazil_state_lons,
    go,
    math,
    mo,
    pl,
    px,
    validation_summary,
):
    mo.stop(
        validation_summary.is_empty(),
        mo.callout(mo.md("No validation data to map."), kind="warn"),
    )

    _city_coordinates = pl.DataFrame(
        [
            {"city": _entry["city"], "lat": _entry["lat"], "lon": _entry["lon"]}
            for _entry in IBGE_CITY_MAP.values()
        ]
    )

    _display_offsets = {
        "Porto Alegre": (-1.0, -1.0),
        "Gravataí": (1.0, 1.0),
    }

    def _wedge_path(center_lat, center_lon, fraction, radius_deg=1.3, n_points=48):
        _fraction = max(min(fraction, 1.0), 0.0)
        _cos_lat = math.cos(math.radians(center_lat))
        _angles = [90.0 - 360.0 * _fraction * step / n_points for step in range(n_points + 1)]
        _lats = [center_lat + radius_deg * math.sin(math.radians(a)) for a in _angles]
        _lons = [center_lon + radius_deg * math.cos(math.radians(a)) / _cos_lat for a in _angles]
        if _fraction < 1.0:
            _lats = [center_lat, *_lats, center_lat]
            _lons = [center_lon, *_lons, center_lon]
        return _lats, _lons

    def pacman_map(syndrome_key: str, title: str):
        _data = (
            validation_summary.filter(pl.col("syndrome") == syndrome_key)
            .join(_city_coordinates, on="city", how="inner")
            .sort("city")
        )
        if _data.is_empty():
            return mo.callout(mo.md(f"No data for {title}."), kind="warn")

        _rows = list(_data.iter_rows(named=True))
        _display_coordinates = [
            (
                _row["lat"] + _display_offsets.get(_row["city"], (0.0, 0.0))[0],
                _row["lon"] + _display_offsets.get(_row["city"], (0.0, 0.0))[1],
            )
            for _row in _rows
        ]
        _coverages = [max(min(_row["coverage"], 1.0), 0.0) for _row in _rows]
        _fill_colors = px.colors.sample_colorscale("RdYlGn", _coverages)

        _figure = go.Figure()

        _figure.add_trace(
            go.Scattergeo(
                lon=brazil_state_lons,
                lat=brazil_state_lats,
                mode="lines",
                line=dict(color="#CCCCCC", width=0.4),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        _figure.add_trace(
            go.Scattergeo(
                lon=brazil_region_lons,
                lat=brazil_region_lats,
                mode="lines",
                line=dict(color="#3A3A3A", width=1.4),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        for _row, (_display_lat, _display_lon) in zip(_rows, _display_coordinates):
            if (_display_lat, _display_lon) == (_row["lat"], _row["lon"]):
                continue
            _figure.add_trace(
                go.Scattergeo(
                    lat=[_row["lat"], _display_lat],
                    lon=[_row["lon"], _display_lon],
                    mode="lines",
                    line=dict(color="#999999", width=0.8, dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        for (_display_lat, _display_lon), _row, _fill_color in zip(
            _display_coordinates, _rows, _fill_colors
        ):
            _lats, _lons = _wedge_path(_display_lat, _display_lon, _row["best_cluster_recall"])
            _figure.add_trace(
                go.Scattergeo(
                    lat=_lats,
                    lon=_lons,
                    mode="lines",
                    fill="toself",
                    fillcolor=_fill_color,
                    line=dict(color="#444444", width=0.6),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        _figure.add_trace(
            go.Scattergeo(
                lat=[_coordinate[0] for _coordinate in _display_coordinates],
                lon=[_coordinate[1] for _coordinate in _display_coordinates],
                mode="markers",
                marker=dict(size=3, color="#222222"),
                customdata=_data.select(["city", "best_cluster_recall", "coverage"]).to_numpy(),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Best-cluster recall: %{customdata[1]:.0%}<br>"
                    "Coverage: %{customdata[2]:.0%}<extra></extra>"
                ),
                showlegend=False,
            )
        )

        _figure.add_trace(
            go.Scattergeo(
                lat=[_coordinate[0] for _coordinate in _display_coordinates],
                lon=[_coordinate[1] for _coordinate in _display_coordinates],
                mode="markers",
                marker=dict(
                    size=0.1,
                    color=_coverages,
                    colorscale="RdYlGn",
                    cmin=0,
                    cmax=1,
                    showscale=True,
                    colorbar=dict(
                        title="Coverage",
                        tickformat=".0%",
                        orientation="h",
                        x=0.5,
                        xanchor="center",
                        y=-0.02,
                        yanchor="top",
                        thickness=12,
                        len=0.6,
                    ),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        _figure.update_geos(
            showland=False,
            showocean=False,
            showcountries=False,
            showcoastlines=False,
            showframe=False,
            showlakes=False,
            showrivers=False,
            showsubunits=False,
            bgcolor="white",
            lataxis_range=[-34, 6],
            lonaxis_range=[-75, -33],
            projection_type="mercator",
        )
        _figure.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=0, r=0, t=40, b=40),
            width=620,
            height=660,
        )
        return _figure

    mo.vstack(
        [
            mo.md(
                "### Cross-city recovery — pac-man fill = best-cluster recall, colour = coverage"
            ),
            mo.vstack([pacman_map("arboviroses", "Arboviroses")]),
            mo.vstack([pacman_map("uri", "Influenza Like Illness (ILI)")]),
        ]
    )
    return (pacman_map,)


@app.cell
def _(outputs_dir, pacman_map):
    pacman_map("arboviroses", "Arboviroses").write_image(outputs_dir / "pacmanmap_arboviroses.pdf")
    pacman_map("uri", "Influenza Like Illness").write_image(outputs_dir / "pacmanmap_ili.pdf")
    return


@app.cell
def _(mo, outputs_dir, pl, validation_summary):
    mo.stop(
        validation_summary.is_empty(),
        mo.callout(mo.md("No validation data to pivot."), kind="warn"),
    )
    _metric_order = [
        "coverage",
        "best_cluster_recall",
        "n_coherent_clusters",
        "aggregated_recall",
        "cluster_precision",
    ]
    _syndrome_order = ["arboviroses", "uri"]
    _column_order = ["city"] + [
        f"{_syndrome}_{_metric}" for _syndrome in _syndrome_order for _metric in _metric_order
    ]
    _coherent_count_columns = [f"{_syndrome}_n_coherent_clusters" for _syndrome in _syndrome_order]
    _pivot = (
        validation_summary.unpivot(
            index=["city", "syndrome"],
            on=_metric_order,
            variable_name="metric",
            value_name="value",
        )
        .with_columns(pl.concat_str(["syndrome", "metric"], separator="_").alias("column_name"))
        .pivot(index="city", on="column_name", values="value")
        .select(_column_order)
        .with_columns(pl.all().exclude("city").round(2))
        .with_columns([pl.col(_column).cast(pl.Int64) for _column in _coherent_count_columns])
        .sort("city")
    )
    _pivot.write_csv(outputs_dir / "validation_summary_pivot.csv")
    mo.ui.table(_pivot, selection=None)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Cluster vs reference codes — presence matrix

    For each reference syndrome, a wide table where:

    - Rows are the reference codes in their canonical order, followed by the
      other codes that appear in any of the displayed clusters (sorted).
    - Column 1 is `code`. Column 2 (`REFERENCE`) marks the reference
      definition itself (`X` for anchor codes).
    - Following columns are the coherent clusters (cluster precision ≥
      threshold) across the selected city runs, dominant cluster first within
      each city. Column names are `<city> / cluster <id>`.
    - Cell value `X` means the code is present in that cluster.

    One CSV per syndrome is written next to the notebook:
    `arbovirus_clusters_vs_reference.csv` and `uri_clusters_vs_reference.csv`.
    """)
    return


@app.cell
def _(
    anchor_arbovirus,
    anchor_uri,
    cross_city_run_select,
    fetch_cluster_assignments,
    fetch_runs,
    match_reference_codes,
    mo,
    outputs_dir,
    pl,
    precision_threshold,
):
    mo.stop(
        not cross_city_run_select.value,
        mo.callout(mo.md("Select at least one run."), kind="warn"),
    )

    def _build_presence_matrix(
        anchor: pl.DataFrame,
        run_id_to_city: dict[int, str],
        threshold_fraction: float,
    ) -> pl.DataFrame:
        anchor_codes_ordered: list[str] = []
        seen_anchor_codes: set[str] = set()
        for code in anchor["code"].to_list():
            if code not in seen_anchor_codes:
                seen_anchor_codes.add(code)
                anchor_codes_ordered.append(code)

        cluster_columns: list[tuple[str, set[str]]] = []
        present_other_codes: set[str] = set()
        matched_reference_codes: set[str] = set()

        for run_id, city_name in run_id_to_city.items():
            run_assignments = fetch_cluster_assignments(run_id).select(
                ["code", "code_type", "cluster_id"]
            )
            if run_assignments.is_empty():
                continue
            cluster_sizes = run_assignments.group_by("cluster_id").agg(
                pl.len().alias("cluster_size")
            )
            hits = match_reference_codes(anchor, run_assignments)
            if hits.is_empty():
                continue
            run_reference_codes = set(hits["code"].to_list())
            matched_reference_codes.update(run_reference_codes)
            coherent = (
                hits.group_by("cluster_id")
                .agg(pl.len().alias("reference_codes"))
                .join(cluster_sizes, on="cluster_id")
                .filter(pl.col("reference_codes") / pl.col("cluster_size") >= threshold_fraction)
                .sort("reference_codes", descending=True)
            )
            for cluster_id in coherent["cluster_id"].to_list():
                cluster_codes_present = set(
                    run_assignments.filter(pl.col("cluster_id") == cluster_id)["code"].to_list()
                )
                present_other_codes.update(
                    cluster_codes_present - run_reference_codes - seen_anchor_codes
                )
                column_name = f"{city_name} / cluster {cluster_id}"
                cluster_columns.append((column_name, cluster_codes_present))

        reference_code_set = seen_anchor_codes | matched_reference_codes
        expanded_reference_codes = sorted(matched_reference_codes - seen_anchor_codes)
        other_codes_ordered = sorted(present_other_codes - matched_reference_codes)
        all_code_rows = anchor_codes_ordered + expanded_reference_codes + other_codes_ordered

        rows: list[dict] = []
        for code in all_code_rows:
            row = {
                "code": code,
                "REFERENCE": "X" if code in reference_code_set else "",
            }
            for column_name, codes_present in cluster_columns:
                row[column_name] = "X" if code in codes_present else ""
            rows.append(row)

        ordered_columns = ["code", "REFERENCE", *(name for name, _ in cluster_columns)]
        return pl.DataFrame(rows).select(ordered_columns)

    _run_id_to_city = {
        row["run_id"]: row["city"]
        for row in fetch_runs()
        .filter(pl.col("run_id").is_in(cross_city_run_select.value))
        .sort("city")
        .select(["run_id", "city"])
        .iter_rows(named=True)
    }

    _threshold_fraction = precision_threshold.value / 100

    _arbovirus_presence_matrix = _build_presence_matrix(
        anchor_arbovirus, _run_id_to_city, _threshold_fraction
    )
    _uri_presence_matrix = _build_presence_matrix(anchor_uri, _run_id_to_city, _threshold_fraction)

    _arbovirus_presence_matrix.write_csv(outputs_dir / "arbovirus_clusters_vs_reference.csv")
    _uri_presence_matrix.write_csv(outputs_dir / "uri_clusters_vs_reference.csv")

    mo.vstack(
        [
            mo.md("### Arboviroses — cluster presence vs reference codes"),
            mo.ui.table(_arbovirus_presence_matrix, selection=None),
            mo.md("### Influenza Like Illness (ILI) — cluster presence vs reference codes"),
            mo.ui.table(_uri_presence_matrix, selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Best-cluster membership across cities (UpSet)

    Each row is a **code**; the first column marks whether it belongs to the **reference
    syndrome** definition, and the remaining columns are the **cities** — a dot means the code
    appears in that city's *best* cluster (the cluster holding the most reference codes for the
    syndrome in that city's run). Rows are ordered alphabetically by code, and the top bar gives
    each column's total. Reference dots use the syndrome colour, city dots are dark; the
    connecting line links the columns a code is shared across.
    """)
    return


@app.cell
def _(
    anchor_arbovirus,
    anchor_uri,
    cross_city_run_select,
    fetch_cluster_assignments,
    fetch_runs,
    go,
    make_subplots,
    match_reference_codes,
    mo,
    pl,
):
    mo.stop(
        not cross_city_run_select.value,
        mo.callout(mo.md("Select at least one run."), kind="warn"),
    )

    _run_id_to_city = {
        _row["run_id"]: _row["city"]
        for _row in fetch_runs()
        .filter(pl.col("run_id").is_in(cross_city_run_select.value))
        .sort("city")
        .select(["run_id", "city"])
        .iter_rows(named=True)
    }

    def _best_cluster_codes(anchor: pl.DataFrame):
        _city_codes: dict[str, set[str]] = {}
        _code_type_by_code: dict[str, str] = {}
        for _run_id, _city in _run_id_to_city.items():
            _city_assignments = fetch_cluster_assignments(_run_id).select(
                ["code", "code_type", "cluster_id"]
            )
            if _city_assignments.is_empty():
                continue
            _hits = match_reference_codes(anchor, _city_assignments)
            if _hits.is_empty():
                continue
            _best_cluster = (
                _hits.group_by("cluster_id")
                .agg(pl.len().alias("reference"))
                .sort(["reference", "cluster_id"], descending=[True, False])["cluster_id"][0]
            )
            _members = _city_assignments.filter(pl.col("cluster_id") == _best_cluster)
            _city_codes[_city] = set(_members["code"].to_list())
            for _code, _code_type in zip(
                _members["code"].to_list(), _members["code_type"].to_list()
            ):
                _code_type_by_code[_code] = _code_type
        return _city_codes, _code_type_by_code

    def upset(anchor: pl.DataFrame, label: str, dot_color: str):
        _city_codes, _code_type_by_code = _best_cluster_codes(anchor)
        if not _city_codes:
            return mo.callout(mo.md(f"⚠️ No best clusters for {label}."), kind="warn")

        _cities = sorted(_city_codes.keys())
        _all_codes = sorted(set().union(*_city_codes.values()))
        _reference_frame = pl.DataFrame(
            {
                "code": _all_codes,
                "code_type": [_code_type_by_code[_code] for _code in _all_codes],
            }
        )
        _reference_codes = set(match_reference_codes(anchor, _reference_frame)["code"].to_list())

        _all_codes.sort()

        _columns = ["Reference", *_cities]

        def _present(code: str, column: str) -> bool:
            if column == "Reference":
                return code in _reference_codes
            return code in _city_codes[column]

        _grid_x, _grid_y = [], []
        _dot_x, _dot_y, _dot_colors, _dot_text = [], [], [], []
        _line_x, _line_y = [], []
        for _row_index, _code in enumerate(_all_codes):
            _present_columns = []
            for _column_index, _column in enumerate(_columns):
                _grid_x.append(_column_index)
                _grid_y.append(_row_index)
                if _present(_code, _column):
                    _present_columns.append(_column_index)
                    _dot_x.append(_column_index)
                    _dot_y.append(_row_index)
                    _dot_colors.append(dot_color if _column == "Reference" else "#333333")
                    _dot_text.append(f"{_code} — {_column}")
            if len(_present_columns) >= 2:
                _line_x += [min(_present_columns), max(_present_columns), None]
                _line_y += [_row_index, _row_index, None]

        _set_sizes = [
            sum(_present(_code, _column) for _code in _all_codes) for _column in _columns
        ]

        _figure = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.20, 0.84],
            vertical_spacing=0.02,
        )
        _figure.add_trace(
            go.Bar(
                x=list(range(len(_columns))),
                y=_set_sizes,
                marker_color="#888888",
                text=_set_sizes,
                textposition="outside",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        _figure.add_trace(
            go.Scatter(
                x=_grid_x,
                y=_grid_y,
                mode="markers",
                marker=dict(size=7, color="#eeeeee"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        _figure.add_trace(
            go.Scatter(
                x=_line_x,
                y=_line_y,
                mode="lines",
                line=dict(color="#bbbbbb", width=1),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        _figure.add_trace(
            go.Scatter(
                x=_dot_x,
                y=_dot_y,
                mode="markers",
                marker=dict(size=8, color=_dot_colors),
                text=_dot_text,
                hoverinfo="text",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        _figure.update_layout(
            width=760,
            height=max(360, 60 + len(_all_codes) * 15),
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=110, r=20, t=50, b=110),
        )
        _figure.update_yaxes(
            title_text="nr. of codes",
            range=[0, max(_set_sizes) * 1.2 + 0.8],
            row=1,
            col=1,
        )
        _figure.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(_columns))),
            ticktext=_columns,
            tickangle=-40,
            row=2,
            col=1,
        )
        _figure.update_yaxes(
            tickmode="array",
            tickvals=list(range(len(_all_codes))),
            ticktext=_all_codes,
            autorange="reversed",
            row=2,
            col=1,
        )
        return _figure

    mo.vstack(
        [
            mo.vstack([upset(anchor_arbovirus, "Arboviroses", "#2ca02c")]),
            mo.vstack([upset(anchor_uri, "Influenza Like Illness (ILI)", "#1f77b4")]),
        ]
    )
    return (upset,)


@app.cell
def _(anchor_arbovirus, anchor_uri, outputs_dir, upset):
    upset(anchor_arbovirus, "Arboviroses", "#2ca02c").write_image(
        outputs_dir / "upset_arboviroses.pdf"
    )
    upset(anchor_uri, "Influenza Like Illness (ILI)", "#1f77b4").write_image(
        outputs_dir / "upset_ili.pdf"
    )
    return


if __name__ == "__main__":
    app.run()
