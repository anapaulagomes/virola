import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import unicodedata

    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    from dotenv import load_dotenv

    return go, load_dotenv, np, pl, px, unicodedata


@app.cell
def _():
    from vir.config import PROCESSED_DATA_DIR
    from vir.db import get_connection
    from vir.helpers import extract_dataset_name
    from vir.notebook_widgets import (
        create_file_dropdown,
        load_and_prepare_dataframe,
    )
    from vir.views.temporal import TemporalView

    return (
        PROCESSED_DATA_DIR,
        TemporalView,
        create_file_dropdown,
        extract_dataset_name,
        get_connection,
        load_and_prepare_dataframe,
    )


@app.cell
def _(load_dotenv):
    _ = load_dotenv()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Temporal View — Sanity Check

    The temporal view preserves known epidemiological structure: arboviroses
    separate from respiratory diseases, chronic conditions separate from both.
    Full validation is done in the reference syndrome recovery post-SNF (Section 5.2).

    This notebook checks that the MSM distance matrix is **not broken** —
    it does not attempt to prove the view is optimal.
    """)
    return


@app.cell
def _(mo):
    args = mo.cli_args() if mo.app_meta().mode == "script" else {}
    return (args,)


@app.cell
def _(args):
    year_min = args.get("year-min", 2022)
    year_max = args.get("year-max", 2024)
    min_weeks = args.get("min-weeks", 12)
    return min_weeks, year_max, year_min


@app.cell
def _(args, create_file_dropdown):
    default_dataset = args.get("dataset", "salvador")
    files_dropdown = create_file_dropdown(default_pattern=default_dataset)
    return (files_dropdown,)


@app.cell
def _(files_dropdown):
    files_dropdown
    return


@app.cell
def _(files_dropdown):
    file_path = files_dropdown.value
    return (file_path,)


@app.cell
def _(
    extract_dataset_name,
    file_path,
    load_and_prepare_dataframe,
    year_max,
    year_min,
):
    df = load_and_prepare_dataframe(
        file_path=file_path,
        year_min=year_min,
        year_max=year_max,
    )
    dataset_name = extract_dataset_name(file_path)
    return dataset_name, df


@app.cell
def _(TemporalView, dataset_name, df, min_weeks):
    temporal_view = TemporalView(
        df=df,
        dataset_name=dataset_name,
        params={
            "min_weeks": min_weeks,
            "c": 1.0,
        },
    )
    distance_matrix = temporal_view.get_matrix()
    terminology_ids = temporal_view.get_terminology_ids()
    col_labels = temporal_view.get_col_labels()
    id_to_idx = {tid: idx for idx, tid in enumerate(terminology_ids)}
    return (
        col_labels,
        distance_matrix,
        id_to_idx,
        temporal_view,
        terminology_ids,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Distance Distribution
    """)
    return


@app.cell
def _(distance_matrix, go, np):
    _upper = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    _counts, _bin_edges = np.histogram(_upper, bins=100)
    _bin_centers = (_bin_edges[:-1] + _bin_edges[1:]) / 2
    _median = float(np.median(_upper))

    _fig = go.Figure(data=go.Bar(x=_bin_centers, y=_counts, width=_bin_edges[1] - _bin_edges[0]))
    _fig.add_vline(
        x=_median,
        line_dash="dash",
        annotation_text=f"median={_median:.3f}",
    )
    _fig.update_layout(
        title="Distribution of pairwise MSM distances (temporal view)",
        xaxis_title="MSM distance",
        yaxis_title="Count",
        template="plotly_white",
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Sanity Check — Structural Contrast

    **Reading the table**: higher MSM distance = less similar temporal profiles.

    The core claim: conditions with **opposite temporal structure** (summer
    seasonal vs winter seasonal, acute seasonal vs chronic flat) have
    consistently **higher** MSM distances than conditions sharing the same
    temporal structure.

    Seasonal reference (Guia de Vigilância Epidemiológica, 7ª ed.):

    - **Arboviroses** (dengue, chikungunya, zika): peak in summer/rainy season (Jan-Apr), especially N/NE/SE
    - **Respiratory** (influenza, IVAS): peak in winter (Jun-Aug), stronger in S/SE
    - **Chronic/metabolic** (diabetes, hipertensão): no strong seasonality — flat temporal profile

    **Caveats**: seasonal patterns vary by region. Pairs reflect the dominant
    national pattern and may not hold for every city.

    ### Cross-terminology pairs

    Reported separately below. Same condition coded via different terminologies
    (CID-10, CIAP2, AB) *should* produce similar time series, but this is a
    weaker hypothesis — coding practices, care-setting volume differences, and
    incomplete coverage can inflate distances even when the underlying condition
    is identical.
    """)
    return


@app.cell
def _(get_connection):
    _con = get_connection()
    code_lookup = _con.execute(
        "SELECT id, code, code_type, description FROM medical_terminologies ORDER BY id"
    ).pl()
    return (code_lookup,)


@app.cell
def _(code_lookup, pl):
    def get_id(code: str, code_type: str = "cid10"):
        row = code_lookup.filter((pl.col("code") == code) & (pl.col("code_type") == code_type))
        return None if row.is_empty() else row["id"][0]

    def get_desc(code: str, code_type: str = "cid10"):
        row = code_lookup.filter((pl.col("code") == code) & (pl.col("code_type") == code_type))
        return code if row.is_empty() else row["description"][0]

    return get_desc, get_id


@app.cell
def _(get_desc, get_id):
    STRUCTURAL_PAIRS = [
        # --- SIMILAR temporal structure ---
        # Arboviroses — same summer/rainy season peak
        ("A90", "cid10", "A92", "cid10", "similar", "Dengue & chikungunya — same vector/season"),
        # Respiratory — same winter peak
        ("J10", "cid10", "J11", "cid10", "similar", "Influenza variants — same season"),
        ("J10", "cid10", "J06", "cid10", "similar", "Influenza & IVAS — winter cluster"),
        # Chronic — both flat, no seasonality
        ("E11", "cid10", "I10", "cid10", "similar", "Diabetes & hipertensão — both chronic/flat"),
        # --- OPPOSITE temporal structure ---
        # Arboviroses vs respiratory — opposite seasons in Brazil
        ("A90", "cid10", "J10", "cid10", "opposite", "Dengue (summer) vs influenza (winter)"),
        ("A90", "cid10", "J11", "cid10", "opposite", "Dengue (summer) vs influenza (winter)"),
        ("A92", "cid10", "J10", "cid10", "opposite", "Chikungunya (summer) vs influenza (winter)"),
        # Seasonal vs chronic — different temporal structure
        (
            "A90",
            "cid10",
            "E11",
            "cid10",
            "opposite",
            "Dengue (acute seasonal) vs diabetes (chronic flat)",
        ),
        (
            "A90",
            "cid10",
            "I10",
            "cid10",
            "opposite",
            "Dengue (acute seasonal) vs hipertensão (chronic flat)",
        ),
        (
            "J10",
            "cid10",
            "I10",
            "cid10",
            "opposite",
            "Influenza (acute seasonal) vs hipertensão (chronic flat)",
        ),
        (
            "J10",
            "cid10",
            "E11",
            "cid10",
            "opposite",
            "Influenza (acute seasonal) vs diabetes (chronic flat)",
        ),
    ]

    CROSS_TERMINOLOGY_PAIRS = [
        ("A90", "cid10", "A77", "ciap2", "Dengue CID ↔ CIAP2"),
        ("ABP019", "ab", "A90", "cid10", "Dengue AB ↔ CID"),
        ("R80", "ciap2", "J11", "cid10", "Gripe CIAP2 ↔ Influenza CID"),
        ("R74", "ciap2", "J06", "cid10", "IVAS CIAP2 ↔ CID"),
        ("T90", "ciap2", "E11", "cid10", "Diabetes CIAP2 ↔ CID"),
        ("K86", "ciap2", "I10", "cid10", "Hipertensão CIAP2 ↔ CID"),
        ("ABP006", "ab", "T90", "ciap2", "Diabetes AB ↔ CIAP2"),
        ("ABP005", "ab", "K86", "ciap2", "Hipertensão AB ↔ CIAP2"),
    ]

    sanity_resolved = []
    for _code_a, _type_a, _code_b, _type_b, _expected, _rationale in STRUCTURAL_PAIRS:
        _id_a, _id_b = get_id(_code_a, _type_a), get_id(_code_b, _type_b)
        if _id_a and _id_b:
            sanity_resolved.append(
                {
                    "code_a": f"{_code_a} [{_type_a}] ({get_desc(_code_a, _type_a)[:40]})",
                    "code_b": f"{_code_b} [{_type_b}] ({get_desc(_code_b, _type_b)[:40]})",
                    "id_a": _id_a,
                    "id_b": _id_b,
                    "expected": _expected,
                    "rationale": _rationale,
                }
            )

    cross_terminology_resolved = []
    for _code_a, _type_a, _code_b, _type_b, _rationale in CROSS_TERMINOLOGY_PAIRS:
        _id_a, _id_b = get_id(_code_a, _type_a), get_id(_code_b, _type_b)
        if _id_a and _id_b:
            cross_terminology_resolved.append(
                {
                    "code_a": f"{_code_a} [{_type_a}] ({get_desc(_code_a, _type_a)[:40]})",
                    "code_b": f"{_code_b} [{_type_b}] ({get_desc(_code_b, _type_b)[:40]})",
                    "id_a": _id_a,
                    "id_b": _id_b,
                    "rationale": _rationale,
                }
            )
    return cross_terminology_resolved, sanity_resolved


@app.cell
def _(distance_matrix, id_to_idx, pl, sanity_resolved):
    _rows = []
    for _pair in sanity_resolved:
        _idx_a = id_to_idx.get(_pair["id_a"])
        _idx_b = id_to_idx.get(_pair["id_b"])
        if _idx_a is None or _idx_b is None:
            continue
        _rows.append(
            {
                "code_a": _pair["code_a"],
                "code_b": _pair["code_b"],
                "expected": _pair["expected"],
                "rationale": _pair["rationale"],
                "distance": round(float(distance_matrix[_idx_a, _idx_b]), 4),
            }
        )
    sanity_df = (
        pl.DataFrame(_rows).sort("expected", "distance")
        if _rows
        else pl.DataFrame(
            {"code_a": [], "code_b": [], "expected": [], "rationale": [], "distance": []}
        )
    )
    sanity_df
    return (sanity_df,)


@app.cell
def _(np, pl, px, sanity_df):
    _plot_df = sanity_df.with_columns(
        pl.struct(["code_a", "code_b"])
        .map_elements(
            lambda row: f"{row['code_a'].split(' ')[0]} vs {row['code_b'].split(' ')[0]}",
            return_dtype=pl.Utf8,
        )
        .alias("pair"),
    ).sort("expected", "distance")

    _similar_dists = sanity_df.filter(pl.col("expected") == "similar")["distance"].to_numpy()
    _opposite_dists = sanity_df.filter(pl.col("expected") == "opposite")["distance"].to_numpy()
    _boundary = (
        (np.mean(_similar_dists) + np.mean(_opposite_dists)) / 2
        if len(_similar_dists) and len(_opposite_dists)
        else None
    )

    _fig = px.bar(
        _plot_df,
        x="distance",
        y="pair",
        color="expected",
        facet_row="expected",
        orientation="h",
        hover_data=["code_a", "code_b", "rationale"],
        color_discrete_map={"similar": "#2ca02c", "opposite": "#d62728"},
        title="Structural contrast: similar vs opposite temporal profiles",
        labels={"distance": "MSM distance", "pair": ""},
    )

    if _boundary is not None:
        _fig.add_vline(
            x=_boundary,
            line_dash="dot",
            line_color="gray",
            annotation_text="group boundary",
            annotation_position="top right",
        )

    _fig.update_layout(height=450, showlegend=False)
    _fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    _fig
    return


@app.cell
def _(mo, np, sanity_df):
    _similar = sanity_df.filter(sanity_df["expected"] == "similar")["distance"].to_numpy()
    _opposite = sanity_df.filter(sanity_df["expected"] == "opposite")["distance"].to_numpy()

    _pass = len(_similar) > 0 and len(_opposite) > 0 and np.mean(_similar) < np.mean(_opposite)
    mo.callout(
        mo.md(f"""
        **{"✅ PASS" if _pass else "❌ FAIL"}** ({len(sanity_df)} pairs tested)

        Similar structure mean: `{np.mean(_similar):.4f}` | Opposite structure mean: `{np.mean(_opposite):.4f}`
        """),
        kind="success" if _pass else "danger",
    )
    return


@app.cell
def _(cross_terminology_resolved, distance_matrix, id_to_idx, mo, pl):
    _rows = []
    for _pair in cross_terminology_resolved:
        _idx_a = id_to_idx.get(_pair["id_a"])
        _idx_b = id_to_idx.get(_pair["id_b"])
        if _idx_a is None or _idx_b is None:
            continue
        _rows.append(
            {
                "code_a": _pair["code_a"],
                "code_b": _pair["code_b"],
                "rationale": _pair["rationale"],
                "distance": round(float(distance_matrix[_idx_a, _idx_b]), 4),
            }
        )
    _cross_df = pl.DataFrame(_rows).sort("distance") if _rows else pl.DataFrame()

    mo.vstack(
        [
            mo.md(r"""
        ### Cross-terminology (same condition, different coding)

        These pairs compare the same clinical condition coded via different
        terminologies (CID-10, CIAP2, AB). Low distance is *expected* but
        not guaranteed — coding volume, care setting, and incomplete coverage
        can inflate distances even for identical conditions.
        """),
            _cross_df
            if not _cross_df.is_empty()
            else mo.md("*No cross-terminology pairs found in this dataset.*"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. ILI Anchor Check

    Core influenza codes (J09, J10, J11) should have **lower** distances
    among themselves than to related-but-distinct respiratory codes
    (J00 common cold, J06 acute URI, R80 gripe CIAP2, R74 IVAS CIAP2).

    This tests whether MSM separates the tight influenza cluster from the
    broader respiratory group — not that all are uniformly close.
    """)
    return


@app.cell
def _(code_lookup, distance_matrix, get_id, id_to_idx, mo, np, pl, px):
    CORE_ILI = [("J09", "cid10"), ("J10", "cid10"), ("J11", "cid10")]
    PERIPHERY_ILI = [
        ("J00", "cid10"),
        ("J06", "cid10"),
        ("R80", "ciap2"),
        ("R74", "ciap2"),
    ]

    _all_codes = CORE_ILI + PERIPHERY_ILI
    _resolved = [(c, t, get_id(c, t)) for c, t in _all_codes]
    _resolved = [
        (c, t, i) for c, t, i in _resolved if i is not None and id_to_idx.get(i) is not None
    ]

    _indices = [id_to_idx[i] for _, _, i in _resolved]
    _labels = [
        f"{c} [{t}] ({code_lookup.filter(pl.col('id') == i)['description'][0][:30]})"
        for c, t, i in _resolved
    ]
    _is_core = [any(c == cc and t == ct for cc, ct in CORE_ILI) for c, t, _ in _resolved]

    _submatrix = distance_matrix[np.ix_(_indices, _indices)]

    _fig = px.imshow(
        _submatrix,
        x=_labels,
        y=_labels,
        color_continuous_scale="viridis",
        title="MSM distance — ILI core (J09/J10/J11) and respiratory periphery",
        labels={"color": "MSM distance"},
    )

    _n = len(_indices)
    _core_dists = []
    _cross_dists = []
    for _i in range(_n):
        for _j in range(_i + 1, _n):
            _d = float(_submatrix[_i, _j])
            if _is_core[_i] and _is_core[_j]:
                _core_dists.append(_d)
            else:
                _cross_dists.append(_d)

    _core_mean = np.mean(_core_dists) if _core_dists else float("nan")
    _cross_mean = np.mean(_cross_dists) if _cross_dists else float("nan")
    _pass = _core_mean < _cross_mean

    mo.vstack(
        [
            _fig,
            mo.callout(
                mo.md(f"""
                **{"✅ PASS" if _pass else "❌ FAIL"}** — Core influenza (J09/J10/J11) mean distance: `{_core_mean:.4f}` | Core↔periphery mean: `{_cross_mean:.4f}`

                {"Core cluster is tighter than the broader respiratory group." if _pass else "Core influenza codes are NOT closer to each other than to periphery — investigate."}
                """),
                kind="success" if _pass else "danger",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Time Series Explorer

    Select a diagnosis and inspect its raw time series alongside its
    N nearest temporal neighbours.
    """)
    return


@app.cell
def _(mo):
    search_text = mo.ui.text(
        placeholder="e.g. dengue, influenza, diabetes...",
        label="Search diagnosis",
        full_width=True,
    )
    n_neighbours = mo.ui.slider(start=3, stop=15, value=5, label="Neighbours to show")
    mo.hstack([search_text, n_neighbours])
    return n_neighbours, search_text


@app.cell
def _(code_lookup, df, pl, search_text):
    _term = search_text.value.strip().lower()
    search_matches = (
        code_lookup.filter(
            pl.col("description").str.to_lowercase().str.contains(_term)
            & pl.col("id").is_in(df["terminology_id"].unique())
        ).head(20)
        if len(_term) >= 3
        else pl.DataFrame({"id": [], "code": [], "code_type": [], "description": []})
    )
    return (search_matches,)


@app.cell
def _(mo, search_matches):
    if search_matches.height > 0:
        _opts = {
            f"{r['code']} [{r['code_type']}] {r['description'][:60]}": r["id"]
            for r in search_matches.iter_rows(named=True)
        }
        anchor_dropdown = mo.ui.dropdown(options=_opts, label="Select anchor")
    else:
        anchor_dropdown = mo.md("*Type at least 3 characters.*")
    return (anchor_dropdown,)


@app.cell
def _(anchor_dropdown):
    anchor_dropdown
    return


@app.cell
def _(
    anchor_dropdown,
    code_lookup,
    col_labels,
    distance_matrix,
    id_to_idx,
    mo,
    n_neighbours,
    np,
    pl,
    px,
    search_matches,
    temporal_view,
):
    mo.stop(not (search_matches.height > 0 and anchor_dropdown.value is not None))
    _anchor_id = anchor_dropdown.value
    _anchor_idx = id_to_idx.get(_anchor_id)
    mo.stop(_anchor_idx is None)
    _dists = distance_matrix[_anchor_idx].copy()
    _dists[_anchor_idx] = np.inf
    _top_idx = np.argsort(_dists)[: n_neighbours.value]
    _ids = list(id_to_idx.keys())

    _ts_matrix = temporal_view.get_time_series()
    _plot_rows = []

    _anchor_meta = code_lookup.filter(pl.col("id") == _anchor_id)
    _anchor_label = f"{_anchor_meta['code'][0]} {_anchor_meta['description'][0][:40]} [ANCHOR]"
    for _t, _val in enumerate(_ts_matrix[_anchor_idx]):
        _plot_rows.append(
            {
                "label": _anchor_label,
                "year_week": col_labels[_t],
                "value": float(_val),
                "is_anchor": True,
            }
        )

    for _i in _top_idx:
        _tid = _ids[_i]
        _meta = code_lookup.filter(pl.col("id") == _tid)
        if _meta.is_empty():
            continue
        _lbl = f"{_meta['code'][0]} {_meta['description'][0][:35]} (d={_dists[_i]:.3f})"
        for _t, _val in enumerate(_ts_matrix[_i]):
            _plot_rows.append(
                {
                    "label": _lbl,
                    "year_week": col_labels[_t],
                    "value": float(_val),
                    "is_anchor": False,
                }
            )

    _plot_df = pl.DataFrame(_plot_rows)
    _fig = px.line(
        _plot_df,
        x="year_week",
        y="value",
        color="label",
        title=f"Time series: anchor and {n_neighbours.value} nearest temporal neighbours",
        labels={
            "year_week": "Year-Week",
            "value": "Raw count",
            "label": "Diagnosis",
        },
    )
    _fig.update_layout(height=500, xaxis_tickangle=45)

    _neighbour_rows = []
    for _i in _top_idx:
        _tid = _ids[_i]
        _meta = code_lookup.filter(pl.col("id") == _tid)
        if not _meta.is_empty():
            _neighbour_rows.append(
                {
                    "code": _meta["code"][0],
                    "type": _meta["code_type"][0],
                    "description": _meta["description"][0][:60],
                    "MSM distance": round(float(_dists[_i]), 4),
                }
            )

    mo.vstack(
        [
            _fig,
            mo.md("**Nearest temporal neighbours**"),
            mo.ui.table(pl.DataFrame(_neighbour_rows), selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Coverage Report
    """)
    return


@app.cell
def _(df, mo, terminology_ids):
    _total_codes = df["terminology_id"].n_unique()
    _kept_codes = len(terminology_ids)
    _dropped = _total_codes - _kept_codes

    mo.callout(
        mo.md(f"""
        **Temporal coverage filter**

        Total unique codes in dataset: `{_total_codes}`
        Codes kept (≥ min_weeks): `{_kept_codes}` ({_kept_codes / _total_codes:.1%})
        Codes dropped: `{_dropped}` ({_dropped / _total_codes:.1%})
        """),
        kind="warn" if _dropped / _total_codes > 0.4 else "info",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Cross-City: Intra vs Inter-Chapter Separation

    Tests whether codes within the same CID-10 chapter (e.g. A = infectious,
    J = respiratory, E = endocrine) have lower MSM distance than codes across
    chapters. Three cities with distinct regional epidemiological profiles:

    | City | Region | Climate/Epidemiological profile |
    |------|--------|--------------------------------|
    | Salvador | Nordeste | Tropical, strong arbovirose season, year-round diarreicas |
    | Belo Horizonte | Sudeste | Subtropical transition, both arboviral and respiratory seasons |
    | Gravataí | Sul | Subtropical, strong respiratory/influenza winter peak |

    If the temporal view captures real structure, intra-chapter < inter-chapter
    should hold across all three — despite their different seasonal profiles.

    Only CID-10 codes are used because the chapter-letter hierarchy provides
    a natural ground-truth grouping. CIAP2 and AB use different classification
    structures (CIAP2 chapters overlap with CID but don't map 1:1).
    """)
    return


@app.cell
def _(
    PROCESSED_DATA_DIR,
    TemporalView,
    code_lookup,
    extract_dataset_name,
    load_and_prepare_dataframe,
    min_weeks,
    np,
    pl,
    unicodedata,
    year_max,
    year_min,
):
    def _normalize_ascii(text):
        return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode()

    CROSS_CITY_DATASETS = {
        "salvador": "Nordeste",
        "belo_horizonte": "Sudeste",
        "gravatai": "Sul",
    }

    _tid_to_code = {
        r["id"]: r["code"]
        for r in code_lookup.filter(pl.col("code_type") == "cid10").iter_rows(named=True)
    }

    cross_city_results = []

    for _city_key, _region in CROSS_CITY_DATASETS.items():
        _matches = [
            _f
            for _f in PROCESSED_DATA_DIR.glob("**/prepared_*.parquet")
            if _city_key in _normalize_ascii(str(_f.name).lower())
        ]
        if not _matches:
            continue
        _path = _matches[0]
        _city = extract_dataset_name(str(_path))

        _df = load_and_prepare_dataframe(
            file_path=str(_path),
            year_min=year_min,
            year_max=year_max,
        )

        _view = TemporalView(
            df=_df,
            dataset_name=_city,
            params={"min_weeks": min_weeks, "c": 1.0},
        )
        _matrix = _view.get_matrix()
        _tids = _view.get_terminology_ids()

        _chapters = []
        _valid_indices = []
        for _i, _tid in enumerate(_tids):
            _code = _tid_to_code.get(_tid)
            if _code and len(_code) >= 1 and _code[0].isalpha():
                _chapters.append(_code[0])
                _valid_indices.append(_i)

        if len(_valid_indices) < 10:
            continue

        _chapters_arr = np.array(_chapters)
        _sub_matrix = _matrix[np.ix_(_valid_indices, _valid_indices)]
        _n = len(_valid_indices)

        _row_idx, _col_idx = np.triu_indices(_n, k=1)
        _all_dists = _sub_matrix[_row_idx, _col_idx]
        _same_chapter = _chapters_arr[_row_idx] == _chapters_arr[_col_idx]
        _intra_mean = float(np.mean(_all_dists[_same_chapter]))
        _inter_mean = float(np.mean(_all_dists[~_same_chapter]))

        cross_city_results.append(
            {
                "city": _city,
                "region": _region,
                "n_cid10_codes": _n,
                "n_chapters": len(set(_chapters)),
                "intra_chapter_mean": round(_intra_mean, 4),
                "inter_chapter_mean": round(_inter_mean, 4),
                "separation": round(_inter_mean - _intra_mean, 4),
                "ratio": round(_inter_mean / _intra_mean, 4) if _intra_mean > 0 else None,
                "pass": _intra_mean < _inter_mean,
            }
        )

    cross_city_df = pl.DataFrame(cross_city_results) if cross_city_results else pl.DataFrame()
    cross_city_df
    return (cross_city_df,)


@app.cell
def _(cross_city_df, mo, pl, px):
    mo.stop(cross_city_df.is_empty())

    _n_cities = len(cross_city_df)
    _n_pass = cross_city_df.filter(pl.col("pass"))["pass"].len()

    _fig = px.bar(
        cross_city_df.sort("separation", descending=True),
        x="city",
        y=["intra_chapter_mean", "inter_chapter_mean"],
        barmode="group",
        title="Intra- vs inter-chapter MSM distance by city",
        labels={"value": "Mean MSM distance", "city": "", "variable": ""},
        color_discrete_map={
            "intra_chapter_mean": "#2ca02c",
            "inter_chapter_mean": "#d62728",
        },
    )
    _fig.update_layout(template="plotly_white", height=400)

    _all_pass = bool(cross_city_df["pass"].all())

    _summary = mo.callout(
        mo.md(f"""
        **{"✅ PASS" if _all_pass else "⚠️ PARTIAL"}** — {_n_pass}/{_n_cities} cities show intra < inter

        {"Pattern holds across NE, SE, and S — temporal structure is consistent despite regional differences." if _all_pass else "Some cities violate the expected pattern — investigate coding practices or sample sizes."}
        """),
        kind="success" if _all_pass else "warn",
    )

    mo.vstack([_fig, cross_city_df, _summary])
    return


if __name__ == "__main__":
    app.run()
