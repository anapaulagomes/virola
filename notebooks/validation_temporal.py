import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    from dotenv import load_dotenv
    return go, load_dotenv, np, pl, px


@app.cell
def _():
    from vir.db import get_connection
    from vir.helpers import extract_dataset_name
    from vir.notebook_widgets import (
        create_file_dropdown,
        load_and_prepare_dataframe,
    )
    from vir.views.temporal import TemporalView
    return (
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
    # Temporal View Validation

    Validates that the MSM distance matrix captures epidemiological similarity
    between the time series of diagnoses.

    Series are chronological year_week sequences — surges and secular trends
    are preserved (not flattened into a single annual cycle).

    **Key question**: do diagnoses with similar temporal behaviour (same
    seasonality, co-occurring outbreaks) have lower MSM distance?
    """)
    return


@app.cell
def _(mo):
    args = mo.cli_args() if mo.app_meta().mode == "script" else {}
    return (args,)


@app.cell
def _(args):
    year_min = args.get("year_min", 2022)
    year_max = args.get("year_max", 2025)
    min_weeks = args.get("min_weeks", 12)
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

    print(f"Matrix shape:  {distance_matrix.shape}")
    print(f"Series length: {len(col_labels)} year_weeks ({col_labels[0]} → {col_labels[-1]})")
    return col_labels, distance_matrix, id_to_idx, temporal_view, terminology_ids


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Distance Distribution

    Should be skewed right — most diagnoses have different temporal profiles.
    """)
    return


@app.cell
def _(distance_matrix, mo, np, px):
    _upper = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    _fig = px.histogram(
        x=_upper,
        nbins=100,
        title="Distribution of pairwise MSM distances (temporal view)",
        labels={"x": "MSM distance"},
    )
    _fig.add_vline(
        x=float(np.median(_upper)),
        line_dash="dash",
        annotation_text=f"median={np.median(_upper):.3f}",
    )
    _path = "/tmp/virola_temporal_dist.png"
    _fig.write_image(_path)
    mo.image(src=_path)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Sanity Check — Known Pairs

    Pairs that should be **close** (same seasonality / co-occurring outbreaks)
    or **far** (opposite seasonal patterns in Brazil).

    Brazil reference:
    - Arboviroses (dengue, chikungunya, zika): peak in summer/rainy season (Jan-Apr)
    - Respiratory (influenza, RSV): peak in winter (Jun-Aug, SE/S regions)
    - Chronic/metabolic: no strong seasonality
    """)
    return


@app.cell
def _(get_connection, pl):
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
    SANITY_PAIRS = [
        # --- should be CLOSE ---
        ("A90", "cid10", "A92", "cid10", "close", "Dengue & other arboviral — same season"),
        ("A90", "cid10", "A96", "cid10", "close", "Dengue & haemorrhagic fever — related"),
        ("J10", "cid10", "J11", "cid10", "close", "Influenza variants — same season"),
        ("J10", "cid10", "J06", "cid10", "close", "Influenza & acute URI — winter cluster"),
        # --- should be FAR ---
        ("A90", "cid10", "J10", "cid10", "far", "Dengue (summer) vs influenza (winter)"),
        ("A90", "cid10", "J11", "cid10", "far", "Dengue (summer) vs influenza (winter)"),
        ("A90", "cid10", "E11", "cid10", "far", "Dengue (seasonal) vs diabetes (chronic)"),
        ("J10", "cid10", "I10", "cid10", "far", "Influenza (seasonal) vs hypertension (chronic)"),
    ]

    sanity_resolved = []
    for _code_a, _type_a, _code_b, _type_b, _expected, _rationale in SANITY_PAIRS:
        _id_a, _id_b = get_id(_code_a, _type_a), get_id(_code_b, _type_b)
        if _id_a and _id_b:
            sanity_resolved.append(
                {
                    "code_a": f"{_code_a} ({get_desc(_code_a, _type_a)[:40]})",
                    "code_b": f"{_code_b} ({get_desc(_code_b, _type_b)[:40]})",
                    "id_a": _id_a,
                    "id_b": _id_b,
                    "expected": _expected,
                    "rationale": _rationale,
                }
            )
        else:
            print(f"Not found in dataset: {_code_a}/{_type_a} or {_code_b}/{_type_b}")
    return (sanity_resolved,)


@app.cell
def _(distance_matrix, id_to_idx, pl, sanity_resolved):
    rows = []
    for pair in sanity_resolved:
        idx_a = id_to_idx.get(pair["id_a"])
        idx_b = id_to_idx.get(pair["id_b"])
        if idx_a is None or idx_b is None:
            continue
        rows.append(
            {
                "code_a": pair["code_a"],
                "code_b": pair["code_b"],
                "expected": pair["expected"],
                "rationale": pair["rationale"],
                "distance": round(float(distance_matrix[idx_a, idx_b]), 4),
            }
        )
    sanity_df = (
        pl.DataFrame(rows).sort("expected", "distance")
        if rows
        else pl.DataFrame(
            {"code_a": [], "code_b": [], "expected": [], "rationale": [], "distance": []}
        )
    )
    sanity_df
    return (sanity_df,)


@app.cell
def _(px, sanity_df):
    px.bar(
        sanity_df,
        x="distance",
        y="code_a",
        color="expected",
        orientation="h",
        hover_data=["code_b", "rationale"],
        color_discrete_map={"close": "#2ca02c", "far": "#d62728"},
        title="Sanity check: MSM distance for known temporal pairs",
        labels={"distance": "MSM distance", "code_a": "Pair (A)"},
    ).update_layout(height=420)
    return


@app.cell
def _(mo, np, sanity_df):
    _n_tested = len(sanity_df)
    _n_total = 8  # total pairs defined in SANITY_PAIRS
    _close = sanity_df.filter(sanity_df["expected"] == "close")["distance"].to_numpy()
    _far = sanity_df.filter(sanity_df["expected"] == "far")["distance"].to_numpy()

    _coverage_warning = (
        mo.callout(
            mo.md(
                f"⚠️ **Low coverage**: only {_n_tested}/{_n_total} pairs found in this dataset. Result may not be representative."
            ),
            kind="warn",
        )
        if _n_tested < _n_total // 2
        else None
    )

    if len(_close) and len(_far):
        _pass = np.mean(_close) < np.mean(_far)
        _result = mo.callout(
            mo.md(f"""
            **{"✅ PASS" if _pass else "❌ FAIL"}** ({_n_tested}/{_n_total} pairs tested)
            Close pairs mean: `{np.mean(_close):.4f}` | Far pairs mean: `{np.mean(_far):.4f}`
            """),
            kind="success" if _pass else "danger",
        )
        mo.vstack([x for x in [_coverage_warning, _result] if x is not None])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. ILI Anchor Check

    ILI codes (J09, J10, J11, J06, J00) should form a tight temporal cluster.
    This is the primary anchor for paper validation.
    """)
    return


@app.cell
def _(code_lookup, distance_matrix, get_id, id_to_idx, mo, np, pl, px):
    ILI_CODES = [
        ("J09", "cid10"),
        ("J10", "cid10"),
        ("J11", "cid10"),
        ("J06", "cid10"),
        ("J00", "cid10"),
    ]

    ili_ids = [(code, ctype, get_id(code, ctype)) for code, ctype in ILI_CODES]
    ili_ids = [(c, t, i) for c, t, i in ili_ids if i is not None and id_to_idx.get(i) is not None]

    if len(ili_ids) >= 2:
        ili_indices = [id_to_idx[i] for _, _, i in ili_ids]
        ili_labels = [
            f"{c} ({code_lookup.filter(pl.col('id') == i)['description'][0][:30]})"
            for c, _, i in ili_ids
        ]

        # Submatrix of ILI codes
        ili_submatrix = distance_matrix[np.ix_(ili_indices, ili_indices)]

        _fig = px.imshow(
            ili_submatrix,
            x=ili_labels,
            y=ili_labels,
            color_continuous_scale="viridis",
            title="MSM distance — ILI codes (should be low/uniform)",
            labels={"color": "MSM distance"},
        )
        _mean_ili = float(np.mean(ili_submatrix[np.triu_indices_from(ili_submatrix, k=1)]))
        mo.vstack(
            [
                _fig,
                mo.callout(
                    mo.md(f"Mean pairwise MSM distance within ILI codes: `{_mean_ili:.4f}`"),
                    kind="info",
                ),
            ]
        )
    else:
        mo.callout(mo.md("ILI codes not found in current dataset."), kind="warn")
    return ILI_CODES, ili_ids, ili_indices, ili_labels, ili_submatrix


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Time Series Explorer

    Select a diagnosis and inspect its raw time series alongside its
    N nearest temporal neighbours. Validates that neighbours share a
    visually recognisable pattern.
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

    # Build time series plot
    _ts_matrix = temporal_view.get_time_series()
    _plot_rows = []

    # Anchor series
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

    # Neighbour series
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
            "value": "Z-score normalised count",
            "label": "Diagnosis",
        },
    )
    _fig.update_layout(height=500, xaxis_tickangle=45)

    # Neighbour table
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


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Coverage Report

    How many codes passed the `min_weeks` filter vs. total codes in the dataset.
    Large drop-off suggests the threshold is too strict.
    """)
    return


@app.cell
def _(df, mo, pl, terminology_ids):
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


if __name__ == "__main__":
    app.run()
