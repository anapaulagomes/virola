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
    from vir.db import get_connection, get_cross_terminology_pairs
    from vir.helpers import extract_dataset_name
    from vir.notebook_widgets import (
        create_file_dropdown,
        load_and_prepare_dataframe,
    )
    from vir.views.clinical import ClinicalView

    return (
        ClinicalView,
        create_file_dropdown,
        extract_dataset_name,
        get_connection,
        get_cross_terminology_pairs,
        load_and_prepare_dataframe,
    )


@app.cell
def _(load_dotenv):
    _ = load_dotenv()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Clinical View Validation 🧬

    Validates that the cosine distance matrix from SapBERT embeddings captures
    semantic similarity between diagnoses.

    **Key question**: do semantically similar diagnoses (same ICD chapter, related
    concepts) have lower cosine distance than unrelated ones?
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
    return year_max, year_min


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
def _(extract_dataset_name, file_path, load_and_prepare_dataframe, year_max, year_min):
    df = load_and_prepare_dataframe(
        file_path=file_path,
        year_min=year_min,
        year_max=year_max,
    )
    dataset_name = extract_dataset_name(file_path)
    return dataset_name, df


@app.cell
def _(files_dropdown):
    file_path = files_dropdown.value
    return (file_path,)


@app.cell
def _(ClinicalView, dataset_name, df):
    clinical_view = ClinicalView(df=df, dataset_name=dataset_name)
    distance_matrix = clinical_view.get_matrix()
    terminology_ids = clinical_view.get_terminology_ids()
    id_to_idx = {tid: idx for idx, tid in enumerate(terminology_ids)}
    print(f"Matrix shape: {distance_matrix.shape}")
    return clinical_view, distance_matrix, id_to_idx, terminology_ids


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Distance Distribution

    The overall distribution of pairwise cosine distances.
    Should be skewed right — most pairs are dissimilar, few are very close.
    """)
    return


@app.cell
def _(distance_matrix, mo, np, px):
    _upper = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    _fig = px.histogram(
        x=_upper,
        nbins=100,
        title="Distribution of pairwise cosine distances (clinical view)",
        labels={"x": "Cosine distance"},
    )
    _fig.add_vline(
        x=float(np.median(_upper)),
        line_dash="dash",
        annotation_text=f"median={np.median(_upper):.3f}",
    )
    _path = "/tmp/virola_clinical_dist.png"
    _fig.write_image(_path)
    mo.image(src=_path)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Sanity Check — Known Pairs

    Manually defined pairs that should be **close** (same concept / related) or
    **far** (different clinical domains).  Uses ICD-10 codes present in the dataset.
    """)
    return


@app.cell
def _(get_connection):
    # Fetch code → terminology_id mapping for lookup
    _con = get_connection()
    code_lookup = _con.execute(
        "SELECT id, code, code_type, description FROM medical_terminologies ORDER BY id"
    ).pl()
    return (code_lookup,)


@app.cell
def _(code_lookup, pl):
    def get_id(code: str, code_type: str = "cid10"):
        row = code_lookup.filter((pl.col("code") == code) & (pl.col("code_type") == code_type))
        if row.is_empty():
            return None
        return row["id"][0]

    def get_desc(code: str, code_type: str = "cid10"):
        row = code_lookup.filter((pl.col("code") == code) & (pl.col("code_type") == code_type))
        if row.is_empty():
            return code
        return row["description"][0]

    return get_desc, get_id


@app.cell
def _(get_desc, get_id):
    # (code_a, code_type_a, code_b, code_type_b, expected, rationale)
    SANITY_PAIRS = [
        # --- should be CLOSE ---
        ("J09", "cid10", "J10", "cid10", "close", "Influenza variants"),
        ("J10", "cid10", "J11", "cid10", "close", "Influenza variants"),
        ("A90", "cid10", "A91", "cid10", "close", "Dengue / dengue haemorrhagic"),
        ("A90", "cid10", "A92", "cid10", "close", "Dengue vs other mosquito-borne"),
        ("E11", "cid10", "E14", "cid10", "close", "Diabetes type 2 vs unspecified"),
        ("I10", "cid10", "I11", "cid10", "close", "Hypertension variants"),
        # --- should be FAR ---
        ("J11", "cid10", "E11", "cid10", "far", "Influenza vs diabetes"),
        ("A90", "cid10", "I10", "cid10", "far", "Dengue vs hypertension"),
        ("J10", "cid10", "K29", "cid10", "far", "Influenza vs gastritis"),
        ("E11", "cid10", "J45", "cid10", "far", "Diabetes vs asthma"),
    ]

    # Resolve to ids with descriptions
    sanity_resolved = []
    for _code_a, _type_a, _code_b, _type_b, _expected, _rationale in SANITY_PAIRS:
        _id_a = get_id(_code_a, _type_a)
        _id_b = get_id(_code_b, _type_b)
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
            print(f"⚠️  Not found in dataset: {_code_a}/{_type_a} or {_code_b}/{_type_b}")
    return (sanity_resolved,)


@app.cell
def _(distance_matrix, id_to_idx, pl, sanity_resolved):
    rows = []
    for _pair in sanity_resolved:
        _idx_a = id_to_idx.get(_pair["id_a"])
        _idx_b = id_to_idx.get(_pair["id_b"])
        if _idx_a is None or _idx_b is None:
            continue
        _dist = float(distance_matrix[_idx_a, _idx_b])
        rows.append(
            {
                "code_a": _pair["code_a"],
                "code_b": _pair["code_b"],
                "expected": _pair["expected"],
                "rationale": _pair["rationale"],
                "distance": round(_dist, 4),
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
    _fig = px.bar(
        sanity_df,
        x="distance",
        y="code_a",
        color="expected",
        orientation="h",
        hover_data=["code_b", "rationale"],
        color_discrete_map={"close": "#2ca02c", "far": "#d62728"},
        title="Sanity check: cosine distance for known pairs",
        labels={"distance": "Cosine distance", "code_a": "Pair (A → B)"},
    )
    _fig.update_layout(height=450)
    _fig
    return


@app.cell
def _(mo, np, sanity_df):
    _n_tested = len(sanity_df)
    _n_total = 10  # total pairs defined in SANITY_PAIRS
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
        _status = "✅ PASS" if _pass else "❌ FAIL"
        _result = mo.callout(
            mo.md(f"""
            **Sanity check result: {_status}** ({_n_tested}/{_n_total} pairs tested)

            Mean distance — close pairs: `{np.mean(_close):.4f}` | far pairs: `{np.mean(_far):.4f}`
            """),
            kind="success" if _pass else "danger",
        )
        mo.vstack([x for x in [_coverage_warning, _result] if x is not None])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Intra- vs Inter-Chapter Distance

    Diagnoses within the same ICD-10 chapter should have lower distance
    than diagnoses across chapters.
    """)
    return


@app.cell
def _(code_lookup, distance_matrix, id_to_idx, np, pl, terminology_ids):
    # Assign ICD chapter (first letter of code) to each terminology_id
    icd_codes = code_lookup.filter(pl.col("code_type") == "cid10")
    id_to_chapter = {
        row["id"]: row["code"][0]  # first letter = chapter
        for row in icd_codes.iter_rows(named=True)
        if row["id"] in id_to_idx
    }

    intra, inter = [], []
    ids_list = list(terminology_ids)
    for i in range(len(ids_list)):
        for j in range(i + 1, len(ids_list)):
            ch_i = id_to_chapter.get(ids_list[i])
            ch_j = id_to_chapter.get(ids_list[j])
            if ch_i is None or ch_j is None:
                continue
            d = float(distance_matrix[i, j])
            if ch_i == ch_j:
                intra.append(d)
            else:
                inter.append(d)

    intra_arr = np.array(intra)
    inter_arr = np.array(inter)
    print(f"Intra-chapter pairs: {len(intra_arr):,} | Inter-chapter pairs: {len(inter_arr):,}")
    return inter_arr, intra_arr


@app.cell
def _(go, inter_arr, intra_arr, mo, np):
    _fig = go.Figure()
    _fig.add_trace(
        go.Histogram(
            x=intra_arr, name="Intra-chapter", opacity=0.7, nbinsx=80, marker_color="#2ca02c"
        )
    )
    _fig.add_trace(
        go.Histogram(
            x=inter_arr, name="Inter-chapter", opacity=0.7, nbinsx=80, marker_color="#d62728"
        )
    )
    _fig.update_layout(
        barmode="overlay",
        title="Cosine distance: intra-chapter vs inter-chapter (ICD-10)",
        xaxis_title="Cosine distance",
        yaxis_title="Count",
    )
    _fig.add_vline(x=float(np.mean(intra_arr)), line_dash="dash", line_color="#2ca02c")
    _fig.add_vline(x=float(np.mean(inter_arr)), line_dash="dash", line_color="#d62728")
    # annotation_text=f"inter mean={np.mean(inter_arr):.3f}"
    _path = "/tmp/cosine_distance_intra_inter.png"
    _fig.write_image(_path)
    mo.image(src=_path)
    return


@app.cell
def _(inter_arr, intra_arr, mo, np):
    _pass = np.mean(intra_arr) < np.mean(inter_arr)
    _status = "✅ PASS" if _pass else "❌ FAIL"
    mo.callout(
        mo.md(f"""
        **Intra vs inter chapter: {_status}**

        Mean intra-chapter distance: `{np.mean(intra_arr):.4f}`
        Mean inter-chapter distance: `{np.mean(inter_arr):.4f}`
        """),
        kind="success" if _pass else "danger",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Cross-Terminology Check

    Same concept expressed in different terminologies (ICD vs CIAP2) should be close.
    Uses known mappings between ICD-10 and CIAP2 codes.
    """)
    return


@app.cell
def _(get_cross_terminology_pairs, get_desc, get_id):
    cross_resolved = []
    for _code_a, _type_a, _code_b, _type_b, _ab_code in get_cross_terminology_pairs():
        _id_a = get_id(_code_a, _type_a)
        _id_b = get_id(_code_b, _type_b)
        if _id_a and _id_b:
            cross_resolved.append(
                {
                    "ab_code": _ab_code,
                    "code_a": f"{_code_a} [{_type_a}] {get_desc(_code_a, _type_a)[:35]}",
                    "code_b": f"{_code_b} [{_type_b}] {get_desc(_code_b, _type_b)[:35]}",
                    "id_a": _id_a,
                    "id_b": _id_b,
                }
            )
    print(f"{len(cross_resolved)} cross-terminology pairs resolved from reference mapping")
    return (cross_resolved,)


@app.cell
def _(cross_resolved, distance_matrix, id_to_idx, pl):
    cross_rows = []
    for _pair in cross_resolved:
        _idx_a = id_to_idx.get(_pair["id_a"])
        _idx_b = id_to_idx.get(_pair["id_b"])
        if _idx_a is None or _idx_b is None:
            continue
        _dist = float(distance_matrix[_idx_a, _idx_b])
        cross_rows.append(
            {
                "ab_code": _pair["ab_code"],
                "ICD code": _pair["code_a"],
                "CIAP2 code": _pair["code_b"],
                "cross-terminology distance": round(_dist, 4),
            }
        )

    cross_df = (
        pl.DataFrame(cross_rows).sort("cross-terminology distance")
        if cross_rows
        else pl.DataFrame(
            {"ab_code": [], "ICD code": [], "CIAP2 code": [], "cross-terminology distance": []}
        )
    )
    cross_df
    return (cross_df,)


@app.cell
def _(cross_df, intra_arr, mo, np):
    _n_pairs = len(cross_df)
    if _n_pairs == 0:
        mo.callout(mo.md("⚠️ No cross-terminology pairs found in this dataset."), kind="warn")
    else:
        _mean_cross = float(np.mean(cross_df["cross-terminology distance"].to_numpy()))
        _mean_intra = float(np.mean(intra_arr))
        _pass = _mean_cross < _mean_intra
        _status = "✅ PASS" if _pass else "❌ FAIL"
        mo.callout(
            mo.md(f"""
            **Cross-terminology check: {_status}** ({_n_pairs} pairs tested)

            Mean cross-terminology distance: `{_mean_cross:.4f}`
            Mean intra-chapter distance (reference): `{_mean_intra:.4f}`

            Same concept expressed in ICD-10 and CIAP2 should be closer than
            randomly paired codes within the same ICD chapter.
            """),
            kind="success" if _pass else "danger",
        )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Nearest Neighbours Explorer

    For a selected diagnosis, inspect its N closest neighbours.
    Useful for a qualitative sanity check.
    """)
    return


@app.cell
def _(mo):
    search_text = mo.ui.text(
        placeholder="e.g. influenza, dengue, diabetes...",
        label="Search diagnosis",
        full_width=True,
    )
    n_neighbours = mo.ui.slider(start=5, stop=30, value=10, label="Neighbours to show")
    mo.hstack([search_text, n_neighbours])
    return n_neighbours, search_text


@app.cell
def _(code_lookup, df, pl, search_text):
    _term = search_text.value.strip().lower()
    if len(_term) >= 3:
        _matches = code_lookup.filter(
            pl.col("description").str.to_lowercase().str.contains(_term)
            & pl.col("id").is_in(df["terminology_id"].unique())
        ).head(20)
    else:
        _matches = pl.DataFrame({"id": [], "code": [], "code_type": [], "description": []})
    search_matches = _matches
    return (search_matches,)


@app.cell
def _(mo, search_matches):
    if search_matches.height > 0:
        _opts = {
            f"{r['code']} [{r['code_type']}] {r['description'][:60]}": r["id"]
            for r in search_matches.iter_rows(named=True)
        }
        anchor_dropdown = mo.ui.dropdown(options=_opts, label="Select anchor diagnosis")
    else:
        anchor_dropdown = mo.md("*No matches — type at least 3 characters.*")
    return (anchor_dropdown,)


@app.cell
def _(anchor_dropdown):
    anchor_dropdown
    return


@app.cell
def _(
    anchor_dropdown,
    code_lookup,
    distance_matrix,
    id_to_idx,
    mo,
    n_neighbours,
    np,
    pl,
    search_matches,
):
    mo.stop(not (search_matches.height > 0 and anchor_dropdown.value is not None))
    _anchor_id = anchor_dropdown.value
    _anchor_idx = id_to_idx.get(_anchor_id)
    mo.stop(_anchor_idx is None)
    _dists = distance_matrix[_anchor_idx].copy()
    _dists[_anchor_idx] = np.inf  # exclude self
    _top_idx = np.argsort(_dists)[: n_neighbours.value]
    _ids = list(id_to_idx.keys())

    _rows = []
    for _i in _top_idx:
        _tid = _ids[_i]
        _meta = code_lookup.filter(pl.col("id") == _tid)
        if not _meta.is_empty():
            _rows.append(
                {
                    "rank": len(_rows) + 1,
                    "code": _meta["code"][0],
                    "code_type": _meta["code_type"][0],
                    "description": _meta["description"][0],
                    "distance": round(float(_dists[_i]), 4),
                }
            )

    mo.vstack(
        [
            mo.md("**Nearest neighbours of selected diagnosis**"),
            mo.ui.table(pl.DataFrame(_rows), selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Embedding Statistics
    """)
    return


@app.cell
def _(clinical_view, mo):
    _stats = clinical_view.get_embedding_stats()
    mo.vstack(
        [
            mo.md("**Embedding stats**"),
            mo.ui.table([_stats], selection=None),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
