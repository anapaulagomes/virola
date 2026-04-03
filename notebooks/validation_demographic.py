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
    import polars as pl
    from dotenv import load_dotenv

    return load_dotenv, np, pl, px


@app.cell
def _():
    from vir.db import get_connection
    from vir.notebook_widgets import (
        create_file_dropdown,
        extract_dataset_name,
        load_and_prepare_dataframe,
    )
    from vir.views.demographic import DemographicView

    return (
        DemographicView,
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
    # Demographic View Validation

    Validates that the chi-square distance matrix captures similarity between
    the age/sex profiles of diagnoses.

    **Key question**: do diagnoses that affect the same demographic group
    (paediatric, elderly, sex-specific) have lower distance than unrelated ones?
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
def _(files_dropdown):
    file_path = files_dropdown.value
    return (file_path,)


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
def _(DemographicView, dataset_name, df):
    demo_view = DemographicView(df=df, dataset_name=dataset_name, params={"pseudocount": 0.5})
    distance_matrix = demo_view.get_matrix()
    terminology_ids = demo_view.get_terminology_ids()
    id_to_idx = {tid: idx for idx, tid in enumerate(terminology_ids)}
    print(f"Matrix shape: {distance_matrix.shape}")
    return demo_view, distance_matrix, id_to_idx


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Distance Distribution

    Should be skewed right. A bell shape centred in the middle suggests
    the pseudocount is over-smoothing distributions.
    """)
    return


@app.cell
def _(distance_matrix, mo, np, px):
    _upper = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    _fig = px.histogram(
        x=_upper,
        nbins=100,
        title="Distribution of pairwise chi-square distances (demographic view)",
        labels={"x": "Chi-square distance"},
    )
    _fig.add_vline(
        x=float(np.median(_upper)),
        line_dash="dash",
        annotation_text=f"median={np.median(_upper):.3f}",
    )
    _path = "/tmp/virola_demographic_dist.png"
    _fig.write_image(_path)
    mo.image(src=_path)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Sanity Check — Known Pairs
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
    SANITY_PAIRS = [
        ("B05", "cid10", "B06", "cid10", "close", "Measles & rubella — paediatric"),
        ("B05", "cid10", "B01", "cid10", "close", "Measles & varicella — paediatric"),
        ("I10", "cid10", "E11", "cid10", "close", "Hypertension & diabetes — adult/elderly"),
        ("M81", "cid10", "I10", "cid10", "close", "Osteoporosis & hypertension — elderly"),
        ("B05", "cid10", "I10", "cid10", "far", "Measles (child) vs hypertension (adult)"),
        ("B01", "cid10", "M81", "cid10", "far", "Varicella (child) vs osteoporosis (elderly)"),
        ("N91", "cid10", "N40", "cid10", "far", "Amenorrhoea (female) vs prostate (male)"),
        ("B05", "cid10", "N40", "cid10", "far", "Measles (child) vs prostate (adult male)"),
    ]

    sanity_resolved = []
    for ca, ta, cb, tb, expected, rationale in SANITY_PAIRS:
        id_a, id_b = get_id(ca, ta), get_id(cb, tb)
        if id_a and id_b:
            sanity_resolved.append(
                {
                    "code_a": f"{ca} ({get_desc(ca, ta)[:40]})",
                    "code_b": f"{cb} ({get_desc(cb, tb)[:40]})",
                    "id_a": id_a,
                    "id_b": id_b,
                    "expected": expected,
                    "rationale": rationale,
                }
            )
        else:
            print(f"Not found: {ca}/{ta} or {cb}/{tb}")
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
        title="Sanity check: chi-square distance for known demographic pairs",
        labels={"distance": "Chi-square distance", "code_a": "Pair (A)"},
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
    ## 3. Uniformity Check

    Diagnoses with near-uniform demographic distributions carry no signal in this view.
    High numbers suggest the view has limited discriminating power for those codes.
    """)
    return


@app.cell
def _(demo_view, np, px):
    norm_df = demo_view.get_normalized_distributions()
    entropy = -(np.where(norm_df > 0, norm_df * np.log(norm_df + 1e-12), 0)).sum(axis=1)
    max_entropy = np.log(norm_df.shape[1])
    relative_entropy = entropy / (max_entropy + 1e-12)

    _fig = px.histogram(
        x=relative_entropy,
        nbins=50,
        title="Relative entropy of demographic distributions (1.0 = uniform)",
        labels={"x": "Relative entropy"},
    )
    _fig.add_vline(
        x=0.9, line_dash="dash", line_color="red", annotation_text="high uniformity (0.9)"
    )
    _fig
    return (relative_entropy,)


@app.cell
def _(mo, np, relative_entropy):
    _n_uniform = int(np.sum(relative_entropy > 0.9))
    _total = len(relative_entropy)
    mo.callout(
        mo.md(f"""
        Near-uniform distributions: **{_n_uniform} / {_total} ({_n_uniform / _total:.1%})**

        These codes have no demographic signal and will cluster randomly in this view.
        """),
        kind="warn" if _n_uniform / _total > 0.3 else "info",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Nearest Neighbours Explorer
    """)
    return


@app.cell
def _(mo):
    search_text = mo.ui.text(
        placeholder="e.g. varicela, diabetes, osteoporose...",
        label="Search diagnosis",
        full_width=True,
    )
    n_neighbours = mo.ui.slider(start=5, stop=30, value=10, label="Neighbours")
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
    df,
    distance_matrix,
    id_to_idx,
    mo,
    n_neighbours,
    np,
    pl,
    search_matches,
):
    mo.stop(not (search_matches.height > 0 and anchor_dropdown.value is not None))
    _anchor_idx = id_to_idx.get(anchor_dropdown.value)
    mo.stop(_anchor_idx is None)
    _dists = distance_matrix[_anchor_idx].copy()
    _dists[_anchor_idx] = np.inf
    _top_idx = np.argsort(_dists)[: n_neighbours.value]
    _ids = list(id_to_idx.keys())

    _rows = []
    for _i in _top_idx:
        _tid = _ids[_i]
        _meta = code_lookup.filter(pl.col("id") == _tid)
        if _meta.is_empty():
            continue
        _cdf = df.filter(pl.col("terminology_id") == _tid)
        _total = _cdf["quantity"].sum() or 1
        _female = float(_cdf.filter(pl.col("sex") == "Feminino")["quantity"].sum() / _total)
        _child = float(
            _cdf.filter(pl.col("age_group").is_in(["< 1 ano", "1 a 15 anos"]))["quantity"].sum()
            / _total
        )
        _rows.append(
            {
                "rank": len(_rows) + 1,
                "code": _meta["code"][0],
                "type": _meta["code_type"][0],
                "description": _meta["description"][0][:60],
                "distance": round(float(_dists[_i]), 4),
                "female_%": round(_female * 100, 1),
                "child_%": round(_child * 100, 1),
            }
        )

    mo.vstack(
        [
            mo.md("**Nearest neighbours by demographic profile**"),
            mo.ui.table(pl.DataFrame(_rows), selection=None),
        ]
    )


if __name__ == "__main__":
    app.run()
