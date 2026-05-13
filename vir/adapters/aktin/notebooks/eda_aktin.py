import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import plotly.express as px
    import polars as pl

    from vir.config import INTERIM_DATA_DIR

    return INTERIM_DATA_DIR, mo, pl, px


@app.cell
def _(mo):
    mo.md(r"""
    # AKTIN — Exploratory data analysis

    Basic shape, demographic distribution, code coverage, and temporal distribution
    of the cleaned AKTIN interim parquet (one row per encounter × confirmed
    diagnosis).
    """)
    return


@app.cell
def _(INTERIM_DATA_DIR, mo):
    available_files = sorted(INTERIM_DATA_DIR.glob("cleaned_aktin_*.parquet"))
    file_dropdown = mo.ui.dropdown(
        options={path.name: str(path) for path in available_files},
        value=available_files[0].name if available_files else None,
        label="Cleaned AKTIN parquet",
    )
    file_dropdown
    return (file_dropdown,)


@app.cell
def _(file_dropdown, mo, pl):
    mo.stop(
        not file_dropdown.value,
        mo.callout(
            mo.md("No `cleaned_aktin_*.parquet` files found under `INTERIM_DATA_DIR`."),
            kind="warn",
        ),
    )
    df = pl.read_parquet(file_dropdown.value)
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Shape
    """)
    return


@app.cell
def _(df, mo):
    n_rows = df.height
    n_encounters = df["a_encounter_num"].n_unique()
    n_codes = df["code"].n_unique()
    year_min = df["year"].min()
    year_max = df["year"].max()
    week_min = df["week"].min()
    week_max = df["week"].max()
    diagnoses_per_encounter = n_rows / n_encounters if n_encounters else 0

    mo.md(f"""
    | metric | value |
    |---|---|
    | rows (encounter × diagnosis) | {n_rows:,} |
    | distinct encounters | {n_encounters:,} |
    | distinct ICD-10-GM codes | {n_codes:,} |
    | mean diagnoses per encounter | {diagnoses_per_encounter:.2f} |
    | ISO year range | {year_min} – {year_max} |
    | ISO week range | {week_min} – {week_max} |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Demographics
    """)
    return


@app.cell
def _():
    age_order = ["0-4", "5-9", "10-14", "15-19", "20-39", "40-59", "60-79", "80+"]
    return (age_order,)


@app.cell
def _(df, mo, px):
    sex_per_encounter = df.unique(subset=["a_encounter_num"]).group_by("sex").len().sort("sex")
    sex_chart = px.bar(
        sex_per_encounter,
        x="sex",
        y="len",
        title="Encounters by sex",
        labels={"len": "encounters"},
    )
    mo.ui.plotly(sex_chart)
    return


@app.cell
def _(age_order, df, mo, pl, px):
    age_per_encounter = (
        df.unique(subset=["a_encounter_num"])
        .group_by("age_group")
        .len()
        .with_columns(pl.col("age_group").cast(pl.Enum(age_order)).alias("age_group_ordered"))
        .sort("age_group_ordered")
    )
    age_chart = px.bar(
        age_per_encounter,
        x="age_group",
        y="len",
        title="Encounters by age group",
        category_orders={"age_group": age_order},
        labels={"len": "encounters"},
    )
    mo.ui.plotly(age_chart)
    return


@app.cell
def _(age_order, df, mo, px):
    sex_age = df.unique(subset=["a_encounter_num"]).group_by(["age_group", "sex"]).len()
    sex_age_chart = px.bar(
        sex_age,
        x="age_group",
        y="len",
        color="sex",
        barmode="group",
        title="Encounters by age group × sex",
        category_orders={"age_group": age_order},
        labels={"len": "encounters"},
    )
    mo.ui.plotly(sex_age_chart)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## ICD-10-GM coverage
    """)
    return


@app.cell
def _(df, mo, pl, px):
    chapter_distribution = (
        df.with_columns(pl.col("code").str.slice(0, 1).alias("chapter_letter"))
        .group_by("chapter_letter")
        .len()
        .sort("chapter_letter")
    )
    chapter_chart = px.bar(
        chapter_distribution,
        x="chapter_letter",
        y="len",
        title="Diagnoses by ICD-10-GM chapter letter",
        labels={"len": "diagnoses", "chapter_letter": "chapter (A–Z, U)"},
    )
    mo.ui.plotly(chapter_chart)
    return


@app.cell
def _(df, mo, px):
    top_codes = df.group_by("code").len().sort("len", descending=True).head(20)
    top_codes_chart = px.bar(
        top_codes,
        x="code",
        y="len",
        title="Top-20 ICD-10-GM codes by diagnosis count",
        labels={"len": "diagnoses"},
    )
    mo.ui.plotly(top_codes_chart)
    return (top_codes,)


@app.cell
def _(mo, top_codes):
    mo.ui.table(top_codes, label="Top-20 codes")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Temporal distribution
    """)
    return


@app.cell
def _(df, pl):
    has_aufnahme_ts = "aufnahme_ts" in df.columns
    timestamps_df = (
        df.with_columns(
            pl.col("aufnahme_ts").str.to_datetime("%Y-%m-%dT%H:%M:%SZ").alias("_admission")
        )
        if has_aufnahme_ts
        else df
    )
    return has_aufnahme_ts, timestamps_df


@app.cell
def _(has_aufnahme_ts, mo, pl, px, timestamps_df):
    mo.stop(
        not has_aufnahme_ts,
        mo.callout(
            mo.md(
                "``aufnahme_ts`` was dropped from the interim parquet — month-level chart unavailable."
            ),
            kind="warn",
        ),
    )
    monthly = (
        timestamps_df.unique(subset=["a_encounter_num"])
        .with_columns(pl.col("_admission").dt.strftime("%Y-%m").alias("year_month"))
        .group_by("year_month")
        .len()
        .sort("year_month")
    )
    monthly_chart = px.bar(
        monthly,
        x="year_month",
        y="len",
        title="Encounters per month",
        labels={"len": "encounters", "year_month": "year-month"},
    )
    mo.ui.plotly(monthly_chart)
    return


@app.cell
def _(df, mo, px):
    weekly = (
        df.unique(subset=["a_encounter_num"])
        .group_by(["year", "week"])
        .len()
        .sort(["year", "week"])
    )
    weekly_chart = px.scatter(
        weekly,
        x="week",
        y="len",
        color="year",
        title="Encounters per ISO week, by year",
        labels={"len": "encounters", "week": "ISO week"},
    )
    weekly_chart.update_layout(xaxis=dict(tickmode="linear", dtick=4))
    mo.ui.plotly(weekly_chart)
    return


if __name__ == "__main__":
    app.run()
