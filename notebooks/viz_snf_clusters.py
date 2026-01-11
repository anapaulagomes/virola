import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import plotly.express as px
    import polars as pl
    from dotenv import load_dotenv

    return Path, json, load_dotenv, mo, pl, px


@app.cell
def _():
    from vir.config import CLUSTERING_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
    from vir.db import (
        fetch_terminologies,
    )
    from vir.features import AGE_GROUP
    from vir.modeling.clustering.describe import describe_cluster
    from vir.viz import plot_categories

    return (
        AGE_GROUP,
        CLUSTERING_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        describe_cluster,
        fetch_terminologies,
        plot_categories,
    )


@app.cell
def _(load_dotenv):
    _ = load_dotenv()
    return


@app.cell
def _(INTERIM_DATA_DIR, mo):
    _all_options = [
        str(_file.resolve()) for _file in INTERIM_DATA_DIR.glob("**/cleaned_*.parquet")
    ]
    default = [
        _option
        for _option in _all_options
        if "cleaned_aesop_raw_dados_2017_20250223_20municipios_salvador.parquet" in _option
    ]

    files_dropdown = mo.ui.dropdown(
        options=_all_options,
        label="Choose dataset",
        searchable=True,
        value=default[0] if default else None,
    )
    files_dropdown
    return (files_dropdown,)


@app.cell
def _(fetch_terminologies):
    terminology_labels = fetch_terminologies()
    return (terminology_labels,)


@app.cell
def _(PROCESSED_DATA_DIR, pl):
    codes_df = (
        pl.read_csv(f"{PROCESSED_DATA_DIR}/cid10_ciap2_ab.csv")
        .with_columns(terminology_label=pl.col("code") + pl.lit(":") + pl.col("code_type"))
        .rename({"description": "terminology_description"})
    )
    codes_df = codes_df.drop(["code", "code_type"])
    return (codes_df,)


@app.cell
def _(codes_df, files_dropdown, pl, terminology_labels):
    selected_file_name = files_dropdown.value
    df = pl.read_parquet(selected_file_name)
    df = df.drop(["unique_id", "state", "city", "region", "population_range"])  # make it per city
    df = df.group_by(["year", "week", "sex", "age_group", "code_type", "code"]).agg(
        pl.col("quantity").sum()
    )
    df = df.join(terminology_labels, on=["code", "code_type"]).rename(
        {"id": "terminology_id", "label": "terminology_label"}
    )
    df = df.join(codes_df, on=["terminology_label"])
    df = df.filter(pl.col("year").ge(2022))
    return (df,)


@app.cell
def _(CLUSTERING_DIR, mo):
    _all_options = [str(_file.resolve()) for _file in CLUSTERING_DIR.glob("**/*.json")]
    clusters_results_dropdown = mo.ui.dropdown(
        options=_all_options,
        label="Choose cluster results",
        searchable=True,
    )
    clusters_results_dropdown
    return (clusters_results_dropdown,)


@app.cell
def _(Path, clusters_results_dropdown, json):
    found_clusters = json.loads(
        Path(clusters_results_dropdown.value).read_text()
    )  # terminology id - cluster label
    return (found_clusters,)


@app.cell
def _(found_clusters):
    terminology_id_list = [int(_fc["terminology_id"]) for _fc in found_clusters]
    found_clusters_list = [int(_fc["cluster"]) for _fc in found_clusters]
    return found_clusters_list, terminology_id_list


@app.cell
def _(found_clusters_list, pl, terminology_id_list):
    clusters_df = pl.DataFrame(
        {"terminology_id": terminology_id_list, "cluster": found_clusters_list}
    )
    return (clusters_df,)


@app.cell
def _(clusters_df, df):
    df_with_clusters = df.join(clusters_df, on=["terminology_id"], how="left")
    return (df_with_clusters,)


@app.cell
def _(found_clusters_list):
    cluster_labels = set(found_clusters_list)
    number_of_clusters = len(cluster_labels)
    return cluster_labels, number_of_clusters


@app.cell
def _(df_with_clusters, pl):
    clusters_distribution = (
        df_with_clusters.group_by("cluster")
        .agg(pl.col("terminology_id").n_unique().alias("unique_ids"))
        .sort("cluster")  # Opcional: para ordenar pelo nome do cluster
    )
    clusters_median_size = clusters_distribution["unique_ids"].median()
    return clusters_distribution, clusters_median_size


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data-Driven Syndrome Detection
    """)
    return


@app.cell
def _(clusters_distribution, px):
    _df = clusters_distribution.group_by("unique_ids").len()
    unique_codes_histogram = px.bar(
        _df,
        x="unique_ids",
        y="len",
        labels={"unique_ids": "Number of unique codes", "len": "Number of clusters"},
        log_x=True,
    )
    return (unique_codes_histogram,)


@app.cell
def _(
    clusters_median_size,
    df_with_clusters,
    found_clusters,
    mo,
    number_of_clusters,
    unique_codes_histogram,
):
    mo.stop(df_with_clusters is None)

    _cards = [
        mo.stat(
            label="Nr. of clusters",
            value=number_of_clusters,
            bordered=True,
        ),
        mo.stat(
            label="Median",
            value=f"{clusters_median_size}",
            caption="Median nr. of codes that belong to each cluster",
        ),
        mo.stat(
            label="Nr. of rows",
            value=df_with_clusters.shape[0],
        ),
        mo.stat(
            label="Nr. of codes (ICD-10, ICPC-2, ABP together)",
            value=len(found_clusters),  # one code per cluster
        ),
        mo.stat(
            label="Period",
            value=f"From {df_with_clusters['year'].min()} to {df_with_clusters['year'].max()}",
        ),
    ]

    _title = "## General info"

    mo.vstack(
        [mo.md(_title), mo.hstack(_cards, widths="equal", align="center"), unique_codes_histogram]
    )
    return


@app.cell
def _(mo):
    terminology_search = mo.ui.text(
        placeholder="Search terminology descriptions...",
        label="Search terminologies",
        full_width=True,
    )
    terminology_search
    return (terminology_search,)


@app.cell
def _(cluster_labels, df_with_clusters, pl, terminology_search):
    # Filter clusters based on search term
    search_term = terminology_search.value.strip().lower() if terminology_search.value else ""

    if search_term:
        # Find clusters that contain terminologies with matching descriptions
        matching_clusters = (
            df_with_clusters.filter(
                pl.col("cluster").is_not_null(),
                df_with_clusters["terminology_description"]
                .str.to_lowercase()
                .str.contains(search_term),
            )["cluster"]
            .unique()
            .to_list()
        )

        filtered_cluster_labels = set(matching_clusters)
    else:
        # If search is empty, show all clusters
        filtered_cluster_labels = cluster_labels
    return (filtered_cluster_labels,)


@app.cell
def _(filtered_cluster_labels, mo):
    cluster_dropdown = mo.ui.dropdown(
        sorted(filtered_cluster_labels) if filtered_cluster_labels else [],
        value=sorted(filtered_cluster_labels)[0] if filtered_cluster_labels else None,
        label="Choose a cluster",
    )
    return (cluster_dropdown,)


@app.cell
def _(cluster_dropdown, df_with_clusters, pl):
    current_cluster_df = df_with_clusters.filter(pl.col("cluster").eq(cluster_dropdown.value))
    return (current_cluster_df,)


@app.cell
def _(current_cluster_df, pl):
    terminology_labels_in_cluster = current_cluster_df.group_by(
        "terminology_label", "terminology_description"
    ).agg(pl.col("quantity").sum())
    terminology_labels_count = {}

    for term in terminology_labels_in_cluster.iter_rows(named=True):
        terminology_labels_count[
            f"{term['terminology_description']} ({term['terminology_label']})"
        ] = term["quantity"]
    return (terminology_labels_count,)


@app.cell
def _(describe_cluster, terminology_labels_count):
    current_cluster_description = describe_cluster(terminology_labels_count)
    return (current_cluster_description,)


@app.cell
def _(
    cluster_dropdown,
    cluster_labels,
    current_cluster_description,
    current_cluster_df,
    mo,
    terminology_labels_count,
):
    mo.stop(cluster_labels is None)

    _cards = [
        cluster_dropdown,
        mo.stat(
            label="Nr. of rows",
            value=current_cluster_df.shape[0],
        ),
        mo.stat(
            label="Nr. of unique codes",
            value=len(terminology_labels_count),
        ),
        mo.stat(
            label="Description (powered by LLM)",
            value=current_cluster_description["short_description"],
            caption=current_cluster_description["description"],
        ),
    ]

    _title = "## Evaluation per cluster"

    mo.vstack(
        [
            mo.md(_title),
            mo.hstack(_cards, widths="equal", align="center"),
        ]
    )
    return


@app.cell
def _(terminology_labels_count):
    terminology_labels_count
    return


@app.cell
def _(current_cluster_df):
    current_cluster_df
    return


@app.cell
def _(current_cluster_df, px):
    _fig = px.line(
        current_cluster_df.group_by(["year", "week"]).len("count").sort(["year", "week"]),
        x="week",
        y="count",
        color="year",
    )
    _fig.update_xaxes(dtick="M1", tickangle=45, showgrid=True)
    _fig
    return


@app.cell
def _(current_cluster_df, pl, px):
    _fig = px.treemap(
        current_cluster_df.group_by(["code_type", "code", "terminology_description"]).agg(
            pl.len().alias("count")
        ),
        path=["code_type", "code"],
        hover_data=["terminology_description"],
        values="count",
        title="Cluster codes",
    )
    _fig
    return


@app.cell
def _(AGE_GROUP, current_cluster_df, plot_categories):
    plot_categories(current_cluster_df, "age_group", AGE_GROUP, title="Age group distribution")
    return


@app.cell
def _(current_cluster_df, plot_categories):
    plot_categories(current_cluster_df, "sex", title="Sex distribution")
    return


if __name__ == "__main__":
    app.run()
