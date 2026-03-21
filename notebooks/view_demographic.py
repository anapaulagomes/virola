import marimo

__generated_with = "0.18.4"
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
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    return AgglomerativeClustering, load_dotenv, np, pl, px, silhouette_score


@app.cell
def _():
    from archive.demographic_features import DemographicFeaturesView
    from vir.db import get_connection
    from vir.notebook_widgets import (
        create_file_dropdown,
        extract_dataset_name,
        load_and_prepare_dataframe,
        load_indicative_codes,
        load_terminology_labels,
    )
    from vir.views.demographic import DemographicView

    return (
        DemographicFeaturesView,
        DemographicView,
        create_file_dropdown,
        extract_dataset_name,
        get_connection,
        load_and_prepare_dataframe,
        load_indicative_codes,
        load_terminology_labels,
    )


@app.cell
def _(load_dotenv):
    _ = load_dotenv()
    return


@app.cell
def _(mo):
    args = mo.cli_args() if mo.app_meta().mode == "script" else {}
    return (args,)


@app.cell
def _(args):
    year_min = args.get("year_min", 2022)
    year_max = args.get("year_max", 2024)
    return year_max, year_min


@app.cell
def _(mo):
    mo.md(r"""
    ## Data Loading

    * Fetch diagnosis terminology: this is needed to ensure that they have the same order always
    * Fetch indicative codes
    """)
    return


@app.cell
def _(load_terminology_labels):
    terminology_descriptions = load_terminology_labels()
    return (terminology_descriptions,)


@app.cell
def _(args, load_indicative_codes):
    exclude_codes, indicative_codes_enabled = load_indicative_codes(
        enabled=not args.get("no_indicative", False)
    )
    return exclude_codes, indicative_codes_enabled


@app.cell
def _(mo):
    mo.md(r"""
    ### Prepare dataframe
    """)
    return


@app.cell
def _(args, create_file_dropdown):
    default_dataset = args.get("dataset", "salvador")
    files_dropdown = create_file_dropdown(default_pattern=default_dataset)
    return (files_dropdown,)


@app.cell
def _(files_dropdown):
    file_path = files_dropdown.value
    return (file_path,)


@app.cell
def _(
    extract_dataset_name,
    file_path,
    exclude_codes,
    load_and_prepare_dataframe,
    mo,
    terminology_descriptions,
    year_max,
    year_min,
):
    df = load_and_prepare_dataframe(
        file_path=file_path,
        terminology_labels=terminology_descriptions,
        exclude_codes=exclude_codes,
        year_min=year_min,
        year_max=year_max,
    )
    dataset_name = extract_dataset_name(file_path)
    if mo.app_meta().mode != "script":
        df.head()
    return dataset_name, df


@app.cell
def _(mo):
    mo.md(r"""
    ## Analysis Configuration
    """)
    return


@app.cell
def _(mo):
    k_slider = mo.ui.slider(start=2, stop=15, value=6, label="Number of Clusters (k)")
    k_slider
    return (k_slider,)


@app.cell
def _(mo):
    validation_mode_toggle = mo.ui.checkbox(label="Enable validation mode", value=False)
    validation_mode_toggle
    return (validation_mode_toggle,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Views

    Instantiate the two view objects that will be compared.
    """)
    return


@app.cell
def _(DemographicFeaturesView, DemographicView, dataset_name, df, validation_mode_toggle):
    # Determine pseudocount parameter
    if validation_mode_toggle.value:
        # Future: Could implement pseudocount grid search here
        print("ℹ️ Validation mode enabled (no grid search implemented yet)")
        best_pseudocount = 0.5
    else:
        best_pseudocount = 0.5

    # Chi-Square View (Original)
    chi_square_view = DemographicView(
        df=df, dataset_name=dataset_name, params={"pseudocount": best_pseudocount}
    )

    # Feature-Based View (Alternative)
    feature_view = DemographicFeaturesView(df=df, dataset_name=dataset_name, params={})
    return chi_square_view, feature_view


@app.cell
def _(mo):
    mo.md(r"""
    ### Helper functions

    Common functions for clustering, analysis, and visualization to ensure both views are processed identically.
    """)
    return


@app.cell
def _(AgglomerativeClustering, np, pl, silhouette_score):
    def run_clustering(distance_matrix, k, ids):
        """Runs clustering and returns labels and silhouette score."""
        if distance_matrix is None or distance_matrix.shape[0] < k:
            return None, np.nan, pl.DataFrame()

        cluster_labels = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average"
        ).fit_predict(distance_matrix)

        score = silhouette_score(distance_matrix, cluster_labels, metric="precomputed")
        cluster_df = pl.DataFrame({"terminology_id": ids, "cluster": cluster_labels})
        return cluster_labels, score, cluster_df

    return (run_clustering,)


@app.cell
def _(pl):
    def get_cluster_summary_stats(df, cluster_df):
        """Computes aggregated statistics (sex/age distributions) for each cluster."""
        if cluster_df.is_empty() or df.is_empty():
            return pl.DataFrame(), pl.DataFrame()

        df_with_clusters = df.join(cluster_df, on="terminology_id", how="inner")
        if df_with_clusters.is_empty():
            return pl.DataFrame(), pl.DataFrame()

        sex_dist = df_with_clusters.group_by("cluster", "sex").agg(
            pl.col("quantity").sum().alias("total")
        )
        sex_dist = sex_dist.with_columns(
            (pl.col("total") / pl.col("total").sum().over("cluster") * 100).alias("percentage")
        ).sort("cluster", "sex")

        age_dist = df_with_clusters.group_by("cluster", "age_group").agg(
            pl.col("quantity").sum().alias("total")
        )
        age_dist = age_dist.with_columns(
            (pl.col("total") / pl.col("total").sum().over("cluster") * 100).alias("percentage")
        ).sort("cluster", "age_group")

        return sex_dist, age_dist

    return (get_cluster_summary_stats,)


@app.cell
def _(pl):
    def get_feature_cluster_means(feature_df, cluster_df):
        if cluster_df.is_empty() or feature_df.is_empty():
            return pl.DataFrame()
        return (
            feature_df.join(cluster_df, on="terminology_id", how="inner")
            .group_by("cluster")
            .agg(
                pl.len().alias("n_diagnoses"),
                pl.col("child_ratio").mean().alias("avg_child_ratio"),
                pl.col("female_ratio").mean().alias("avg_female_ratio"),
                pl.col("adult_ratio").mean().alias("avg_adult_ratio"),
                pl.col("elderly_ratio").mean().alias("avg_elderly_ratio"),
                pl.col("young_adult_ratio").mean().alias("avg_young_adult_ratio"),
                pl.col("reproductive_female_ratio").mean().alias("avg_reproductive_female_ratio"),
            )
            .sort("cluster")
        )

    return (get_feature_cluster_means,)


@app.cell
def _(pl):
    def get_demographic_cluster_summary(df, cluster_df):
        """Calculates demographic profile summary for clusters (works with any view)."""
        if cluster_df.is_empty() or df.is_empty():
            return pl.DataFrame()

        df_with_clusters = df.join(cluster_df, on="terminology_id", how="inner")
        if df_with_clusters.is_empty():
            return pl.DataFrame()

        # Calculate proportions by cluster
        summary = (
            df_with_clusters.group_by("cluster")
            .agg(
                [
                    pl.len().alias("n_diagnoses"),
                    # Sex ratios
                    (
                        pl.when(pl.col("sex") == "Feminino")
                        .then(pl.col("quantity"))
                        .otherwise(0)
                        .sum()
                        / pl.col("quantity").sum()
                    ).alias("avg_female_ratio"),
                    # Age group ratios (using actual age groups from data)
                    (
                        pl.when(pl.col("age_group").is_in(["< 1 ano", "1 a 15 anos"]))
                        .then(pl.col("quantity"))
                        .otherwise(0)
                        .sum()
                        / pl.col("quantity").sum()
                    ).alias("avg_child_ratio"),
                    (
                        pl.when(pl.col("age_group") == "16 a 59 anos")
                        .then(pl.col("quantity"))
                        .otherwise(0)
                        .sum()
                        / pl.col("quantity").sum()
                    ).alias("avg_adult_ratio"),
                    (
                        pl.when(pl.col("age_group") == "Maior que 60 anos")
                        .then(pl.col("quantity"))
                        .otherwise(0)
                        .sum()
                        / pl.col("quantity").sum()
                    ).alias("avg_elderly_ratio"),
                ]
            )
            .sort("cluster")
        )

        return summary

    return (get_demographic_cluster_summary,)


@app.cell
def _(pl):
    def get_top_diagnoses(df, cluster_df, terminology_descriptions):
        """Gets the top 5 most frequent diagnoses for each cluster."""
        if cluster_df.is_empty() or df.is_empty():
            return pl.DataFrame()

        # Join with descriptions if not already present
        result_df = df.join(cluster_df, on="terminology_id")
        if "description" not in result_df.columns:
            result_df = result_df.join(
                terminology_descriptions, left_on="terminology_id", right_on="id", how="left"
            )

        # Create combined label with description
        result_df = result_df.with_columns(
            (
                pl.col("description") + pl.lit(" (") + pl.col("terminology_label") + pl.lit(")")
            ).alias("diagnosis")
        )

        return (
            result_df.group_by(["cluster", "diagnosis"])
            .agg(pl.col("quantity").sum().alias("count"))
            .sort(["cluster", "count"], descending=[False, True])
            .group_by("cluster", maintain_order=True)
            .head(5)
        )

    return (get_top_diagnoses,)


@app.cell
def _(px):
    def plot_distance_matrix(matrix, title):
        """Generates a heatmap of the distance matrix."""
        return px.imshow(
            matrix,
            title=title,
            color_continuous_scale="viridis",
            labels={"x": "Diagnosis", "y": "Diagnosis", "color": "Distance"},
            aspect="equal",
        ).update_layout(height=500, width=600)

    return (plot_distance_matrix,)


@app.cell
def _(mo, px):
    def plot_cluster_sex_distribution(sex_dist_df, title):
        """Generates a bar chart for sex distribution by cluster."""
        if sex_dist_df.is_empty():
            return mo.md("### Sex Distribution\n*No data to plot.*")
        fig = px.bar(
            sex_dist_df.to_pandas(),
            x="cluster",
            y="percentage",
            color="sex",
            title=title,
            barmode="group",
            labels={"percentage": "% of Diagnoses"},
        )
        return fig

    return (plot_cluster_sex_distribution,)


@app.cell
def _(mo, px):
    def plot_cluster_age_distribution(age_dist_df, title):
        """Generates a stacked bar chart for age distribution by cluster."""
        if age_dist_df.is_empty():
            return mo.md("### Age Distribution\n*No data to plot.*")
        fig = px.bar(
            age_dist_df.to_pandas(),
            x="cluster",
            y="percentage",
            color="age_group",
            title=title,
            barmode="stack",
            labels={"percentage": "% of Diagnoses"},
        )
        return fig

    return (plot_cluster_age_distribution,)


@app.cell
def _(mo, pl, px):
    def plot_feature_means_dot_plot(feature_means_df):
        """Generates a dot plot of mean feature values for each cluster."""
        if feature_means_df.is_empty():
            return mo.md("*No feature data to plot.*")

        features_to_plot = [
            "avg_female_ratio",
            "avg_child_ratio",
            "avg_adult_ratio",
            "avg_elderly_ratio",
            "avg_young_adult_ratio",
            "avg_reproductive_female_ratio",
        ]

        plot_df = feature_means_df.unpivot(
            index=["cluster"],
            on=features_to_plot,
            variable_name="feature",
            value_name="average_value",
        )

        # Clean up feature names
        plot_df = plot_df.with_columns(
            pl.col("feature")
            .str.replace("avg_", "")
            .str.replace("_", " ")
            .str.to_titlecase()
            .alias("feature")
        )

        plot_df_pd = plot_df.to_pandas()

        # Create scatter plot with better sizing
        fig = px.scatter(
            plot_df_pd,
            x="average_value",
            y="feature",  # Changed to show features on y-axis
            color="cluster",  # Changed to color by cluster
            size="average_value",
            title="Cluster Characteristics by Mean Feature Value",
            labels={
                "average_value": "Average Ratio",
                "feature": "Demographic Feature",
                "cluster": "Cluster",
            },
            height=400,
            category_orders={"feature": sorted(plot_df_pd["feature"].unique())},
        )

        # Update layout for better readability
        fig.update_layout(
            xaxis_title="Average Feature Value",
            yaxis_title="Demographic Feature",
            showlegend=True,
            legend_title="Cluster",
        )

        # Better marker sizing
        if not plot_df_pd.empty and plot_df_pd["average_value"].max() > 0:
            fig.update_traces(marker=dict(sizemin=8, sizemode="diameter", sizeref=0.1))

        return fig

    return (plot_feature_means_dot_plot,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Comparative Analysis
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Feature-Based View

    This view extracts interpretable demographic features (age/sex ratios) and uses Euclidean distance.
    """)
    return


@app.cell
def _(
    df,
    feature_view,
    get_cluster_summary_stats,
    get_feature_cluster_means,
    get_top_diagnoses,
    k_slider,
    run_clustering,
    terminology_descriptions,
):
    feature_matrix = feature_view.get_matrix(force_recompute=True)
    feature_ids = feature_view.get_terminology_ids().to_list()

    feature_labels, feature_score, feature_cluster_df = run_clustering(
        feature_matrix, k_slider.value, feature_ids
    )

    feature_sex_dist, feature_age_dist = get_cluster_summary_stats(df, feature_cluster_df)
    feature_top_diagnoses = get_top_diagnoses(df, feature_cluster_df, terminology_descriptions)

    feature_df = feature_view.get_features()
    feature_means = get_feature_cluster_means(feature_df, feature_cluster_df)
    return (
        feature_age_dist,
        feature_df,
        feature_matrix,
        feature_means,
        feature_score,
        feature_sex_dist,
        feature_top_diagnoses,
    )


@app.cell
def _(feature_df):
    feature_df
    return


@app.cell
def _(feature_matrix, feature_score, mo, plot_distance_matrix):
    mo.hstack(
        [
            plot_distance_matrix(feature_matrix, "Distance Matrix (Feature-Based)"),
            mo.md(f"**Silhouette Score:** `{feature_score:.4f}`"),
        ],
        justify="center",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Top 5 Diagnoses per Cluster
    """)
    return


@app.cell
def _(feature_top_diagnoses, mo):
    mo.ui.table(feature_top_diagnoses, selection=None)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Cluster Demographic Profiles
    """)
    return


@app.cell
def _(
    feature_age_dist,
    feature_sex_dist,
    mo,
    plot_cluster_age_distribution,
    plot_cluster_sex_distribution,
):
    mo.hstack(
        [
            plot_cluster_sex_distribution(feature_sex_dist, "Sex Distribution (Feature-Based)"),
            plot_cluster_age_distribution(feature_age_dist, "Age Distribution (Feature-Based)"),
        ],
        justify="center",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Cluster Mean Feature Analysis
    """)
    return


@app.cell
def _(feature_means, plot_feature_means_dot_plot):
    plot_feature_means_dot_plot(feature_means)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 5.2 Chi-Square View
    This view compares complete demographic distributions using Chi-Square distance.
    """)
    return


@app.cell
def _(
    chi_square_view,
    df,
    get_cluster_summary_stats,
    get_demographic_cluster_summary,
    get_top_diagnoses,
    k_slider,
    run_clustering,
    terminology_descriptions,
):
    # Get matrix and IDs for chi-square view
    chi_matrix = chi_square_view.get_matrix()
    try:
        chi_ids = chi_square_view.get_terminology_ids().to_list()
    except AttributeError:
        chi_ids = df.select("terminology_id").unique().to_series().to_list()
        chi_ids = sorted(chi_ids)

    # Run clustering
    chi_labels, chi_score, chi_cluster_df = run_clustering(chi_matrix, k_slider.value, chi_ids)

    # Get summary stats
    chi_sex_dist, chi_age_dist = get_cluster_summary_stats(df, chi_cluster_df)
    chi_top_diagnoses = get_top_diagnoses(df, chi_cluster_df, terminology_descriptions)

    # Get demographic summary for comparison
    chi_demographic_summary = get_demographic_cluster_summary(df, chi_cluster_df)
    return (
        chi_age_dist,
        chi_demographic_summary,
        chi_matrix,
        chi_score,
        chi_sex_dist,
        chi_top_diagnoses,
    )


@app.cell
def _(chi_matrix, chi_score, mo, plot_distance_matrix):
    mo.hstack(
        [
            plot_distance_matrix(chi_matrix, "Distance Matrix (Chi-Square)"),
            mo.md(f"**Silhouette Score:** `{chi_score:.4f}`"),
        ],
        justify="center",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    #### Top 5 Diagnoses per Cluster
    """)
    return


@app.cell
def _(chi_top_diagnoses, mo):
    mo.ui.table(chi_top_diagnoses, selection=None)
    return


@app.cell
def _(mo):
    mo.md("""
    #### Cluster Demographic Profiles
    """)
    return


@app.cell
def _(
    chi_age_dist,
    chi_sex_dist,
    mo,
    plot_cluster_age_distribution,
    plot_cluster_sex_distribution,
):
    mo.hstack(
        [
            plot_cluster_sex_distribution(chi_sex_dist, "Sex Distribution (Chi-Square)"),
            plot_cluster_age_distribution(chi_age_dist, "Age Distribution (Chi-Square)"),
        ],
        justify="center",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    #### Cluster Demographic Profile Summary
    """)
    return


@app.cell
def _(chi_demographic_summary, mo, pl, px):
    if chi_demographic_summary.is_empty():
        chi_demo_plot = mo.md("*No demographic summary available.*")
    else:
        _features = ["avg_female_ratio", "avg_child_ratio", "avg_adult_ratio", "avg_elderly_ratio"]
        _plot_df = chi_demographic_summary.unpivot(
            index=["cluster"],
            on=_features,
            variable_name="feature",
            value_name="average_value",
        )
        _plot_df = _plot_df.with_columns(
            pl.col("feature")
            .str.replace("avg_", "")
            .str.replace("_", " ")
            .str.to_titlecase()
            .alias("feature")
        )
        _plot_df_pd = _plot_df.to_pandas()

        chi_demo_plot = px.scatter(
            _plot_df_pd,
            x="average_value",
            y="feature",
            color="cluster",
            size="average_value",
            title="Cluster Demographic Profile (Chi-Square)",
            labels={
                "average_value": "Average Ratio",
                "feature": "Demographic Feature",
                "cluster": "Cluster",
            },
            height=300,
            category_orders={"feature": sorted(_plot_df_pd["feature"].unique())},
        )
        chi_demo_plot.update_layout(
            xaxis_title="Average Ratio",
            yaxis_title="Demographic Feature",
            showlegend=True,
            legend_title="Cluster",
        )
        chi_demo_plot.update_traces(marker=dict(sizemin=8, sizemode="diameter", sizeref=0.1))

    chi_demo_plot
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Interactive Disease Browser
    Search for a specific diagnosis to see its demographic features and find similar diagnoses based on the **Feature-Based View**.
    """)
    return


@app.cell
def _(df):
    # Get available diagnosis types
    available_types = sorted(df["code_type"].unique().to_list())
    return (available_types,)


@app.cell
def _(available_types, mo):
    # Create search widgets
    disease_search = mo.ui.text(
        placeholder="Search diagnosis description (e.g., 'pneumonia', 'diabetes')...",
        label="Disease Search",
        full_width=True,
    )

    diagnosis_type_dropdown = mo.ui.dropdown(
        options=available_types,
        label="Diagnosis Type",
        value=available_types[0] if available_types else None,
    )

    mo.hstack([disease_search, diagnosis_type_dropdown], justify="start")
    return diagnosis_type_dropdown, disease_search


@app.cell
def _(df, diagnosis_type_dropdown, disease_search, get_connection, pl):
    # Search for matching diagnoses
    _search_term = disease_search.value.lower() if disease_search.value else ""
    _selected_type = diagnosis_type_dropdown.value

    if _search_term and len(_search_term) >= 3:
        _con = get_connection()
        _descriptions = _con.execute(
            """
            SELECT id, code, code_type, description
            FROM medical_terminologies
            WHERE code_type = ? AND LOWER(description) LIKE ?
            ORDER BY description
            LIMIT 20
            """,
            [_selected_type, f"%{_search_term}%"],
        ).pl()

        matching_diagnoses = _descriptions.filter(
            pl.col("id").is_in(df["terminology_id"].unique())
        )
    else:
        matching_diagnoses = pl.DataFrame(
            {"id": [], "code": [], "code_type": [], "description": []}
        )
    return (matching_diagnoses,)


@app.cell
def _(matching_diagnoses, mo):
    if matching_diagnoses.height > 0:
        # Create dropdown with matching diagnoses
        _diagnosis_options = {
            f"{row['code']} - {row['description'][:80]}": row["id"]
            for row in matching_diagnoses.iter_rows(named=True)
        }

        selected_diagnosis_dropdown = mo.ui.dropdown(
            options=_diagnosis_options,
            label="Select Diagnosis",
            full_width=True,
        )
    else:
        selected_diagnosis_dropdown = mo.md(
            "*No matches found. Try a different search term (min 3 characters).*"
        )
    return (selected_diagnosis_dropdown,)


@app.cell
def _(feature_df, matching_diagnoses, mo, pl, selected_diagnosis_dropdown):
    if (
        matching_diagnoses.height > 0
        and selected_diagnosis_dropdown
        and selected_diagnosis_dropdown.value
    ):
        _selected_id = selected_diagnosis_dropdown.value

        # Get selected diagnosis features
        _selected_features = feature_df.filter(pl.col("terminology_id") == _selected_id)

        if _selected_features.height > 0:
            mo.md(f"""
            **Selected Diagnosis Features:**
            - Female Ratio: {_selected_features["female_ratio"][0]:.1%}
            - Child Ratio: {_selected_features["child_ratio"][0]:.1%}
            - Adult Ratio: {_selected_features["adult_ratio"][0]:.1%}
            - Elderly Ratio: {_selected_features["elderly_ratio"][0]:.1%}
            - Young Adult Ratio: {_selected_features["young_adult_ratio"][0]:.1%}
            - Reproductive Female Ratio: {_selected_features["reproductive_female_ratio"][0]:.1%}
            """)
        else:
            mo.md("*Selected diagnosis not in filtered dataset.*")
    return


@app.cell
def _(
    feature_df,
    feature_view,
    get_connection,
    matching_diagnoses,
    mo,
    pl,
    selected_diagnosis_dropdown,
):
    if (
        matching_diagnoses.height > 0
        and selected_diagnosis_dropdown
        and selected_diagnosis_dropdown.value
    ):
        _selected_id = selected_diagnosis_dropdown.value

        distance_matrix = feature_view.get_matrix()
        ids = feature_view.get_terminology_ids().to_list()

        if _selected_id in ids:
            _selected_idx = ids.index(_selected_id)
            _distances = distance_matrix[_selected_idx]

            similar_diagnoses = (
                pl.DataFrame({"terminology_id": ids, "distance": _distances})
                .filter(pl.col("terminology_id") != _selected_id)
                .sort("distance")
                .head(10)
            )

            similar_with_features = similar_diagnoses.join(feature_df, on="terminology_id")

            # Get descriptions
            _ids_list = similar_with_features["terminology_id"].to_list()
            _placeholders = ",".join(["?"] * len(_ids_list))
            _con = get_connection()
            _descriptions = _con.execute(
                f"SELECT id, code, description FROM medical_terminologies WHERE id IN ({_placeholders})",
                _ids_list,
            ).pl()

            similar_with_desc = similar_with_features.join(
                _descriptions, left_on="terminology_id", right_on="id"
            )

            mo.vstack(
                [
                    mo.md("**Top 10 Similar Diagnoses (by demographic features):**"),
                    mo.ui.table(
                        similar_with_desc.select(
                            "code",
                            "description",
                            "distance",
                            "female_ratio",
                            "child_ratio",
                            "adult_ratio",
                            "elderly_ratio",
                        )
                    ),
                ]
            )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. LLM Cluster Descriptions (Optional)
    This section generates descriptive names for each cluster using an LLM. It is commented out by default as it can be slow to run.
    """)
    return


@app.cell
def _(mo):
    # This cell is commented out to prevent long run times.
    # To run, uncomment the lines and ensure you have an Anthropic API key set up.

    # from vir.modeling.clustering.describe import describe_cluster

    # Get clusters from one of the views (e.g., feature_view)
    # matrix = feature_view.get_matrix()
    # ids = feature_view.get_terminology_ids().to_list()
    # _, _, cluster_df = run_clustering(matrix, k_slider.value, ids)

    # if not cluster_df.is_empty():
    #     top_diagnoses = get_top_diagnoses(df, cluster_df, terminology_descriptions)
    #     cluster_descriptions = {}
    #     for _cluster_id in sorted(top_diagnoses["cluster"].unique().to_list()):
    #         _cluster_terms = top_diagnoses.filter(pl.col("cluster") == _cluster_id)
    #         _term_dict = {
    #             row["terminology_label"]: row["count"]
    #             for row in _cluster_terms.iter_rows(named=True)
    #         }
    #
    #         # _description = describe_cluster(_term_dict)
    #         # cluster_descriptions[_cluster_id] = _description
    #
    #     mo.ui.table(pl.from_dicts([
    #        {"cluster": k, "description": v.get("short_description", "")}
    #        for k, v in cluster_descriptions.items()
    #     ]))

    mo.md("""
    *Uncomment the code above to generate LLM cluster descriptions.*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Cache
    """)
    return


@app.cell
def _(chi_square_view, feature_view):
    chi_square_view.get_cache_info()
    feature_view.get_cache_info()
    return


@app.cell
def _(mo):
    if mo.app_meta().mode == "script":
        print("✓ Demographic view analysis completed")
    return


if __name__ == "__main__":
    app.run()
