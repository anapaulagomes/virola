import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Similarity Network Fusion

    ## Dimensions 🎲

    * **Clinical**: semantic similarity
    * **Temporal**: temporal frequency
    * **Demographic**: sex and gender characteristics

    Calculate the similarity in all these dimensions, having the diagnosis as the central factor.
    """)
    return


@app.cell
def _():
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from dotenv import load_dotenv
    from sklearn import cluster
    from snf import compute, metrics

    return cluster, compute, load_dotenv, metrics, mo, np, plt, time


@app.cell
def _():
    from vir.notebook_widgets import (
        create_file_dropdown,
        extract_dataset_name,
        load_and_prepare_dataframe,
        load_indicative_codes,
        load_terminology_labels,
    )
    from vir.views import ClinicalView, DemographicView, TemporalView

    return (
        ClinicalView,
        DemographicView,
        TemporalView,
        create_file_dropdown,
        extract_dataset_name,
        load_and_prepare_dataframe,
        load_indicative_codes,
        load_terminology_labels,
    )


@app.cell
def _(load_dotenv):
    _ = load_dotenv()
    return


@app.cell
def _():
    run_k_validation = True
    return (run_k_validation,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Data loading
    """)
    return


@app.cell
def _(load_terminology_labels):
    terminology_labels = load_terminology_labels()
    return (terminology_labels,)


@app.cell
def _(load_indicative_codes):
    include_codes, indicative_codes_enabled = load_indicative_codes()
    return (include_codes,)


@app.cell
def _(create_file_dropdown):
    files_dropdown = create_file_dropdown()
    files_dropdown
    return (files_dropdown,)


@app.cell
def _(
    files_dropdown,
    include_codes,
    load_and_prepare_dataframe,
    terminology_labels,
):
    df = load_and_prepare_dataframe(
        file_path=files_dropdown.value,
        terminology_labels=terminology_labels,
        include_codes=include_codes,
        year_min=2022,
        year_max=2024,
    )
    df.head()
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Dimensions

    Each view computes a distance matrix, then converts to affinity using adaptive Gaussian kernel.
    """)
    return


@app.cell
def _(df, extract_dataset_name, files_dropdown):
    view_params = {"df": df, "dataset_name": extract_dataset_name(files_dropdown.value)}
    return (view_params,)


@app.cell
def _(ClinicalView, DemographicView, TemporalView, view_params):
    # Instantiate views
    clinical_view = ClinicalView(**view_params)
    temporal_view = TemporalView(**view_params)
    demographic_view = DemographicView(**view_params)

    # Get distance matrices (cached)
    clinical_distance = clinical_view.get_matrix()
    temporal_distance = temporal_view.get_matrix()
    demographic_distance = demographic_view.get_matrix()

    print(f"Clinical distance shape: {clinical_distance.shape}")
    print(f"Temporal distance shape: {temporal_distance.shape}")
    print(f"Demographic distance shape: {demographic_distance.shape}")
    return clinical_view, demographic_view, temporal_view


@app.cell
def _(mo):
    mo.md(r"""
    ## Convert distances to affinities

    Using adaptive Gaussian kernel with k-nearest neighbor based local scaling.
    """)
    return


@app.cell
def _(clinical_view, demographic_view, temporal_view):
    # Convert to affinity matrices using BaseView.to_affinity()
    # k=20 is the standard value for SNF
    clinical_affinity = clinical_view.to_affinity(k=20, mu=0.5)
    temporal_affinity = temporal_view.to_affinity(k=20, mu=0.5)
    demographic_affinity = demographic_view.to_affinity(k=20, mu=0.5)

    print(
        f"Clinical affinity range: [{clinical_affinity.min():.4f}, {clinical_affinity.max():.4f}]"
    )
    print(
        f"Temporal affinity range: [{temporal_affinity.min():.4f}, {temporal_affinity.max():.4f}]"
    )
    print(
        f"Demographic affinity range: [{demographic_affinity.min():.4f}, {demographic_affinity.max():.4f}]"
    )
    return clinical_affinity, demographic_affinity, temporal_affinity


@app.cell
def _(clinical_affinity, demographic_affinity, temporal_affinity):
    assert clinical_affinity.shape == temporal_affinity.shape == demographic_affinity.shape, (
        clinical_affinity.shape,
        temporal_affinity.shape,
        demographic_affinity.shape,
    )

    affinities = [clinical_affinity, temporal_affinity, demographic_affinity]
    return (affinities,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Validation
    """)
    return


@app.cell
def _(cluster, compute, metrics, np, time):
    def evaluate_snf_k_range(affinities, K_values, t=20, n_range=(2, 80)):
        results = []

        for K in K_values:
            start_time = time.time()
            print(f"\n Running SNF with K={K}...")
            fused = compute.snf(affinities, K=K, t=t)
            fused = np.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=0.0)

            n_opt, n_alt = compute.get_n_clusters(fused, range(*n_range))
            print(f"  Estimated number of clusters: {n_opt} (second best: {n_alt})")

            fused_labels = cluster.spectral_clustering(fused, n_clusters=n_opt)

            silhouette = metrics.silhouette_score(fused, fused_labels)
            nmi = metrics.nmi([fused_labels])

            end_time = time.time()
            elapsed = end_time - start_time

            print(f"  Done in {elapsed / 60:.2f} min | Silhouette = {silhouette:.4f}")

            results.append(
                {
                    "K": K,
                    "clusters": n_opt,
                    "silhouette": float(silhouette),
                    "nmi": float(nmi),
                }
            )
        return results

    return (evaluate_snf_k_range,)


@app.cell
def _(affinities, evaluate_snf_k_range, run_k_validation):
    if run_k_validation:
        K_values = [20, 50, 75, 100, 150]
        results = evaluate_snf_k_range(affinities, K_values, t=20, n_range=(2, 100))
    else:
        results = []
    return (results,)


@app.cell
def _(plt, results, run_k_validation):
    if run_k_validation:
        number_of_neighbors = []
        silhouettes = []
        clusters = []

        for result in results:
            number_of_neighbors.append(result["K"])
            silhouettes.append(result["silhouette"])
            clusters.append(result["clusters"])

        _fig, _ax1 = plt.subplots(figsize=(8, 4))
        _ax1.plot(number_of_neighbors, silhouettes, "-o", color="tab:blue")
        _ax1.set_xlabel("K (number of neighbors)")
        _ax1.set_ylabel("Silhouette", color="tab:blue")
        _ax1.tick_params(axis="y", labelcolor="tab:blue")

        _ax2 = _ax1.twinx()
        _ax2.plot(number_of_neighbors, clusters, "--s", color="tab:orange")
        _ax2.set_ylabel("# Clusters (eigengap)", color="tab:orange")
        _ax2.tick_params(axis="y", labelcolor="tab:orange")

        plt.title("K evaluation in SNF")
        plt.tight_layout()
        plt.show()
    return


@app.cell
def _():
    # hyperparameters chosen based on K validation step
    K = 10
    number_of_clusters = 60
    return K, number_of_clusters


@app.cell
def _(K, affinities, compute, np):
    fused = compute.snf(affinities, K=K)  # alpha = 1 - 5 min
    print(np.isnan(fused).any())  # nan might occurr due to divisions to 0
    fused = np.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=0.0)
    # np.savetxt(f'fused_snf_{selected_filename}.txt', fused)
    return (fused,)


@app.cell
def _(fused):
    fused
    return


@app.cell
def _(fused, plt):
    plt.matshow(fused, cmap="viridis")
    plt.colorbar()
    plt.title("Fused matrices")
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Clustering 🩻
    """)
    return


@app.cell
def _(cluster, fused, number_of_clusters):
    fused_labels = cluster.spectral_clustering(fused, n_clusters=number_of_clusters)
    return (fused_labels,)


@app.cell
def _(fused, fused_labels, metrics):
    silhouette = metrics.silhouette_score(fused, fused_labels)  # nan se tiver apenas uma amostra!
    silhouette
    return


@app.cell
def _(cluster, fused, metrics):
    experiments_cluster_silhouette = {}

    for _nr_cluster in [20, 40, 60, 100, 150]:
        _fused_labels = cluster.spectral_clustering(fused, n_clusters=_nr_cluster)
        _silhouette = metrics.silhouette_score(fused, _fused_labels)
        experiments_cluster_silhouette[_nr_cluster] = _silhouette

    experiments_cluster_silhouette
    return


if __name__ == "__main__":
    app.run()
