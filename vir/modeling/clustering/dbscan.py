from pathlib import Path

import polars as pl
import typer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from vir.helpers import suffix_from
from vir.modeling.clustering.commons import (
    calculate_min_samples,
    load_processed_data,
    save_results,
)

MODEL_NAME = "dbscan"
app = typer.Typer()


def prepare_features(df):
    """Prepare feature matrix for clustering"""
    print("Preparing feature matrix...")

    if "quantity" not in df.columns:
        raise KeyError("Column 'quantity' not found for sample weights")

    sample_weight = df["quantity"]  # rows' frequency
    min_samples = calculate_min_samples(sample_weight)
    feature_df = df.drop(["unique_id", "quantity"])  # exclude weight from features
    unique_ids = df.select("unique_id")

    print(f"✅ Feature matrix: {feature_df.shape}")
    print(f"   Unique IDs preserved: {len(unique_ids):,}")
    print(f"   Weights available: {len(sample_weight):,}")

    return feature_df, unique_ids, sample_weight, min_samples


def perform_pca_analysis(feature_df, n_components=10):
    print("Performing PCA analysis...")

    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(feature_df.to_numpy())
    prop_var = pca.explained_variance_ratio_
    eigenvalues = pca.explained_variance_

    print("✅ PCA completed:")
    print(f"   Input features: {feature_df.shape[1]}")
    print(f"   PCA components: {pca.n_components_}")
    print(f"   Total variance explained: {prop_var.sum():.1%}")
    print(f"   First 3 components: {prop_var[:3].sum():.1%}\n\n")

    print("Eigenvalues (variance explained by each component):**")

    eigenvalues_text = "\n".join(
        [
            f"- **PC{i + 1}**: eigenvalue={ev:.2f}, variance={pv:.1%}"
            for i, (ev, pv) in enumerate(zip(eigenvalues, prop_var))
        ]
    )
    print(f"```\n{eigenvalues_text}\n```")

    return pca, pca_features, prop_var, eigenvalues


def perform_clustering(feature_df, sample_weight, min_samples, n_components=3):
    print("Performing DBSCAN clustering with sample weights...")
    reduced_features = PCA(n_components=n_components).fit_transform(feature_df.to_numpy())
    clusterer = DBSCAN(eps=0.5, min_samples=min_samples, metric="euclidean")
    cluster_labels = clusterer.fit_predict(reduced_features, sample_weight=sample_weight)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    print("✅ Clustering completed:")
    print(f"   Clusters found: {n_clusters}")
    print(f"   Noise points: {n_noise:,}")
    print(f"   Clustered points: {len(cluster_labels) - n_noise:,}")
    print(f"   Clustering coverage: {(len(cluster_labels) - n_noise) / len(cluster_labels):.1%}")

    return cluster_labels, clusterer, reduced_features, n_clusters, n_noise


def create_cluster_results(unique_ids, cluster_labels, reduced_features):
    print("Creating cluster results...")

    cluster_results = pl.concat(
        [unique_ids, pl.DataFrame({"cluster": cluster_labels})], how="horizontal"
    )

    viz_df = pl.concat(
        [
            cluster_results,
            pl.DataFrame(
                {
                    "pc1": reduced_features[:, 0],
                    "pc2": reduced_features[:, 1],
                    "pc3": reduced_features[:, 2],
                }
            ),
        ],
        how="horizontal",
    )

    print(f"✅ Cluster results created: {len(cluster_results):,} records")
    return cluster_results, viz_df


@app.command()
def main(processed_dataset_filepath: Path, label: str = None):
    """Main clustering analysis pipeline"""
    print("=" * 60)
    print(f"Clustering with {MODEL_NAME}")
    print("=" * 60)
    if not label:
        label = f"_{label}_"

    try:
        # Step 1: Load data
        df = load_processed_data(processed_dataset_filepath)

        # Step 2: Prepare features
        feature_df, unique_ids, sample_weight, min_samples = prepare_features(df)

        # Step 3: PCA analysis
        perform_pca_analysis(feature_df)

        # Step 4: Clustering
        cluster_labels, clusterer, reduced_features, n_clusters, n_noise = perform_clustering(
            feature_df, sample_weight, min_samples
        )

        # Step 5: Create results
        cluster_results, vis_df = create_cluster_results(
            unique_ids, cluster_labels, reduced_features
        )

        # Step 6: Save results
        suffix = suffix_from(processed_dataset_filepath, remove_extension=True)
        cluster_path = save_results(cluster_results, label, suffix)

        print("\n" + "=" * 60)
        print("✅ CLUSTERING ANALYSIS COMPLETED!")
        print("=" * 60)
        print(f"📊 Total records: {len(cluster_results):,}")
        print(f"🎯 Clusters found: {n_clusters}")
        print(
            f"📈 Clustering coverage: {(len(cluster_results) - n_noise) / len(cluster_results):.1%}"
        )
        print(f"📁 Results saved: {cluster_path}")

        return True
    except Exception as e:
        print(f"\n❌ Error during clustering analysis: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    app()
