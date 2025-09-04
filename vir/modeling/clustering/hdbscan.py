from pathlib import Path

import hdbscan
import numpy as np
import polars as pl
import typer
from sklearn.decomposition import PCA

from vir.config import PROCESSED_DATA_DIR
from vir.helpers import suffix_from

MODEL_NAME = "hdbscan"
app = typer.Typer()


def load_processed_data(data_path):
    print("Loading processed data...")
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found: {data_path}")

    df = pl.read_parquet(data_path)

    print(f"✅ Data loaded: {len(df):,} records, {len(df.columns)} features")
    print(f"   Unique IDs: {df['unique_id'].n_unique():,}")

    return df


def prepare_features(df):
    """Prepare feature matrix for clustering"""
    print("Preparing feature matrix...")

    # Separate features from unique_id
    feature_df = df.drop("unique_id")
    unique_ids = df.select("unique_id")

    print(f"✅ Feature matrix: {feature_df.shape}")
    print(f"   Unique IDs preserved: {len(unique_ids):,}")

    return feature_df, unique_ids


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


def perform_clustering(feature_df, n_components=3):
    print("Performing HDBSCAN clustering...")
    n_samples = len(feature_df)
    reduced_features = PCA(n_components=n_components).fit_transform(feature_df.to_numpy())
    optimal_min_cluster_size = int(np.sqrt(n_samples))
    optimal_min_samples = max(1, optimal_min_cluster_size // 2)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=optimal_min_cluster_size,
        min_samples=optimal_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    cluster_labels = clusterer.fit_predict(reduced_features)

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


def save_results(cluster_results, output_dir, label=None):
    print("Saving results...")
    if not label:
        label = ""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    cluster_path = output_dir / f"cluster_results_{MODEL_NAME}{label}.parquet"
    cluster_results.write_parquet(cluster_path)

    print(f"✅ Results saved to: {output_dir}")
    return cluster_path


@app.command()
def main(processed_dataset_filepath: Path, label: str = None):
    """Main clustering analysis pipeline"""
    print("=" * 60)
    print(f"Clustering with {MODEL_NAME}")
    print("=" * 60)

    try:
        # Step 1: Load data
        df = load_processed_data(processed_dataset_filepath)

        # Step 2: Prepare features
        feature_df, unique_ids = prepare_features(df)

        # Step 3: PCA analysis
        perform_pca_analysis(feature_df)

        # Step 4: Clustering
        cluster_labels, clusterer, reduced_features, n_clusters, n_noise = perform_clustering(
            feature_df
        )

        # Step 5: Create results
        cluster_results, vis_df = create_cluster_results(
            unique_ids, cluster_labels, reduced_features
        )

        # Step 6: Save results
        filename = suffix_from(processed_dataset_filepath, remove_extension=True)
        output_dir = PROCESSED_DATA_DIR / f"clustering_results{filename}{label}.parquet"
        cluster_path = save_results(cluster_results, output_dir)

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
