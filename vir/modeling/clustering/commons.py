import polars as pl

from vir.config import CLUSTERING_DIR


def calculate_min_samples(quantity):
    total_weight = quantity.sum()
    alpha = 0.01  # 1% of total weight
    return max(2, int(alpha * total_weight))


def save_results(cluster_results, label, suffix):
    from vir.modeling.clustering.hdbscan import MODEL_NAME

    print("Saving results...")
    if label:
        label = f"_{label}_"
    else:
        label = ""
    cluster_path = CLUSTERING_DIR / f"clustering_results_{suffix}{label}{MODEL_NAME}.parquet"
    cluster_results.write_parquet(cluster_path)

    print(f"✅ Results saved to: {cluster_path}")
    return cluster_path


def load_processed_data(data_path):
    print("Loading processed data...")
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found: {data_path}")

    df = pl.read_parquet(data_path)

    print(f"✅ Data loaded: {len(df):,} records, {len(df.columns)} features")
    print(f"   Unique IDs: {df['unique_id'].n_unique():,}")

    return df
