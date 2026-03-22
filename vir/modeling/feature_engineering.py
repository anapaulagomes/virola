import gc
import os
from pathlib import Path

import numpy as np
import polars as pl
import typer
from loguru import logger
from scipy import stats
from sklearn.decomposition import IncrementalPCA

from vir.config import (
    CHUNK_SIZE,
    EMBEDDING_COMPONENTS,
    PROCESSED_DATA_DIR,
)
from vir.features import AGE_GROUP, create_weeks_cyclical_features
from vir.helpers import suffix_from

app = typer.Typer()


def load_data_lazy(dataset_filename):
    """Load data using lazy evaluation for memory efficiency"""
    logger.info(f"Loading dataset: {dataset_filename}")

    lazy_df = pl.scan_parquet(dataset_filename).with_columns(
        [
            pl.col("code").cast(pl.Categorical),
            pl.col("code_type").cast(pl.Categorical),
        ]
    )
    return lazy_df


def load_embeddings():
    logger.info("Loading medical code embeddings...")

    embeddings_df = pl.read_parquet(
        f"{PROCESSED_DATA_DIR}/bra_medter_cambridgeltl__SapBERT-from-PubMedBERT-fulltext.parquet"
    )
    logger.info(f"  Raw embeddings columns: {embeddings_df.columns}")
    logger.info(f"  Raw embeddings shape: {embeddings_df.shape}")

    codes_embeddings = embeddings_df.with_columns(
        [
            pl.col("code").cast(pl.Categorical),
            pl.col("code_type").cast(pl.Categorical),
            # convert embeddings to float32 for memory efficiency
            pl.col("embedding").cast(pl.List(pl.Float32)),
        ]
    )

    logger.info(f"Loaded {len(codes_embeddings):,} embedding entries")
    logger.info(f"Memory usage: {codes_embeddings.estimated_size() / 1024 / 1024:.2f} MB")

    return codes_embeddings


def join_data_lazy(lazy_df, codes_embeddings):
    """Perform lazy join between main data and embeddings"""
    logger.info("Performing lazy join...")
    lazy_embeddings = pl.LazyFrame(codes_embeddings)
    lazy_joined = lazy_df.join(lazy_embeddings, on=["code", "code_type"], how="left")
    return lazy_joined


def process_chunks(lazy_processed, chunk_size=100_000):
    """Process data in chunks to manage memory"""
    logger.info(f"Starting chunked processing with chunk size: {chunk_size:,}")

    ordinal_features = ["age_group_ordinal"]

    logger.info("Getting unique values for encoding...")
    sample_df = lazy_processed.head(10000).collect()
    sex = sample_df["sex"].unique().to_list()

    logger.info(f"Unique SEX values: {sex}")

    ipca = IncrementalPCA(n_components=EMBEDDING_COMPONENTS, batch_size=1000)

    # TODO isn't this done by scikit?
    age_group_mapping = {val: idx for idx, val in enumerate(AGE_GROUP)}
    sex_mapping = {val: idx for idx, val in enumerate(sex)}

    processed_chunks = []
    total_processed = 0

    total_count = lazy_processed.select(pl.len()).collect().item()
    logger.info(f"Total records to process: {total_count:,}")

    for chunk_idx in range(0, total_count, chunk_size):
        logger.info(
            f"Processing chunk {chunk_idx // chunk_size + 1}/{(total_count // chunk_size) + 1}"
        )

        # load chunk using slice for pagination
        chunk = lazy_processed.slice(chunk_idx, chunk_size).collect()
        chunk_with_embeddings = chunk.filter(pl.col("embedding").is_not_null())

        if len(chunk_with_embeddings) == 0:
            logger.info(f"  No records with embeddings in chunk {chunk_idx // chunk_size + 1}")
            continue

        logger.info(f"  Processing {len(chunk_with_embeddings):,} records with embeddings")

        embeddings_list = chunk_with_embeddings["embedding"].to_list()
        embeddings_array = np.array(
            [list(emb) for emb in embeddings_list], dtype=np.float32
        )  # TODO hmmm precisa disso?

        if chunk_idx == 0:
            # FIXME Error: LinAlgError: SVD did not converge for Nova Mutum
            embeddings_processed = ipca.fit_transform(embeddings_array)
        else:
            embeddings_processed = ipca.transform(embeddings_array)

        ordinal_processed = np.column_stack(
            [
                chunk_with_embeddings["age_group"]
                .map_elements(lambda x: age_group_mapping.get(x, 0), return_dtype=pl.Int32)
                .to_numpy(),
            ]
        ).astype(np.float32)

        sex_encoded = np.zeros((len(chunk_with_embeddings), len(sex)), dtype=np.float32)

        for index, sex_val in enumerate(chunk_with_embeddings["sex"].to_list()):
            if sex_val in sex_mapping:
                sex_encoded[index, sex_mapping[sex_val]] = 1.0

        combined_features = np.column_stack(
            [
                ordinal_processed,
                sex_encoded,
                embeddings_processed,
            ]
        )

        embedding_names = [f"embed_{i}" for i in range(EMBEDDING_COMPONENTS)]
        ordinal_names = ordinal_features
        sex_names = [f"sex_{s}" for s in sex]

        feature_names = embedding_names + ordinal_names + sex_names

        chunk_df = pl.DataFrame(combined_features, schema=feature_names)
        keep_columns = [
            "unique_id",
            "year",
            "week",
            "sex",
            "age_group",
            "quantity",
            "code_type",
            "code",
            "description",
        ]
        chunk_df = pl.concat(
            [chunk_with_embeddings.select(keep_columns), chunk_df], how="horizontal"
        )

        processed_chunks.append(chunk_df)
        total_processed += len(chunk_with_embeddings)

        # memory cleanup
        del chunk, chunk_with_embeddings, embeddings_list, embeddings_array
        del embeddings_processed, ordinal_processed
        del sex_encoded, combined_features, chunk_df
        gc.collect()

        logger.info(f"  Total processed so far: {total_processed:,}")

    logger.info("Combining all processed chunks...")

    if not processed_chunks:
        logger.info(
            "❌ No records with embeddings found! Check the join between main data and embeddings."
        )
        logger.info("   This could be due to:")
        logger.info("   - Different code_type values between datasets")
        logger.info("   - Different code formats between datasets")
        logger.info("   - Missing embeddings for the codes in the main dataset")
        raise ValueError("No records with embeddings found. Check the join conditions.")

    final_df = pl.concat(processed_chunks, how="vertical")

    logger.info("Processing completed:")
    logger.info(f"  Total records processed: {len(final_df):,}")
    logger.info(f"  Final features: {len(final_df.columns)}")
    logger.info(f"  Memory usage: {final_df.estimated_size() / 1024 / 1024:.2f} MB")

    return final_df, feature_names


def process_diagnosis_level(lazy_processed, chunk_size=100_000):
    """
    Process data at diagnosis level - aggregate by (code + code_type).

    Creates comprehensive feature vectors per unique diagnosis including:
    - Temporal patterns (52-week series + cyclical features)
    - Demographic distributions (age/sex percentages)
    - Quantitative metrics (volume, peaks, trends, variability)
    - Semantic embeddings (averaged per diagnosis)
    """
    logger.info(f"Starting diagnosis-level processing with chunk size: {chunk_size:,}")

    # Get cyclical features for temporal encoding
    week_cos, week_sin = create_weeks_cyclical_features()

    logger.info("Collecting all data for diagnosis-level aggregation...")

    # Collect all data - needed for proper aggregation
    total_count = lazy_processed.select(pl.len()).collect().item()
    logger.info(f"Total records to process: {total_count:,}")

    all_data = lazy_processed.collect()
    data_with_embeddings = all_data.filter(pl.col("embedding").is_not_null())

    if len(data_with_embeddings) == 0:
        raise ValueError("No records with embeddings found. Check the join conditions.")

    logger.info(f"Records with embeddings: {len(data_with_embeddings):,}")

    # Group by diagnosis (code + code_type)
    logger.info("Grouping by diagnosis (code + code_type)...")
    diagnosis_groups = data_with_embeddings.group_by(["code", "code_type"])

    processed_diagnoses = []
    diagnosis_info = []

    for group_key, group_df in diagnosis_groups:
        code, code_type = group_key

        # Skip if insufficient data
        if len(group_df) < 2:
            continue

        try:
            # 1. TEMPORAL FEATURES
            temporal_features = _extract_temporal_features(group_df, week_cos, week_sin)

            # 2. QUANTITATIVE FEATURES
            quantitative_features = _extract_quantitative_features(group_df)

            # 3. DEMOGRAPHIC FEATURES
            demographic_features = _extract_demographic_features(group_df)

            # 4. SEMANTIC FEATURES (averaged embeddings)
            semantic_features = _extract_semantic_features(group_df)

            # Combine all features
            combined_features = np.concatenate(
                [temporal_features, quantitative_features, demographic_features, semantic_features]
            )

            processed_diagnoses.append(combined_features)
            diagnosis_info.append(
                {
                    "code": code,
                    "code_type": code_type,
                    "total_cases": len(group_df),
                    "description": group_df["description"].first()
                    if "description" in group_df.columns
                    else None,
                }
            )

        except Exception as e:
            logger.warning(f"Skipping diagnosis {code} ({code_type}): {str(e)}")
            continue

    if not processed_diagnoses:
        raise ValueError("No valid diagnoses processed")

    # Convert to DataFrame
    features_array = np.vstack(processed_diagnoses)

    # Create feature names
    feature_names = _create_diagnosis_feature_names()

    # Create final DataFrame
    features_df = pl.DataFrame(features_array, schema=feature_names)
    diagnosis_info_df = pl.DataFrame(diagnosis_info)

    final_df = pl.concat([diagnosis_info_df, features_df], how="horizontal")

    logger.info("Diagnosis-level processing completed:")
    logger.info(f"  Unique diagnoses processed: {len(final_df):,}")
    logger.info(f"  Features per diagnosis: {len(feature_names)}")
    logger.info(f"  Memory usage: {final_df.estimated_size() / 1024 / 1024:.2f} MB")

    return final_df, feature_names


def _extract_temporal_features(group_df, week_cos, week_sin):
    """Extract temporal features: 52-week series + cyclical features."""
    # Create 52-week series
    weekly_counts = np.zeros(53)  # weeks 0-52, using week 0 as placeholder

    for week, count in zip(group_df["week"].to_list(), group_df["quantity"].to_list()):
        if 0 <= week <= 52:
            weekly_counts[week] += count

    # Use weeks 1-52 (skip week 0)
    weekly_series = weekly_counts[1:]

    # Normalize series (Min-Max normalization)
    if np.max(weekly_series) > np.min(weekly_series):
        normalized_series = (weekly_series - np.min(weekly_series)) / (
            np.max(weekly_series) - np.min(weekly_series)
        )
    else:
        normalized_series = (
            weekly_series / np.max(weekly_series) if np.max(weekly_series) > 0 else weekly_series
        )

    # Find peak week for cyclical features
    peak_week = np.argmax(weekly_series) + 1  # +1 because we start from week 1
    cyclical_features = np.array([week_sin[peak_week], week_cos[peak_week]])

    return np.concatenate([normalized_series, cyclical_features])


def _extract_quantitative_features(group_df):
    """Extract quantitative features: volume, intensity, peaks, trends, variability."""
    quantities = group_df["quantity"].to_numpy()

    # Basic volume metrics
    total_cases = np.sum(quantities)
    total_cases_log = np.log1p(total_cases)  # log(1 + x) to handle zeros

    # Weekly aggregation for temporal analysis
    weekly_counts = np.zeros(53)
    for week, count in zip(group_df["week"].to_list(), group_df["quantity"].to_list()):
        if 0 <= week <= 52:
            weekly_counts[week] += count

    weekly_series = weekly_counts[1:]  # weeks 1-52

    # Mean weekly cases
    mean_weekly = np.mean(weekly_series)
    mean_weekly_log = np.log1p(mean_weekly)

    # Peak intensity (peak / mean ratio)
    peak_intensity = np.max(weekly_series) / mean_weekly if mean_weekly > 0 else 0

    # Temporal slope (trend)
    weeks = np.arange(1, 53)
    if len(weekly_series) == 52 and np.sum(weekly_series) > 0:
        slope, _, _, _, _ = stats.linregress(weeks, weekly_series)
        # Normalize slope by mean to make comparable across diseases
        temporal_slope = slope / mean_weekly if mean_weekly > 0 else 0
    else:
        temporal_slope = 0

    # Coefficient of variation
    if mean_weekly > 0:
        coefficient_variation = np.std(weekly_series) / mean_weekly
    else:
        coefficient_variation = 0

    return np.array(
        [total_cases_log, mean_weekly_log, peak_intensity, temporal_slope, coefficient_variation]
    )


def _extract_demographic_features(group_df):
    """Extract demographic distribution features."""
    # Age distribution
    age_counts = group_df.group_by("age_group").agg(pl.col("quantity").sum())
    total_cases = age_counts["quantity"].sum()

    age_distribution = np.zeros(len(AGE_GROUP))
    for age_group, count in zip(
        age_counts["age_group"].to_list(), age_counts["quantity"].to_list()
    ):
        if age_group in AGE_GROUP:
            idx = AGE_GROUP.index(age_group)
            age_distribution[idx] = count / total_cases if total_cases > 0 else 0

    # Sex distribution
    sex_counts = group_df.group_by("sex").agg(pl.col("quantity").sum())
    sex_values = sex_counts["sex"].to_list()
    sex_counts_values = sex_counts["quantity"].to_list()

    # Assuming binary sex classification
    sex_distribution = np.zeros(2)
    for i, sex_val in enumerate(sex_values):
        if i < 2:  # Safety check
            sex_distribution[i] = sex_counts_values[i] / total_cases if total_cases > 0 else 0

    # Ensure distributions sum to 1 (normalize)
    if np.sum(age_distribution) > 0:
        age_distribution = age_distribution / np.sum(age_distribution)
    if np.sum(sex_distribution) > 0:
        sex_distribution = sex_distribution / np.sum(sex_distribution)

    return np.concatenate([age_distribution, sex_distribution])


def _extract_semantic_features(group_df):
    """Extract semantic features: averaged embeddings."""
    embeddings_list = group_df["embedding"].to_list()

    # Convert to numpy array and compute mean
    embeddings_array = np.array([list(emb) for emb in embeddings_list], dtype=np.float32)
    mean_embedding = np.mean(embeddings_array, axis=0)

    return mean_embedding


def _create_diagnosis_feature_names():
    """Create feature names for diagnosis-level features."""
    # Temporal features: 52 weeks + 2 cyclical
    temporal_names = [f"week_{i}" for i in range(1, 53)]
    cyclical_names = ["peak_week_sin", "peak_week_cos"]

    # Quantitative features
    quantitative_names = [
        "total_cases_log",
        "mean_weekly_log",
        "peak_intensity",
        "temporal_slope",
        "coefficient_variation",
    ]

    # Demographic features
    age_names = [f"age_{group.replace(' ', '_').replace('>', 'gt')}" for group in AGE_GROUP]
    sex_names = ["sex_0", "sex_1"]  # Generic sex distribution

    # Semantic features (embeddings)
    # Note: Using original embedding dimension since we haven't applied PCA yet
    embedding_names = [f"embedding_{i}" for i in range(768)]  # Assuming SapBERT dimensions

    return (
        temporal_names
        + cyclical_names
        + quantitative_names
        + age_names
        + sex_names
        + embedding_names
    )


@app.command()
def main(dataset_filename: Path, level: str = "records"):
    """Main processing pipeline.

    Args:
        dataset_filename: Path to cleaned input file
        level: Processing level - "records" (default) or "diagnosis"

    Expects a cleaned file as input e.g.
    cleaned_aesop_raw_dados_2017_20250223_20municipios_082025_salvador.parquet
    """
    logger.info("=" * 60)
    logger.info("virola - feature engineering step")
    logger.info(f"Processing level: {level}")
    logger.info("=" * 60)

    if level not in ["records", "diagnosis"]:
        raise ValueError(f"Invalid level '{level}'. Must be 'records' or 'diagnosis'")

    try:
        # Step 1: Load data with lazy evaluation
        logger.info("1. Loading data with lazy evaluation...")
        lazy_df = load_data_lazy(dataset_filename)

        # Step 2: Load embeddings
        logger.info("2. Loading embeddings...")
        codes_embeddings = load_embeddings()

        # Step 3: Join data
        logger.info("3. Joining data...")
        lazy_joined = join_data_lazy(lazy_df, codes_embeddings)

        # Step 4: Feature selection
        logger.info("4. Feature selection...")
        lazy_processed = lazy_joined.drop(["state", "city"])

        # Step 5: Process based on selected level
        if level == "diagnosis":
            logger.info("5. Processing at diagnosis level...")
            final_df, feature_names = process_diagnosis_level(lazy_processed, CHUNK_SIZE)
            level_suffix = "diagnosis_level"
        else:
            logger.info("5. Processing at record level...")
            final_df, feature_names = process_chunks(lazy_processed, CHUNK_SIZE)
            level_suffix = "record_level"

        # Step 6: Save results
        logger.info("6. Saving results...")
        filename = suffix_from(dataset_filename)
        # Insert level suffix before file extension
        base_name = filename.replace(".parquet", "")
        output_path = PROCESSED_DATA_DIR / f"processed{base_name}_{level_suffix}.parquet"
        final_df.write_parquet(output_path)

        logger.success("\n✅ Processing completed successfully!")
        logger.success(f"📁 Output file: {output_path}")
        logger.success(f"📊 Records: {len(final_df):,}")
        logger.success(f"🔢 Features: {len(final_df.columns)}")
        logger.success(f"💾 File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

        if level == "diagnosis":
            unique_codes = final_df.select("code").n_unique()
            logger.success(f"🏥 Unique diagnoses: {unique_codes:,}")

    except Exception as e:
        logger.error(f"\n❌ Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    app()
