import gc
import os
from pathlib import Path

import numpy as np
import polars as pl
import typer
from loguru import logger
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

from vir.config import (
    CHUNK_SIZE,
    EMBEDDING_COMPONENTS,
    PROCESSED_DATA_DIR,
)
from vir.features import AGE_GROUP, POPULATION_RANGE, create_weeks_cyclical_features
from vir.helpers import suffix_from

app = typer.Typer()


def load_data_lazy(dataset_filename, week_cos, week_sin):
    """Load data using lazy evaluation for memory efficiency"""
    logger.info(f"Loading dataset: {dataset_filename}")

    lazy_df = pl.scan_parquet(dataset_filename).with_columns(
        [
            pl.col("code").cast(pl.Categorical),
            pl.col("code_type").cast(pl.Categorical),
            pl.col("week")
            .map_elements(lambda w: week_sin.get(w, 0.0), return_dtype=pl.Float32)
            .alias("week_sin"),
            pl.col("week")
            .map_elements(lambda w: week_cos.get(w, 0.0), return_dtype=pl.Float32)
            .alias("week_cos"),
        ]
    )
    return lazy_df


def load_embeddings():
    logger.info("Loading medical code embeddings...")

    _code_types = {
        "cid10": "CID",
        "ciap2": "CIAP",
        "ab": "AB",
    }
    embeddings_df = pl.read_parquet(
        f"{PROCESSED_DATA_DIR}/bra_medter_cambridgeltl__SapBERT-from-PubMedBERT-fulltext.parquet"
    )
    logger.info(f"  Raw embeddings columns: {embeddings_df.columns}")
    logger.info(f"  Raw embeddings shape: {embeddings_df.shape}")

    codes_embeddings = embeddings_df.with_columns(
        [
            pl.col("code").cast(pl.Categorical),
            # map code_type from embeddings format to main dataset format
            pl.col("code_type")
            .map_elements(lambda x: _code_types.get(x, x), return_dtype=pl.String)
            .cast(pl.Categorical),
            # convert embeddings to float32 for memory efficiency
            pl.col("embeddings").cast(pl.List(pl.Float32)),
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

    numeric_features = ["week_sin", "week_cos", "year"]
    ordinal_features = ["age_group", "population_range"]

    logger.info("Getting unique values for encoding...")
    sample_df = lazy_processed.head(10000).collect()
    sex = sample_df["sex"].unique().to_list()
    regions = sample_df["region"].unique().to_list()

    logger.info(f"Unique SEX values: {sex}")
    logger.info(f"Unique REGIONS values: {regions}")

    ipca = IncrementalPCA(n_components=EMBEDDING_COMPONENTS, batch_size=1000)
    numeric_scaler = StandardScaler()

    # TODO isn't this done by scikit?
    age_group_mapping = {val: idx for idx, val in enumerate(AGE_GROUP)}
    population_mapping = {val: idx for idx, val in enumerate(POPULATION_RANGE)}
    sex_mapping = {val: idx for idx, val in enumerate(sex)}
    region_mapping = {val: idx for idx, val in enumerate(regions)}

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
        chunk_with_embeddings = chunk.filter(pl.col("embeddings").is_not_null())

        if len(chunk_with_embeddings) == 0:
            logger.info(f"  No records with embeddings in chunk {chunk_idx // chunk_size + 1}")
            continue

        logger.info(f"  Processing {len(chunk_with_embeddings):,} records with embeddings")

        embeddings_list = chunk_with_embeddings["embeddings"].to_list()
        embeddings_array = np.array(
            [list(emb) for emb in embeddings_list], dtype=np.float32
        )  # TODO hmmm precisa disso?

        if chunk_idx == 0:
            embeddings_processed = ipca.fit_transform(embeddings_array)
        else:
            embeddings_processed = ipca.transform(embeddings_array)

        numeric_data = chunk_with_embeddings.select(numeric_features).to_numpy()
        if chunk_idx == 0:
            numeric_processed = numeric_scaler.fit_transform(numeric_data)
        else:
            numeric_processed = numeric_scaler.transform(numeric_data)

        ordinal_processed = np.column_stack(
            [
                chunk_with_embeddings["age_group"]
                .map_elements(lambda x: age_group_mapping.get(x, 0), return_dtype=pl.Int32)
                .to_numpy(),
                chunk_with_embeddings["population_range"]
                .map_elements(lambda x: population_mapping.get(x, 0), return_dtype=pl.Int32)
                .to_numpy(),
            ]
        ).astype(np.float32)

        # one-hot encoding
        sex_encoded = np.zeros((len(chunk_with_embeddings), len(sex)), dtype=np.float32)
        region_encoded = np.zeros((len(chunk_with_embeddings), len(regions)), dtype=np.float32)

        for index, (sex_val, region_val) in enumerate(
            zip(chunk_with_embeddings["sex"].to_list(), chunk_with_embeddings["region"].to_list())
        ):
            if sex_val in sex_mapping:
                sex_encoded[index, sex_mapping[sex_val]] = 1.0
            if region_val in region_mapping:
                region_encoded[index, region_mapping[region_val]] = 1.0

        combined_features = np.column_stack(
            [
                embeddings_processed,
                numeric_processed,
                ordinal_processed,
                sex_encoded,
                region_encoded,
            ]
        )

        embedding_names = [f"embed_{i}" for i in range(EMBEDDING_COMPONENTS)]
        numeric_names = numeric_features
        ordinal_names = ordinal_features
        sex_names = [f"sex_{s}" for s in sex]
        region_names = [f"region_{r}" for r in regions]

        feature_names = embedding_names + numeric_names + ordinal_names + sex_names + region_names

        # Create polars dataframe for this chunk
        chunk_df = pl.DataFrame(combined_features, schema=feature_names)

        # Add unique_id back # TODO check this
        chunk_df = pl.concat(
            [chunk_with_embeddings.select("unique_id"), chunk_df], how="horizontal"
        )

        processed_chunks.append(chunk_df)
        total_processed += len(chunk_with_embeddings)

        # memory cleanup
        del chunk, chunk_with_embeddings, embeddings_list, embeddings_array
        del embeddings_processed, numeric_data, numeric_processed, ordinal_processed
        del sex_encoded, region_encoded, combined_features, chunk_df
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


@app.command()
def main(dataset_filename: Path):
    """Main processing pipeline.

    Expects a cleaned file as input e.g.
    cleaned_aesop_raw_dados_2017_20250223_20municipios_082025_salvador.parquet
    """
    logger.info("=" * 60)
    logger.info("virola - feature engineering step")
    logger.info("=" * 60)

    try:
        # Step 1: Create cyclical features
        logger.info("1. Creating cyclical features...")
        week_cos, week_sin = create_weeks_cyclical_features()

        # Step 2: Load data with lazy evaluation
        logger.info("2. Loading data with lazy evaluation...")
        lazy_df = load_data_lazy(dataset_filename, week_cos, week_sin)

        # Step 3: Load embeddings
        logger.info("3. Loading embeddings...")
        codes_embeddings = load_embeddings()

        # Step 4: Join data
        logger.info("4. Joining data...")
        lazy_joined = join_data_lazy(lazy_df, codes_embeddings)

        # Step 5: Feature selection
        logger.info("5. Feature selection...")
        lazy_processed = lazy_joined.drop(["week", "state", "city"])

        # Step 6: Process in chunks
        logger.info("6. Processing in chunks...")
        final_df, feature_names = process_chunks(lazy_processed, CHUNK_SIZE)

        # Step 7: Save results
        logger.info("7. Saving results...")
        filename = suffix_from(dataset_filename)
        output_path = PROCESSED_DATA_DIR / f"processed_optimized{filename}"
        final_df.write_parquet(output_path)

        logger.success("\n✅ Processing completed successfully!")
        logger.success(f"📁 Output file: {output_path}")
        logger.success(f"📊 Records: {len(final_df):,}")
        logger.success(f"🔢 Features: {len(final_df.columns)}")
        logger.success(f"💾 File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    except Exception as e:
        logger.info(f"\n❌ Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    app()
