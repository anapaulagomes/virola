import gc
import os
from pathlib import Path

import numpy as np
import polars as pl
import typer
from loguru import logger
from sklearn.decomposition import IncrementalPCA

from vir.config import (
    CHUNK_SIZE,
    EMBEDDING_COMPONENTS,
    PROCESSED_DATA_DIR,
)
from vir.features import AGE_GROUP
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
        # Step 2: Load data with lazy evaluation
        logger.info("2. Loading data with lazy evaluation...")
        lazy_df = load_data_lazy(dataset_filename)

        # Step 3: Load embeddings
        logger.info("3. Loading embeddings...")
        codes_embeddings = load_embeddings()

        # Step 4: Join data
        logger.info("4. Joining data...")
        lazy_joined = join_data_lazy(lazy_df, codes_embeddings)

        # Step 5: Feature selection
        logger.info("5. Feature selection...")
        lazy_processed = lazy_joined.drop(["state", "city"])

        # Step 6: Process in chunks
        logger.info("6. Processing in chunks...")
        final_df, feature_names = process_chunks(lazy_processed, CHUNK_SIZE)

        # Step 7: Save results
        logger.info("7. Saving results...")
        filename = suffix_from(dataset_filename)
        output_path = PROCESSED_DATA_DIR / f"processed_with_original_info_{filename}"
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
