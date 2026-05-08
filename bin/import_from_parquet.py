from vir.config import DATA_DIR, logger
from vir.db import (
    create_dimensions_table_if_not_exists,
    create_embeddings_table_if_not_exists,
    create_results_schema,
    get_connection,
)

input_directory = DATA_DIR / "parquet"

if not input_directory.exists():
    raise FileNotFoundError(f"Parquet directory not found: {input_directory}")

import_order = [
    "medical_terminologies",
    "dimensions",
    "runs",
    "cluster_assignments",
    "cluster_profiles",
    "run_multi_k",
    "run_leiden",
]

create_embeddings_table_if_not_exists()
create_dimensions_table_if_not_exists()
create_results_schema()

connection = get_connection(write=True)

connection.execute("SET preserve_insertion_order=false")

for table_name in import_order:
    parquet_path = input_directory / f"{table_name}.parquet"
    if not parquet_path.exists():
        logger.warning(f"Skipping {table_name}: {parquet_path} not found")
        continue

    existing_rows = connection.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
    if existing_rows > 0:
        logger.warning(f"Skipping {table_name}: table already has {existing_rows:,} rows")
        continue

    logger.info(f"Importing {parquet_path} -> {table_name}")
    connection.execute(f"INSERT INTO {table_name} SELECT * FROM read_parquet('{parquet_path}')")
    inserted_rows = connection.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
    logger.info(f"  {inserted_rows:,} rows")

sequence_resets = [
    ("id_sequence", "SELECT COALESCE(MAX(id), 0) FROM medical_terminologies"),
    ("run_id_sequence", "SELECT COALESCE(MAX(run_id), 0) FROM runs"),
]
for sequence_name, max_query in sequence_resets:
    max_value = connection.execute(max_query).fetchone()[0]
    next_value = max_value + 1
    connection.execute(f"CREATE OR REPLACE SEQUENCE {sequence_name} START WITH {next_value}")
    logger.info(f"Reset {sequence_name} to start at {next_value}")

logger.info("Done.")
