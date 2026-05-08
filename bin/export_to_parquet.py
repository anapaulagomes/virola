import duckdb

from vir.config import DATA_DIR, logger
from vir.db import PROJECT_DB

output_directory = DATA_DIR / "parquet"
output_directory.mkdir(exist_ok=True)

connection = duckdb.connect(PROJECT_DB, read_only=True)

tables = connection.execute(
    """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
    ORDER BY table_name
    """
).fetchall()

logger.info(f"Found {len(tables)} tables to export.")

for (table_name,) in tables:
    output_path = output_directory / f"{table_name}.parquet"
    logger.info(f"Exporting {table_name} -> {output_path}")
    connection.execute(
        f"COPY (SELECT * FROM {table_name}) TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    row_count = connection.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
    logger.info(f"  {row_count:,} rows")

logger.info("Done.")
