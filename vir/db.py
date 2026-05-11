import atexit
import os
import re
import signal
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

import duckdb
import numpy as np
import polars as pl

from vir.adapters.datasus.terminologies.terminology_mapping import load_mapping
from vir.config import DATA_DIR, DEFAULT_MODEL, MODEL_DIMENSIONS, logger
from vir.features import IBGE_CITY_MAP
from vir.helpers import extract_dataset_name, get_n_jobs

PROJECT_DB = DATA_DIR / "virola.db"

# ---------------------------------------------------------------------------
# Connection
#
# Two singletons: one read-only (default), one read-write.
#
# DuckDB local-file locking rules:
#   - At most ONE writer at a time. While a writer holds the lock, no other
#     process can open the file (not even read-only).
#   - Multiple read-only connections are allowed simultaneously when there is
#     no writer.
#
# Strategy:
#   - All reads use get_connection() — read-only by default. Multiple
#     notebooks can coexist freely, and will retry while a writer holds the lock.
#   - Write operations use get_connection(write=True). Only one write
#     connection exists per process; the atexit handler closes it on clean exit.
# ---------------------------------------------------------------------------

_write_connection: duckdb.DuckDBPyConnection | None = None
_read_only_connection: duckdb.DuckDBPyConnection | None = None
DB_CONFIG = {"threads": get_n_jobs()}


def _parse_lock_pid(error_message: str) -> int | None:
    """Extract the PID from a DuckDB lock error message.

    DuckDB reports: ``Conflicting lock is held in ... (PID 12345) by user ...``
    """
    match = re.search(r"\(PID\s+(\d+)\)", error_message)
    return int(match.group(1)) if match else None


def _release_orphaned_lock(pid: int) -> bool:
    """Send SIGTERM to a process if it belongs to the current user.

    Returns True if the signal was sent, False if the process doesn't exist
    or belongs to a different user.
    """
    try:
        os.kill(pid, signal.SIGTERM)

        logger.warning(f"Released orphaned DuckDB lock held by PID {pid}.")
        time.sleep(0.5)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return False


def _open_connection(*, write: bool, retries: int, delay: float) -> duckdb.DuckDBPyConnection:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return duckdb.connect(PROJECT_DB, read_only=not write, config=DB_CONFIG)
        except duckdb.IOException as error:
            last_exc = error
            if attempt == 0:
                pid = _parse_lock_pid(str(error))
                if pid and _release_orphaned_lock(pid):
                    continue
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def get_connection(
    *, write: bool = False, retries: int = 20, delay: float = 0.5
) -> duckdb.DuckDBPyConnection:
    """Return the shared connection for this process.

    Read-only by default — safe to call from notebooks and all read paths.
    Multiple read-only connections can coexist across processes.

    Pass ``write=True`` only from code that inserts or updates data.
    A write connection requires exclusive file access and blocks all other
    processes until it is closed.

    DuckDB forbids opening the same file with different configurations in the
    same process. To avoid this, read requests are served by the write
    connection when one is already open (a write connection can read too).
    Conversely, if a write connection is requested after a read-only one was
    already opened, the read-only connection is closed first.

    The retry loop handles cross-process contention: callers wait up to
    ``retries * delay`` seconds before raising.
    """
    global _write_connection, _read_only_connection

    if write:
        if _write_connection is None:
            if _read_only_connection is not None:
                _close_read_only_connection()
            _write_connection = _open_connection(write=True, retries=retries, delay=delay)
            atexit.register(_close_write_connection)
        return _write_connection
    else:
        if _write_connection is not None:
            return _write_connection
        if _read_only_connection is None:
            _read_only_connection = _open_connection(write=False, retries=retries, delay=delay)
            atexit.register(_close_read_only_connection)
        return _read_only_connection


def create_results_schema():
    """Create runs, cluster_assignments and cluster_profiles tables if they don't exist."""
    with _write_context() as con:
        con.execute("""
            CREATE SEQUENCE IF NOT EXISTS run_id_sequence START 1;

            CREATE TABLE IF NOT EXISTS runs (
                run_id          INTEGER PRIMARY KEY DEFAULT nextval('run_id_sequence'),
                dataset_name    VARCHAR NOT NULL,
                city            VARCHAR,
                region          VARCHAR,
                population_range VARCHAR,
                year_min        INTEGER NOT NULL,
                year_max        INTEGER NOT NULL,
                snf_k           INTEGER NOT NULL,
                n_clusters      INTEGER NOT NULL,
                indicative_enabled BOOLEAN NOT NULL DEFAULT false,
                silhouette_score   FLOAT,
                affinity_zscore    FLOAT,
                notes           VARCHAR,
                created_at      TIMESTAMP DEFAULT now(),
                finished_at     TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS cluster_assignments (
                run_id          INTEGER NOT NULL REFERENCES runs(run_id),
                terminology_id  INTEGER NOT NULL REFERENCES medical_terminologies(id),
                cluster_id      INTEGER NOT NULL,
                PRIMARY KEY (run_id, terminology_id)
            );

            CREATE TABLE IF NOT EXISTS cluster_profiles (
                run_id                  INTEGER NOT NULL REFERENCES runs(run_id),
                cluster_id              INTEGER NOT NULL,
                n_codes                 INTEGER,
                total_cases             INTEGER,
                top_codes               JSON,
                temporal_signature      JSON,
                demographic_profile     JSON,
                explanation             JSON,
                umap_x                  FLOAT,
                umap_y                  FLOAT,
                llm_short_description   VARCHAR,
                llm_description         VARCHAR,
                llm_abbreviation        VARCHAR,
                PRIMARY KEY (run_id, cluster_id)
            );

            CREATE TABLE IF NOT EXISTS run_multi_k (
                run_id          INTEGER NOT NULL REFERENCES runs(run_id),
                k               INTEGER NOT NULL,
                silhouette      FLOAT,
                largest_cluster INTEGER,
                median_size     FLOAT,
                affinity_zscore FLOAT,
                PRIMARY KEY (run_id, k)
            );

            CREATE TABLE IF NOT EXISTS run_leiden (
                run_id        INTEGER NOT NULL REFERENCES runs(run_id),
                resolution    FLOAT NOT NULL,
                n_communities INTEGER NOT NULL,
                PRIMARY KEY (run_id, resolution)
            );
        """)

        # Migrate existing runs table — add new columns if they don't exist yet
        con.execute("""
            ALTER TABLE run_multi_k ADD COLUMN IF NOT EXISTS affinity_zscore FLOAT;
        """)

        con.execute("""
            ALTER TABLE runs ADD COLUMN IF NOT EXISTS n_codes_clinical    INTEGER;
            ALTER TABLE runs ADD COLUMN IF NOT EXISTS n_codes_temporal    INTEGER;
            ALTER TABLE runs ADD COLUMN IF NOT EXISTS n_codes_demographic INTEGER;
            ALTER TABLE runs ADD COLUMN IF NOT EXISTS n_codes_common      INTEGER;
            ALTER TABLE runs ADD COLUMN IF NOT EXISTS n_missing_temporal  INTEGER;
            ALTER TABLE runs ADD COLUMN IF NOT EXISTS n_missing_demographic INTEGER;
            ALTER TABLE runs ADD COLUMN IF NOT EXISTS cluster_largest     INTEGER;
            ALTER TABLE runs ADD COLUMN IF NOT EXISTS cluster_median      FLOAT;
            ALTER TABLE runs ADD COLUMN IF NOT EXISTS n_singletons        INTEGER;
            ALTER TABLE runs ADD COLUMN IF NOT EXISTS pct_singletons      FLOAT;
            ALTER TABLE runs ADD COLUMN IF NOT EXISTS adapter             VARCHAR;
        """)

        con.execute("UPDATE runs SET adapter = 'datasus' WHERE adapter IS NULL;")


def create_embeddings_table_if_not_exists():
    with _write_context() as con:
        con.execute("""
            CREATE SEQUENCE IF NOT EXISTS id_sequence START 1;
            CREATE TABLE IF NOT EXISTS medical_terminologies(
                id INTEGER PRIMARY KEY DEFAULT nextval('id_sequence'),
                code_type VARCHAR,
                code VARCHAR,
                description VARCHAR,
                model VARCHAR,
                embedding FLOAT[],
                UNIQUE(code, code_type, model)
            );
        """)


def create_dimensions_table_if_not_exists():
    with _write_context() as con:
        con.execute("""
            CREATE SEQUENCE IF NOT EXISTS id_sequence START 1;
            CREATE TABLE IF NOT EXISTS dimensions (
                terminology_id_1 INTEGER NOT NULL REFERENCES medical_terminologies(id),
                terminology_id_2 INTEGER NOT NULL REFERENCES medical_terminologies(id),
                value FLOAT4 NOT NULL,
                dimension VARCHAR NOT NULL,
                PRIMARY KEY (terminology_id_1, terminology_id_2, dimension),
                CHECK (terminology_id_1 < terminology_id_2)
            );
        """)


def insert_run(
    dataset_name: str,
    city: str | None,
    region: str | None,
    population_range: str | None,
    year_min: int,
    year_max: int,
    snf_k: int,
    n_clusters: int,
    indicative_enabled: bool = False,
    silhouette_score: float | None = None,
    affinity_zscore: float | None = None,
    notes: str | None = None,
    created_at: str | None = None,
    finished_at: str | None = None,
    n_codes_clinical: int | None = None,
    n_codes_temporal: int | None = None,
    n_codes_demographic: int | None = None,
    n_codes_common: int | None = None,
    n_missing_temporal: int | None = None,
    n_missing_demographic: int | None = None,
    cluster_largest: int | None = None,
    cluster_median: float | None = None,
    n_singletons: int | None = None,
    pct_singletons: float | None = None,
) -> int:
    """Insert a new run and return its run_id."""
    create_results_schema()
    _created_at = created_at if created_at is not None else datetime.now().isoformat()
    with _write_context() as con:
        result = con.execute(
            """
            INSERT INTO runs (
                dataset_name, city, region, population_range,
                year_min, year_max, snf_k, n_clusters,
                indicative_enabled, silhouette_score, affinity_zscore, notes, created_at, finished_at,
                n_codes_clinical, n_codes_temporal, n_codes_demographic, n_codes_common,
                n_missing_temporal, n_missing_demographic,
                cluster_largest, cluster_median, n_singletons, pct_singletons
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING run_id
            """,
            [
                dataset_name,
                city,
                region,
                population_range,
                year_min,
                year_max,
                snf_k,
                n_clusters,
                indicative_enabled,
                silhouette_score,
                affinity_zscore,
                notes,
                _created_at,
                finished_at,
                n_codes_clinical,
                n_codes_temporal,
                n_codes_demographic,
                n_codes_common,
                n_missing_temporal,
                n_missing_demographic,
                cluster_largest,
                cluster_median,
                n_singletons,
                pct_singletons,
            ],
        ).fetchone()
    return result[0]


def update_run_finished_at(run_id: int) -> None:
    """Stamp finished_at on an existing run with the current timestamp."""
    with _write_context() as con:
        con.execute("UPDATE runs SET finished_at = now() WHERE run_id = ?", [run_id])


def delete_run(run_id: int) -> bool:
    """Remove a run, every dependent row, and its parquet sidecar.

    Returns True if the run existed.
    """
    with _write_context() as con:
        # DuckDB's FK checker doesn't see prior deletes within the same
        # transaction, so each statement runs under autocommit. The deletes
        # are idempotent — re-running finishes any partial cleanup.
        con.execute("DELETE FROM cluster_profiles WHERE run_id = ?", [run_id])
        con.execute("DELETE FROM cluster_assignments WHERE run_id = ?", [run_id])
        con.execute("DELETE FROM run_multi_k WHERE run_id = ?", [run_id])
        con.execute("DELETE FROM run_leiden WHERE run_id = ?", [run_id])
        row = con.execute(
            "DELETE FROM runs WHERE run_id = ? RETURNING run_id", [run_id]
        ).fetchone()

    if row is None:
        return False

    (DATA_DIR / "runs" / f"{run_id}.parquet").unlink(missing_ok=True)
    return True


def find_run(
    run_id: int | None = None,
    dataset_path: str | None = None,
) -> dict:
    """Return a single run as a dict.

    Selection priority:
    1. ``run_id`` — return that specific run.
    2. ``dataset_path`` — infer dataset_name from the path and return the latest run for it.
    3. Neither — return the latest run overall.

    Raises ``ValueError`` if no matching run is found.
    """
    runs = fetch_runs()
    if runs.is_empty():
        raise ValueError("No runs found in the database.")

    if run_id is not None:
        subset = runs.filter(pl.col("run_id") == run_id)
        if subset.is_empty():
            raise ValueError(f"Run {run_id} not found.")
        return subset.row(0, named=True)

    if dataset_path is not None:
        name = extract_dataset_name(dataset_path)
        subset = runs.filter(pl.col("dataset_name") == name)
        if subset.is_empty():
            raise ValueError(f"No runs found for dataset '{name}'.")
        return subset.row(0, named=True)

    return runs.row(0, named=True)


def fetch_runs() -> pl.DataFrame:
    """Return all runs as a Polars DataFrame, including duration in minutes."""
    return (
        get_connection()
        .execute("""
        SELECT run_id, dataset_name, city, region, population_range,
               year_min, year_max, snf_k, n_clusters,
               indicative_enabled, silhouette_score, affinity_zscore,
               n_codes_clinical, n_codes_temporal, n_codes_demographic, n_codes_common,
               n_missing_temporal, n_missing_demographic,
               cluster_largest, cluster_median, n_singletons, pct_singletons,
               notes, created_at, finished_at,
               CASE
                   WHEN finished_at IS NOT NULL
                   THEN ROUND(EXTRACT(EPOCH FROM (finished_at - created_at)) / 60.0, 1)
               END AS duration_min
        FROM runs ORDER BY created_at DESC
    """)
        .pl()
    )


def insert_cluster_assignments(run_id: int, assignments: list[dict]) -> None:
    """Bulk insert cluster assignments for a run.

    Args:
        run_id: The run identifier.
        assignments: List of {terminology_id, cluster_id} dicts.
    """
    df = pl.DataFrame(assignments).with_columns(pl.lit(run_id).alias("run_id"))
    with _write_context() as con:
        con.register("assignments_view", df)
        con.execute("""
            INSERT INTO cluster_assignments (run_id, terminology_id, cluster_id)
            SELECT run_id, terminology_id, cluster_id FROM assignments_view
        """)


def fetch_cluster_assignments(run_id: int) -> pl.DataFrame:
    """Return all assignments for a run joined with terminology metadata."""
    return (
        get_connection()
        .execute(
            """
        SELECT ca.cluster_id, ca.terminology_id, mt.code, mt.code_type, mt.description
        FROM cluster_assignments ca
        JOIN medical_terminologies mt ON ca.terminology_id = mt.id
        WHERE ca.run_id = ?
        ORDER BY ca.cluster_id, mt.code
        """,
            [run_id],
        )
        .pl()
    )


def upsert_cluster_profile(
    run_id: int,
    cluster_id: int,
    *,
    n_codes: int | None = None,
    total_cases: int | None = None,
    top_codes: str | None = None,
    temporal_signature: str | None = None,
    demographic_profile: str | None = None,
    explanation: str | None = None,
    umap_x: float | None = None,
    umap_y: float | None = None,
    llm_short_description: str | None = None,
    llm_description: str | None = None,
    llm_abbreviation: str | None = None,
) -> None:
    """Insert or update a cluster profile (subset of fields can be provided)."""
    create_results_schema()
    with _write_context() as con:
        con.execute(
            """
            INSERT INTO cluster_profiles (
                run_id, cluster_id, n_codes, total_cases,
                top_codes, temporal_signature, demographic_profile, explanation,
                umap_x, umap_y,
                llm_short_description, llm_description, llm_abbreviation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (run_id, cluster_id) DO UPDATE SET
                n_codes               = COALESCE(EXCLUDED.n_codes, cluster_profiles.n_codes),
                total_cases           = COALESCE(EXCLUDED.total_cases, cluster_profiles.total_cases),
                top_codes             = COALESCE(EXCLUDED.top_codes, cluster_profiles.top_codes),
                temporal_signature    = COALESCE(EXCLUDED.temporal_signature, cluster_profiles.temporal_signature),
                demographic_profile   = COALESCE(EXCLUDED.demographic_profile, cluster_profiles.demographic_profile),
                explanation           = COALESCE(EXCLUDED.explanation, cluster_profiles.explanation),
                umap_x                = COALESCE(EXCLUDED.umap_x, cluster_profiles.umap_x),
                umap_y                = COALESCE(EXCLUDED.umap_y, cluster_profiles.umap_y),
                llm_short_description = COALESCE(EXCLUDED.llm_short_description, cluster_profiles.llm_short_description),
                llm_description       = COALESCE(EXCLUDED.llm_description, cluster_profiles.llm_description),
                llm_abbreviation      = COALESCE(EXCLUDED.llm_abbreviation, cluster_profiles.llm_abbreviation)
            """,
            [
                run_id,
                cluster_id,
                n_codes,
                total_cases,
                top_codes,
                temporal_signature,
                demographic_profile,
                explanation,
                umap_x,
                umap_y,
                llm_short_description,
                llm_description,
                llm_abbreviation,
            ],
        )


def fetch_cluster_profiles(run_id: int) -> pl.DataFrame:
    """Return all profiles for a run."""
    return (
        get_connection()
        .execute(
            "SELECT * FROM cluster_profiles WHERE run_id = ? ORDER BY cluster_id",
            [run_id],
        )
        .pl()
    )


def fetch_cluster_profiles_for_runs(run_ids: list[int]) -> pl.DataFrame:
    """Return cluster profiles for multiple runs, ordered by run_id and cluster_id."""
    if not run_ids:
        return pl.DataFrame()
    return (
        get_connection()
        .execute(
            "SELECT * FROM cluster_profiles WHERE run_id IN (SELECT unnest(?::INTEGER[])) ORDER BY run_id, cluster_id",
            [run_ids],
        )
        .pl()
    )


def insert_run_multi_k(run_id: int, results: list[dict]) -> None:
    """Bulk insert multi-K comparison results for a run.

    Args:
        run_id: The run identifier.
        results: List of {n_clusters, silhouette, largest_cluster, median_size, affinity_zscore} dicts.
    """
    rows = [
        (
            run_id,
            r["n_clusters"],
            r["silhouette"],
            r["largest_cluster"],
            r["median_size"],
            r.get("affinity_zscore"),
        )
        for r in results
    ]
    with _write_context() as con:
        con.executemany(
            """
            INSERT INTO run_multi_k (run_id, k, silhouette, largest_cluster, median_size, affinity_zscore)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (run_id, k) DO UPDATE SET
                silhouette      = EXCLUDED.silhouette,
                largest_cluster = EXCLUDED.largest_cluster,
                median_size     = EXCLUDED.median_size,
                affinity_zscore = EXCLUDED.affinity_zscore
            """,
            rows,
        )


def fetch_run_multi_k(run_id: int) -> pl.DataFrame:
    """Return multi-K comparison results for a run as a Polars DataFrame."""
    return (
        get_connection()
        .execute(
            """
            SELECT k, silhouette, largest_cluster, median_size, affinity_zscore
            FROM run_multi_k WHERE run_id = ? ORDER BY k
            """,
            [run_id],
        )
        .pl()
    )


def insert_run_leiden(run_id: int, results: list[dict]) -> None:
    """Bulk insert Leiden community detection sweep results for a run.

    Args:
        run_id: The run identifier.
        results: List of {resolution, n_communities} dicts.
    """
    rows = [(run_id, r["resolution"], r["n_communities"]) for r in results]
    with _write_context() as con:
        con.executemany(
            """
            INSERT INTO run_leiden (run_id, resolution, n_communities)
            VALUES (?, ?, ?)
            ON CONFLICT (run_id, resolution) DO UPDATE SET
                n_communities = EXCLUDED.n_communities
            """,
            rows,
        )


def fetch_run_leiden(run_id: int) -> pl.DataFrame:
    """Return Leiden sweep results for a run as a Polars DataFrame."""
    return (
        get_connection()
        .execute(
            "SELECT resolution, n_communities FROM run_leiden WHERE run_id = ? ORDER BY resolution",
            [run_id],
        )
        .pl()
    )


def get_all_embeddings(model: str | None = None) -> list[dict]:
    """Get all embeddings with their metadata."""
    rows = (
        get_connection()
        .execute(
            "SELECT id, code, code_type, description, embedding FROM medical_terminologies WHERE model = $model ORDER BY id",
            {"model": model or DEFAULT_MODEL},
        )
        .fetchall()
    )
    return [
        {"id": r[0], "code": r[1], "code_type": r[2], "description": r[3], "embedding": r[4]}
        for r in rows
    ]


def insert(df, model: str | None = None) -> None:
    with _write_context() as con:
        con.register("df_view", df)
        con.execute(
            "INSERT INTO medical_terminologies (code, code_type, description, model, embedding) SELECT code, code_type, description, $model AS model, embedding FROM df_view",
            {"model": model or DEFAULT_MODEL},
        )


def count_terminologies(model_name: str | None = None) -> int:
    if model_name:
        return (
            get_connection()
            .execute(
                "SELECT COUNT(*) FROM medical_terminologies WHERE model = $model",
                {"model": model_name},
            )
            .fetchone()[0]
        )
    return get_connection().execute("SELECT COUNT(*) FROM medical_terminologies").fetchone()[0]


def fetch_terminologies(model_name: str | None = None) -> pl.DataFrame:
    query = "SELECT id, code, description, code_type, code || ':' || code_type AS label FROM medical_terminologies"
    if model_name:
        return (
            get_connection()
            .execute(
                query + " WHERE model = $model ORDER BY id",
                {"model": model_name},
            )
            .pl()
        )
    return get_connection().execute(query + " ORDER BY id").pl()


def insert_clinical_dimensions(model_name: str | None = None) -> None:
    """Compute and insert pairwise cosine similarities for all terminologies."""
    with _write_context() as con:
        con.execute(
            f"""
            INSERT INTO dimensions
            SELECT
                t1.id AS terminology_id_1,
                t2.id AS terminology_id_2,
                array_cosine_similarity(t1.embedding::float[{MODEL_DIMENSIONS}], t2.embedding::float[{MODEL_DIMENSIONS}]) AS value,
                'clinical' AS dimension
            FROM medical_terminologies t1
            JOIN medical_terminologies t2 ON t1.id < t2.id
            WHERE t1.model = $model
            """,
            {"model": model_name or DEFAULT_MODEL},
        )


def read_dimension(dimension: str) -> pl.DataFrame:
    return (
        get_connection()
        .execute(
            "SELECT terminology_id_1, terminology_id_2, value FROM dimensions WHERE dimension = $dimension ORDER BY terminology_id_1",
            {"dimension": dimension},
        )
        .pl()
    )


def dimension_exists(dimension: str) -> bool:
    """Check if a given dimension has any rows in the dimensions table."""
    con = get_connection()
    table_exists = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'dimensions'"
    ).fetchone()
    if not table_exists:
        return False
    return (
        con.execute(
            "SELECT 1 FROM dimensions WHERE dimension = $dimension LIMIT 1",
            {"dimension": dimension},
        ).fetchone()
        is not None
    )


def get_cross_terminology_pairs() -> list[tuple[str, str, str, str, str]]:
    """Return CID10-CIAP2 close pairs present in the DB.

    Each tuple is ``(code_a, type_a, code_b, type_b, ab_code)``, matching
    the format used in validation notebooks. All pairs are semantically
    equivalent by definition (same AB programme, different coding system).
    One CID10 is picked per (ab_code, ciap2_code) to avoid duplicates.
    """
    mapping = load_mapping()

    in_db = set(
        get_connection()
        .execute("SELECT code, code_type FROM medical_terminologies")
        .pl()
        .with_columns(pl.concat_str(["code", "code_type"], separator=":").alias("key"))["key"]
        .to_list()
    )

    candidates = (
        mapping.filter(pl.col("ciap2_code").is_not_null() & pl.col("cid10_code").is_not_null())
        .group_by("ab_code", "ciap2_code")
        .agg(pl.col("cid10_code").first())
    )

    return [
        (row["cid10_code"], "cid10", row["ciap2_code"], "ciap2", row["ab_code"])
        for row in candidates.iter_rows(named=True)
        if f"{row['cid10_code']}:cid10" in in_db and f"{row['ciap2_code']}:ciap2" in in_db
    ]


def _city_metadata(dataset_name: str) -> dict:
    """Extract city, region and population_range from dataset_name.

    Matches by normalizing the city name from IBGE_CITY_MAP against the
    dataset_name string (lowercase, underscored). Returns None values
    when no match is found so the run is still recorded.
    """
    for entry in IBGE_CITY_MAP.values():
        normalised = entry["city"].lower().replace(" ", "_")
        if normalised in dataset_name.lower():
            return {
                "city": entry["city"],
                "region": entry["region"],
                "population_range": entry["population_range"],
            }
    logger.warning(f"Could not resolve city metadata for dataset '{dataset_name}'")
    return {"city": None, "region": None, "population_range": None}


def save_run(
    *,
    dataset_name: str,
    year_min: int,
    year_max: int,
    snf_k: int,
    n_clusters: int,
    fused_labels: np.ndarray,
    common_ids: list[int],
    indicative_enabled: bool = True,
    silhouette_score: float | None = None,
    affinity_zscore: float | None = None,
    notes: str | None = None,
    created_at: str | None = None,
    df: pl.DataFrame | None = None,
    n_codes_clinical: int | None = None,
    n_codes_temporal: int | None = None,
    n_codes_demographic: int | None = None,
    n_missing_temporal: int | None = None,
    n_missing_demographic: int | None = None,
    cluster_largest: int | None = None,
    cluster_median: float | None = None,
    n_singletons: int | None = None,
    pct_singletons: float | None = None,
    multi_k_results: list[dict] | None = None,
) -> int:
    """Persist a complete SNF run to the database.

    Inserts one row in `runs` and one row per code in `cluster_assignments`.
    Optionally persists view coverage stats, cluster distribution stats, and
    multi-K comparison results.

    Returns:
        The run_id of the inserted run.
    """
    meta = _city_metadata(dataset_name)
    n_codes_common = len(common_ids)

    run_id = insert_run(
        dataset_name=dataset_name,
        city=meta["city"],
        region=meta["region"],
        population_range=meta["population_range"],
        year_min=year_min,
        year_max=year_max,
        snf_k=snf_k,
        n_clusters=n_clusters,
        indicative_enabled=indicative_enabled,
        silhouette_score=silhouette_score,
        affinity_zscore=affinity_zscore,
        notes=notes,
        created_at=created_at,
        n_codes_clinical=n_codes_clinical,
        n_codes_temporal=n_codes_temporal,
        n_codes_demographic=n_codes_demographic,
        n_codes_common=n_codes_common,
        n_missing_temporal=n_missing_temporal,
        n_missing_demographic=n_missing_demographic,
        cluster_largest=cluster_largest,
        cluster_median=cluster_median,
        n_singletons=n_singletons,
        pct_singletons=pct_singletons,
    )

    assignments = [
        {"terminology_id": int(tid), "cluster_id": int(label)}
        for tid, label in zip(common_ids, fused_labels)
    ]
    insert_cluster_assignments(run_id, assignments)

    if multi_k_results:
        insert_run_multi_k(run_id, multi_k_results)

    if df is not None:
        runs_dir = DATA_DIR / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = runs_dir / f"{run_id}.parquet"
        df.write_parquet(dataset_path)
        logger.info(f"Dataset saved to {dataset_path}")

    logger.info(f"Run {run_id} saved — {len(assignments)} codes across {n_clusters} clusters.")
    return run_id


def _close_write_connection() -> None:
    global _write_connection
    if _write_connection is not None:
        _write_connection.close()
        _write_connection = None


def _close_read_only_connection() -> None:
    global _read_only_connection
    if _read_only_connection is not None:
        _read_only_connection.close()
        _read_only_connection = None


@contextmanager
def _write_context() -> Iterator[duckdb.DuckDBPyConnection]:
    """Yields the shared write connection for a write block (semantic sugar only)."""
    yield get_connection(write=True)
