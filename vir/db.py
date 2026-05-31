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

from vir.adapters import stratum_metadata
from vir.adapters.datasus.terminologies.terminology_mapping import load_mapping
from vir.config import DATA_DIR, DEFAULT_MODEL, MODEL_DIMENSIONS, logger
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
            ALTER TABLE runs ADD COLUMN IF NOT EXISTS model               VARCHAR;
        """)

        con.execute("UPDATE runs SET adapter = 'datasus' WHERE adapter IS NULL;")
        con.execute(
            "UPDATE runs SET model = $model WHERE model IS NULL;", {"model": DEFAULT_MODEL}
        )


def create_embeddings_table_if_not_exists():
    migrate_decouple_terminology_model_if_needed()
    with _write_context() as con:
        con.execute("""
            CREATE SEQUENCE IF NOT EXISTS id_sequence START 1;
            CREATE TABLE IF NOT EXISTS medical_terminologies(
                id INTEGER PRIMARY KEY DEFAULT nextval('id_sequence'),
                code_type VARCHAR,
                code VARCHAR,
                description VARCHAR,
                UNIQUE(code, code_type)
            );
            CREATE TABLE IF NOT EXISTS terminology_embeddings(
                terminology_id INTEGER NOT NULL REFERENCES medical_terminologies(id),
                model VARCHAR NOT NULL,
                embedding FLOAT[],
                PRIMARY KEY (terminology_id, model)
            );
        """)


def create_dimensions_table_if_not_exists():
    migrate_decouple_terminology_model_if_needed()
    with _write_context() as con:
        con.execute("""
            CREATE SEQUENCE IF NOT EXISTS id_sequence START 1;
            CREATE TABLE IF NOT EXISTS dimensions (
                terminology_id_1 INTEGER NOT NULL REFERENCES medical_terminologies(id),
                terminology_id_2 INTEGER NOT NULL REFERENCES medical_terminologies(id),
                value FLOAT4 NOT NULL,
                dimension VARCHAR NOT NULL,
                model VARCHAR NOT NULL,
                PRIMARY KEY (terminology_id_1, terminology_id_2, dimension, model),
                CHECK (terminology_id_1 < terminology_id_2)
            );
            CREATE TABLE IF NOT EXISTS clinical_coverage (
                model VARCHAR NOT NULL,
                terminology_id INTEGER NOT NULL REFERENCES medical_terminologies(id),
                PRIMARY KEY (model, terminology_id)
            );
        """)


def _table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    return (
        con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [name]
        ).fetchone()
        is not None
    )


def _has_legacy_embeddings_schema(con: duckdb.DuckDBPyConnection) -> bool:
    """True when ``medical_terminologies`` still carries the per-model embedding
    columns (``model``/``embedding``), i.e. the database predates the split of
    terminology identity from embeddings."""
    return (
        con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_name = 'medical_terminologies' AND column_name = 'embedding'
            """
        ).fetchone()[0]
        > 0
    )


def migrate_decouple_terminology_model_if_needed() -> None:
    """Decouple ``terminology_id`` from the embedding model.

    Legacy schema stored one ``medical_terminologies`` row per
    ``(code, code_type, model)`` — so each model produced a *different* id for
    the same code, binding every downstream table (and the processed parquet) to
    a single model. This migration makes the identity canonical per
    ``(code, code_type)`` and moves embeddings into ``terminology_embeddings``
    keyed by ``(terminology_id, model)``.

    Canonical ids are the ids of :data:`DEFAULT_MODEL` (PubMedBERT); every other
    model's rows are remapped onto them by ``(code, code_type)``. Existing
    clinical ``dimensions`` are stamped with ``DEFAULT_MODEL`` and
    ``cluster_assignments`` referencing non-canonical ids are remapped, so no run
    is lost. Idempotent: a no-op once the schema is already decoupled.
    """
    with _write_context() as con:
        terminologies_exist = con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'medical_terminologies'"
        ).fetchone()
        if not terminologies_exist or not _has_legacy_embeddings_schema(con):
            return

        canonical_model = DEFAULT_MODEL
        canonical_count = con.execute(
            "SELECT COUNT(*) FROM medical_terminologies WHERE model = $model",
            {"model": canonical_model},
        ).fetchone()[0]
        if canonical_count == 0:
            raise RuntimeError(
                f"Cannot migrate: canonical model {canonical_model!r} has no rows in "
                "medical_terminologies. Set DEFAULT_MODEL to a model present in the DB."
            )

        orphan_codes = con.execute(
            """
            SELECT COUNT(*)
            FROM medical_terminologies mt
            LEFT JOIN medical_terminologies canon
              ON canon.code = mt.code AND canon.code_type = mt.code_type
             AND canon.model = $model
            WHERE canon.id IS NULL
            """,
            {"model": canonical_model},
        ).fetchone()[0]
        if orphan_codes > 0:
            raise RuntimeError(
                f"Cannot migrate: {orphan_codes} embedding rows have (code, code_type) "
                f"absent from the canonical model {canonical_model!r}; remapping them by "
                "code would drop their embeddings. Resolve before migrating."
            )

        has_dimensions = _table_exists(con, "dimensions")
        has_coverage = _table_exists(con, "clinical_coverage")
        has_assignments = _table_exists(con, "cluster_assignments")

        logger.info("Migrating to decoupled terminology/embedding schema...")
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute(
                """
                CREATE TEMP TABLE _id_map AS
                SELECT mt.id AS old_id, canon.id AS canonical_id
                FROM medical_terminologies mt
                JOIN medical_terminologies canon
                  ON canon.code = mt.code AND canon.code_type = mt.code_type
                 AND canon.model = $model
                """,
                {"model": canonical_model},
            )
            con.execute(
                """
                CREATE TEMP TABLE _emb_stage AS
                SELECT m.canonical_id AS terminology_id, mt.model AS model, mt.embedding AS embedding
                FROM medical_terminologies mt
                JOIN _id_map m ON m.old_id = mt.id
                """
            )
            con.execute(
                """
                CREATE TEMP TABLE _mt_stage AS
                SELECT id, code_type, code, description
                FROM medical_terminologies WHERE model = $model
                """,
                {"model": canonical_model},
            )
            if has_dimensions:
                con.execute(
                    """
                    CREATE TEMP TABLE _dims_stage AS
                    SELECT terminology_id_1, terminology_id_2, value, dimension, $model AS model
                    FROM dimensions
                    """,
                    {"model": canonical_model},
                )
            if has_coverage:
                con.execute(
                    "CREATE TEMP TABLE _cov_stage AS SELECT model, terminology_id FROM clinical_coverage"
                )
            if has_assignments:
                con.execute(
                    """
                    CREATE TEMP TABLE _ca_stage AS
                    SELECT ca.run_id AS run_id, m.canonical_id AS terminology_id, max(ca.cluster_id) AS cluster_id
                    FROM cluster_assignments ca
                    JOIN _id_map m ON m.old_id = ca.terminology_id
                    GROUP BY ca.run_id, m.canonical_id
                    """
                )

            if has_assignments:
                con.execute("DROP TABLE cluster_assignments")
            if has_dimensions:
                con.execute("DROP TABLE dimensions")
            if has_coverage:
                con.execute("DROP TABLE clinical_coverage")
            con.execute("DROP TABLE medical_terminologies")

            con.execute("""
                CREATE TABLE medical_terminologies(
                    id INTEGER PRIMARY KEY DEFAULT nextval('id_sequence'),
                    code_type VARCHAR,
                    code VARCHAR,
                    description VARCHAR,
                    UNIQUE(code, code_type)
                );
                CREATE TABLE terminology_embeddings(
                    terminology_id INTEGER NOT NULL REFERENCES medical_terminologies(id),
                    model VARCHAR NOT NULL,
                    embedding FLOAT[],
                    PRIMARY KEY (terminology_id, model)
                );
            """)
            con.execute(
                "INSERT INTO medical_terminologies (id, code_type, code, description) "
                "SELECT id, code_type, code, description FROM _mt_stage"
            )
            con.execute(
                "INSERT INTO terminology_embeddings (terminology_id, model, embedding) "
                "SELECT terminology_id, model, embedding FROM _emb_stage"
            )

            if has_dimensions:
                con.execute("""
                    CREATE TABLE dimensions (
                        terminology_id_1 INTEGER NOT NULL REFERENCES medical_terminologies(id),
                        terminology_id_2 INTEGER NOT NULL REFERENCES medical_terminologies(id),
                        value FLOAT4 NOT NULL,
                        dimension VARCHAR NOT NULL,
                        model VARCHAR NOT NULL,
                        PRIMARY KEY (terminology_id_1, terminology_id_2, dimension, model),
                        CHECK (terminology_id_1 < terminology_id_2)
                    );
                """)
                con.execute(
                    "INSERT INTO dimensions (terminology_id_1, terminology_id_2, value, dimension, model) "
                    "SELECT terminology_id_1, terminology_id_2, value, dimension, model FROM _dims_stage"
                )
                con.execute("DROP TABLE _dims_stage")
            if has_coverage:
                con.execute("""
                    CREATE TABLE clinical_coverage (
                        model VARCHAR NOT NULL,
                        terminology_id INTEGER NOT NULL REFERENCES medical_terminologies(id),
                        PRIMARY KEY (model, terminology_id)
                    );
                """)
                con.execute(
                    "INSERT INTO clinical_coverage (model, terminology_id) "
                    "SELECT model, terminology_id FROM _cov_stage"
                )
                con.execute("DROP TABLE _cov_stage")
            if has_assignments:
                con.execute("""
                    CREATE TABLE cluster_assignments (
                        run_id          INTEGER NOT NULL REFERENCES runs(run_id),
                        terminology_id  INTEGER NOT NULL REFERENCES medical_terminologies(id),
                        cluster_id      INTEGER NOT NULL,
                        PRIMARY KEY (run_id, terminology_id)
                    );
                """)
                con.execute(
                    "INSERT INTO cluster_assignments (run_id, terminology_id, cluster_id) "
                    "SELECT run_id, terminology_id, cluster_id FROM _ca_stage"
                )
                con.execute("DROP TABLE _ca_stage")

            con.execute("DROP TABLE _id_map; DROP TABLE _emb_stage; DROP TABLE _mt_stage;")
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

        n_canonical = con.execute("SELECT COUNT(*) FROM medical_terminologies").fetchone()[0]
        n_embeddings = con.execute("SELECT COUNT(*) FROM terminology_embeddings").fetchone()[0]
        logger.info(
            f"Migration complete: {n_canonical} canonical terminologies, "
            f"{n_embeddings} embeddings across models."
        )


def insert_run(
    *,
    adapter: str,
    dataset_name: str,
    city: str | None,
    region: str | None,
    population_range: str | None,
    year_min: int,
    year_max: int,
    snf_k: int,
    n_clusters: int,
    model: str | None = None,
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
    """Insert a new run and return its run_id.

    ``adapter`` is required: it pins the row to the adapter that produced it so
    notebooks can filter via :func:`fetch_runs`. Without it, the legacy
    ``UPDATE runs SET adapter = 'datasus' WHERE adapter IS NULL`` migration
    silently retags any new run as ``datasus`` on the next schema-create call.
    """
    create_results_schema()
    _created_at = created_at if created_at is not None else datetime.now().isoformat()
    with _write_context() as con:
        result = con.execute(
            """
            INSERT INTO runs (
                adapter, model, dataset_name, city, region, population_range,
                year_min, year_max, snf_k, n_clusters,
                indicative_enabled, silhouette_score, affinity_zscore, notes, created_at, finished_at,
                n_codes_clinical, n_codes_temporal, n_codes_demographic, n_codes_common,
                n_missing_temporal, n_missing_demographic,
                cluster_largest, cluster_median, n_singletons, pct_singletons
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING run_id
            """,
            [
                adapter,
                model or DEFAULT_MODEL,
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


def fetch_runs(adapter: str | None = None) -> pl.DataFrame:
    """Return runs as a Polars DataFrame, including duration in minutes.

    When ``adapter`` is provided, restrict the result to runs produced by that
    adapter (e.g. ``"datasus"`` or ``"aktin"``). This is the supported way to
    list runs for a single dataset family from notebooks — clients should not
    rely on stratum names to do the filtering.
    """
    base_query = """
        SELECT run_id, dataset_name, adapter, city, region, population_range,
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
        FROM runs
    """
    if adapter is None:
        return get_connection().execute(base_query + " ORDER BY created_at DESC").pl()
    return (
        get_connection()
        .execute(base_query + " WHERE adapter = ? ORDER BY created_at DESC", [adapter])
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
    """Get all embeddings (for the given model) with their terminology metadata."""
    rows = (
        get_connection()
        .execute(
            """
            SELECT mt.id, mt.code, mt.code_type, mt.description, te.embedding
            FROM terminology_embeddings te
            JOIN medical_terminologies mt ON mt.id = te.terminology_id
            WHERE te.model = $model
            ORDER BY mt.id
            """,
            {"model": model or DEFAULT_MODEL},
        )
        .fetchall()
    )
    return [
        {"id": r[0], "code": r[1], "code_type": r[2], "description": r[3], "embedding": r[4]}
        for r in rows
    ]


def insert(df, model: str | None = None) -> None:
    """Persist embeddings for ``model``, decoupling identity from embeddings.

    Canonical terminology identity (``code``, ``code_type``, ``description``) is
    upserted into ``medical_terminologies`` once per ``(code, code_type)``; the
    per-model vectors land in ``terminology_embeddings`` keyed by
    ``(terminology_id, model)``. Embedding dimensionality is validated against
    :data:`MODEL_DIMENSIONS`.
    """
    model_name = model or DEFAULT_MODEL
    with _write_context() as con:
        con.register("df_view", df)
        bad_dim = con.execute(
            "SELECT COUNT(*) FROM df_view WHERE len(embedding) <> $dim",
            {"dim": MODEL_DIMENSIONS},
        ).fetchone()[0]
        if bad_dim > 0:
            raise ValueError(
                f"{bad_dim} embeddings do not have the expected dimensionality "
                f"({MODEL_DIMENSIONS}). Model {model_name!r} is incompatible with the "
                "current MODEL_DIMENSIONS."
            )
        con.execute(
            """
            INSERT INTO medical_terminologies (code, code_type, description)
            SELECT DISTINCT code, code_type, description FROM df_view
            ON CONFLICT (code, code_type) DO NOTHING
            """
        )
        con.execute(
            """
            INSERT INTO terminology_embeddings (terminology_id, model, embedding)
            SELECT mt.id, $model AS model, d.embedding
            FROM df_view d
            JOIN medical_terminologies mt ON mt.code = d.code AND mt.code_type = d.code_type
            ON CONFLICT (terminology_id, model) DO UPDATE SET embedding = excluded.embedding
            """,
            {"model": model_name},
        )


def count_terminologies(model_name: str | None = None) -> int:
    if model_name:
        return (
            get_connection()
            .execute(
                "SELECT COUNT(*) FROM terminology_embeddings WHERE model = $model",
                {"model": model_name},
            )
            .fetchone()[0]
        )
    return get_connection().execute("SELECT COUNT(*) FROM medical_terminologies").fetchone()[0]


def fetch_terminologies(model_name: str | None = None) -> pl.DataFrame:
    """Return canonical terminology labels.

    Identity is model-agnostic since the decoupling, so ``model_name`` is
    accepted for backwards compatibility but no longer filters the rows.
    """
    query = (
        "SELECT id, code, description, code_type, code || ':' || code_type AS label "
        "FROM medical_terminologies ORDER BY id"
    )
    return get_connection().execute(query).pl()


def insert_clinical_dimensions(
    model_name: str | None = None,
    terminology_ids: list[int] | None = None,
) -> None:
    """Compute and persist pairwise cosine similarities incrementally.

    Only pairs whose endpoints are needed for the current run are inserted.
    Coverage is tracked in ``clinical_coverage`` so subsequent runs skip work
    that is already done. A pair (a, b) with a < b is inserted whenever at
    least one of {a, b} is newly covered — pairs between already-covered ids
    are guaranteed to exist by the coverage invariant and are not re-computed.
    """
    model = model_name or DEFAULT_MODEL
    with _write_context() as con:
        con.execute("SET preserve_insertion_order=false")
        if terminology_ids is None:
            requested_ids = (
                con.execute(
                    "SELECT terminology_id FROM terminology_embeddings WHERE model = $model",
                    {"model": model},
                )
                .pl()["terminology_id"]
                .to_list()
            )
        else:
            requested_ids = terminology_ids
        covered_ids = (
            con.execute(
                "SELECT terminology_id FROM clinical_coverage WHERE model = $model",
                {"model": model},
            )
            .pl()["terminology_id"]
            .to_list()
        )
        covered_set = set(covered_ids)
        uncovered_ids = sorted({tid for tid in requested_ids if tid not in covered_set})
        if not uncovered_ids:
            return
        affected_ids = sorted(covered_set.union(uncovered_ids))
        con.execute(
            f"""
            INSERT INTO dimensions
            WITH uncovered(id) AS (SELECT unnest($uncovered::INTEGER[])),
                 affected(id) AS (SELECT unnest($affected::INTEGER[]))
            SELECT
                t1.terminology_id AS terminology_id_1,
                t2.terminology_id AS terminology_id_2,
                array_cosine_similarity(t1.embedding::float[{MODEL_DIMENSIONS}], t2.embedding::float[{MODEL_DIMENSIONS}]) AS value,
                'clinical' AS dimension,
                $model AS model
            FROM terminology_embeddings t1
            JOIN terminology_embeddings t2 ON t1.terminology_id < t2.terminology_id
            WHERE t1.model = $model AND t2.model = $model
              AND t1.terminology_id IN (SELECT id FROM affected)
              AND t2.terminology_id IN (SELECT id FROM affected)
              AND (t1.terminology_id IN (SELECT id FROM uncovered) OR t2.terminology_id IN (SELECT id FROM uncovered))
            """,
            {"model": model, "uncovered": uncovered_ids, "affected": affected_ids},
        )
        con.execute(
            """
            INSERT INTO clinical_coverage(model, terminology_id)
            SELECT $model, unnest($uncovered::INTEGER[])
            """,
            {"model": model, "uncovered": uncovered_ids},
        )


def read_dimension(
    dimension: str,
    terminology_ids: list[int] | None = None,
    model_name: str | None = None,
) -> pl.DataFrame:
    """Read a dimension's pairwise values.

    For the clinical dimension, similarities depend on the embedding model, so
    pass ``model_name`` to select the right rows. When omitted, no model filter
    is applied (back-compatible for single-model reads).
    """
    model_filter = "" if model_name is None else " AND model = $model"
    params: dict = {"dimension": dimension}
    if model_name is not None:
        params["model"] = model_name
    if terminology_ids is None:
        return (
            get_connection()
            .execute(
                "SELECT terminology_id_1, terminology_id_2, value FROM dimensions "
                "WHERE dimension = $dimension" + model_filter + " ORDER BY terminology_id_1",
                params,
            )
            .pl()
        )
    params["ids"] = terminology_ids
    return (
        get_connection()
        .execute(
            """
            WITH wanted(id) AS (SELECT unnest($ids::INTEGER[]))
            SELECT terminology_id_1, terminology_id_2, value
            FROM dimensions
            WHERE dimension = $dimension"""
            + model_filter
            + """
              AND terminology_id_1 IN (SELECT id FROM wanted)
              AND terminology_id_2 IN (SELECT id FROM wanted)
            ORDER BY terminology_id_1
            """,
            params,
        )
        .pl()
    )


def dimension_exists(
    dimension: str,
    model_name: str | None = None,
    terminology_ids: list[int] | None = None,
) -> bool:
    """Check whether a dimension is fully cached for the given inputs.

    For ``dimension == "clinical"`` with a model and terminology_ids, returns
    True only when every requested id is recorded in ``clinical_coverage`` for
    that model — meaning every pair among them already exists in
    ``dimensions``. Falls back to the legacy "any row exists" check otherwise.
    """
    con = get_connection()
    table_exists = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'dimensions'"
    ).fetchone()
    if not table_exists:
        return False
    if dimension == "clinical" and model_name is not None and terminology_ids is not None:
        coverage_table_exists = con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'clinical_coverage'"
        ).fetchone()
        if not coverage_table_exists:
            return False
        missing = con.execute(
            """
            SELECT COUNT(*)
            FROM (SELECT unnest($ids::INTEGER[]) AS id) requested
            ANTI JOIN clinical_coverage c
              ON c.model = $model AND c.terminology_id = requested.id
            """,
            {"model": model_name, "ids": terminology_ids},
        ).fetchone()[0]
        return missing == 0
    if model_name is None:
        return (
            con.execute(
                "SELECT 1 FROM dimensions WHERE dimension = $dimension LIMIT 1",
                {"dimension": dimension},
            ).fetchone()
            is not None
        )
    return (
        con.execute(
            """
            SELECT 1
            FROM dimensions d
            WHERE d.dimension = $dimension AND d.model = $model
            LIMIT 1
            """,
            {"dimension": dimension, "model": model_name},
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


def save_run(
    *,
    adapter: str,
    dataset_name: str,
    year_min: int,
    year_max: int,
    snf_k: int,
    n_clusters: int,
    fused_labels: np.ndarray,
    common_ids: list[int],
    model: str | None = None,
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
    meta = stratum_metadata(dataset_name)
    n_codes_common = len(common_ids)

    run_id = insert_run(
        adapter=adapter,
        dataset_name=dataset_name,
        city=meta["city"],
        region=meta["region"],
        population_range=meta["population_range"],
        year_min=year_min,
        year_max=year_max,
        snf_k=snf_k,
        n_clusters=n_clusters,
        model=model,
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

    if df is None:
        logger.warning(
            f"Run {run_id} has no per-record parquet: ``df`` was None when ``save_run`` "
            f"was called. The cluster_assignments rows exist in the DB but the viz "
            f"notebooks won't be able to render this run."
        )
    elif df.is_empty():
        logger.warning(
            f"Run {run_id} has no per-record parquet: ``df`` was empty when ``save_run`` "
            f"was called. Likely cause: year filter + ``filter_min_temporal_coverage`` "
            f"removed every row before SNF. Delete the orphan run with "
            f"``vir results delete --id {run_id}`` and re-run with a wider year range."
        )
    else:
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
