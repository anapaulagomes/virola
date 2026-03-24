from datetime import datetime

import duckdb
import polars as pl

from vir.config import DATA_DIR, logger
from vir.helpers import get_n_jobs

PROJECT_DB = DATA_DIR / "virola.db"
DEFAULT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"  # TODO move to config
MODEL_DIMENSIONS = 768


def create_results_schema():
    """Create runs, cluster_assignments and cluster_profiles tables if they don't exist."""
    con = get_connection(writeable=True)
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
            PRIMARY KEY (run_id, k)
        );
    """)

    # Migrate existing runs table — add new columns if they don't exist yet
    for col_def in [
        ("n_codes_clinical", "INTEGER"),
        ("n_codes_temporal", "INTEGER"),
        ("n_codes_demographic", "INTEGER"),
        ("n_codes_common", "INTEGER"),
        ("n_missing_temporal", "INTEGER"),
        ("n_missing_demographic", "INTEGER"),
        ("cluster_largest", "INTEGER"),
        ("cluster_median", "FLOAT"),
        ("n_singletons", "INTEGER"),
        ("pct_singletons", "FLOAT"),
    ]:
        try:
            con.execute(f"ALTER TABLE runs ADD COLUMN IF NOT EXISTS {col_def[0]} {col_def[1]}")
        except Exception as e:
            logger.warning(f"Migration skipped for column {col_def[0]}: {e}")


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
    con = get_connection(writeable=True)
    _created_at = created_at if created_at is not None else datetime.now().isoformat()
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
    con = get_connection(writeable=True)
    con.execute(
        "UPDATE runs SET finished_at = now() WHERE run_id = ?",
        [run_id],
    )


def insert_cluster_assignments(run_id: int, assignments: list[dict]):
    """Bulk insert cluster assignments for a run.

    Args:
        run_id: The run identifier.
        assignments: List of {terminology_id, cluster_id} dicts.
    """

    con = get_connection(writeable=True)
    df = pl.DataFrame(assignments).with_columns(pl.lit(run_id).alias("run_id"))
    con.register("assignments_view", df)
    con.execute("""
        INSERT INTO cluster_assignments (run_id, terminology_id, cluster_id)
        SELECT run_id, terminology_id, cluster_id FROM assignments_view
    """)


def fetch_runs() -> "pl.DataFrame":
    """Return all runs as a Polars DataFrame, including duration in minutes."""
    con = get_connection()
    return con.execute("""
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
    """).pl()


def insert_run_multi_k(run_id: int, results: list[dict]) -> None:
    """Bulk insert multi-K comparison results for a run.

    Args:
        run_id: The run identifier.
        results: List of {n_clusters, silhouette, largest_cluster, median_size} dicts.
    """
    con = get_connection(writeable=True)
    rows = [
        (run_id, r["n_clusters"], r["silhouette"], r["largest_cluster"], r["median_size"])
        for r in results
    ]
    con.executemany(
        """
        INSERT INTO run_multi_k (run_id, k, silhouette, largest_cluster, median_size)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (run_id, k) DO UPDATE SET
            silhouette = EXCLUDED.silhouette,
            largest_cluster = EXCLUDED.largest_cluster,
            median_size = EXCLUDED.median_size
        """,
        rows,
    )


def fetch_run_multi_k(run_id: int) -> "pl.DataFrame":
    """Return multi-K comparison results for a run as a Polars DataFrame."""
    con = get_connection()
    return con.execute(
        "SELECT k, silhouette, largest_cluster, median_size FROM run_multi_k WHERE run_id = ? ORDER BY k",
        [run_id],
    ).pl()


def fetch_cluster_assignments(run_id: int) -> "pl.DataFrame":
    """Return all assignments for a run joined with terminology metadata."""
    con = get_connection()
    return con.execute(
        """
        SELECT
            ca.cluster_id,
            ca.terminology_id,
            mt.code,
            mt.code_type,
            mt.description
        FROM cluster_assignments ca
        JOIN medical_terminologies mt ON ca.terminology_id = mt.id
        WHERE ca.run_id = ?
        ORDER BY ca.cluster_id, mt.code
    """,
        [run_id],
    ).pl()


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
):
    """Insert or update a cluster profile (subset of fields can be provided)."""
    create_results_schema()
    con = get_connection(writeable=True)
    con.execute(
        """
        INSERT INTO cluster_profiles (
            run_id, cluster_id, n_codes, total_cases,
            top_codes, temporal_signature, demographic_profile, explanation,
            umap_x, umap_y,
            llm_short_description, llm_description, llm_abbreviation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (run_id, cluster_id) DO UPDATE SET
            n_codes                = COALESCE(EXCLUDED.n_codes, cluster_profiles.n_codes),
            total_cases            = COALESCE(EXCLUDED.total_cases, cluster_profiles.total_cases),
            top_codes              = COALESCE(EXCLUDED.top_codes, cluster_profiles.top_codes),
            temporal_signature     = COALESCE(EXCLUDED.temporal_signature, cluster_profiles.temporal_signature),
            demographic_profile    = COALESCE(EXCLUDED.demographic_profile, cluster_profiles.demographic_profile),
            explanation            = COALESCE(EXCLUDED.explanation, cluster_profiles.explanation),
            umap_x                 = COALESCE(EXCLUDED.umap_x, cluster_profiles.umap_x),
            umap_y                 = COALESCE(EXCLUDED.umap_y, cluster_profiles.umap_y),
            llm_short_description  = COALESCE(EXCLUDED.llm_short_description, cluster_profiles.llm_short_description),
            llm_description        = COALESCE(EXCLUDED.llm_description, cluster_profiles.llm_description),
            llm_abbreviation       = COALESCE(EXCLUDED.llm_abbreviation, cluster_profiles.llm_abbreviation)
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


def fetch_cluster_profiles(run_id: int) -> "pl.DataFrame":
    """Return all profiles for a run."""
    con = get_connection()
    return con.execute(
        """
        SELECT * FROM cluster_profiles WHERE run_id = ? ORDER BY cluster_id
    """,
        [run_id],
    ).pl()


def get_connection(writeable=False):
    config = {"threads": get_n_jobs()}
    return duckdb.connect(PROJECT_DB, read_only=not writeable, config=config)


def create_embeddings_table_if_not_exists():
    con = get_connection(writeable=True)
    con.execute(
        """
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
        """
    )


def create_dimensions_table_if_not_exists():
    con = get_connection(writeable=True)
    con.execute(
        """
        CREATE SEQUENCE IF NOT EXISTS id_sequence START 1;
        CREATE TABLE IF NOT EXISTS dimensions (
            terminology_id_1 INTEGER NOT NULL REFERENCES medical_terminologies(id),
            terminology_id_2 INTEGER NOT NULL REFERENCES medical_terminologies(id),
            value FLOAT4 NOT NULL,
            dimension VARCHAR NOT NULL,
            PRIMARY KEY (terminology_id_1, terminology_id_2, dimension),
            CHECK (terminology_id_1 < terminology_id_2)
        );
        """
    )


def get_embedding(code, code_type, model=None):
    model_name = model or DEFAULT_MODEL
    con = get_connection()
    result = con.execute(
        """
        SELECT
        embedding
        FROM medical_terminologies WHERE model = $model AND code = $code AND code_type = $code_type;
        """,
        {"model": model_name, "code": code, "code_type": code_type},
    ).fetchone()
    if result:
        return result[0]
    return


def insert(df, model=None):
    con = get_connection(writeable=True)
    con.register("df_view", df)
    con.execute(
        f"""
        INSERT INTO medical_terminologies (code, code_type, description, model, embedding)
        SELECT code, code_type, description, '{model or DEFAULT_MODEL}' AS model, embedding FROM df_view
        """
    )


def get_all_embeddings(model=None):
    """Get all embeddings with their metadata for visualization."""
    model_name = model or DEFAULT_MODEL
    con = get_connection()
    result = con.execute(
        """
        SELECT
        id,
        code,
        code_type,
        description,
        embedding
        FROM medical_terminologies
        WHERE model = $model
        ORDER BY id;
        """,
        {"model": model_name},
    ).fetchall()

    if result:
        return [
            {
                "id": row[0],
                "code": row[1],
                "code_type": row[2],
                "description": row[3],
                "embedding": row[4],
            }
            for row in result
        ]
    return []


def insert_or_update_dimension(df):
    """
    Inserts dimension data from a Polars DataFrame into the dimensions table.
    """
    con = get_connection(writeable=True)
    con.register("df_view", df)
    con.execute(
        """
        INSERT INTO dimensions (terminology_id_1, terminology_id_2, value, dimension)
        SELECT terminology_id_1, terminology_id_2, value, dimension FROM df_view
        """
    )


def insert_clinical_dimensions(model_name=None):
    """Inserts the clinical dimensions into the database.

    Reads all terminologies and calculate the cosine similarity of its embeddings."""
    con = get_connection(writeable=True)
    con.execute(
        """
        INSERT INTO dimensions
        SELECT
            t1.id AS terminology_id_1,
            t2.id AS terminology_id_2,
            array_cosine_similarity(t1.embedding::float[768], t2.embedding::float[768]) AS value,
            'clinical' AS dimension
        FROM medical_terminologies t1
        JOIN medical_terminologies t2 ON t1.id < t2.id
        WHERE t1.model = $model;
        """,
        {"model": model_name or DEFAULT_MODEL},
    )


def read_dimension(dimension):
    con = get_connection()
    result = con.execute(
        """
        SELECT terminology_id_1, terminology_id_2, value
        FROM dimensions
        WHERE dimension = $dimension
        ORDER BY terminology_id_1 ASC
        """,
        {"dimension": dimension},
    )
    return result.pl()


def count_terminologies(model_name=None):
    con = get_connection()
    where_clause = ""
    if model_name:
        where_clause = f"WHERE model = {model_name}"
    result = con.execute(
        f"""
        SELECT COUNT(*)
        FROM medical_terminologies {where_clause}
        """
    ).fetchone()
    return result


def fetch_terminologies(model_name=None):
    con = get_connection()
    where_clause = ""
    if model_name:
        where_clause = f"WHERE model = '{model_name}'"
    result = con.execute(
        f"""
        SELECT id, code, description, code_type, code || ':' || code_type AS label
        FROM medical_terminologies {where_clause}
        ORDER BY id
        """,
    ).pl()
    return result


def get_cross_terminology_pairs() -> list[tuple[str, str, str, str, str]]:
    """Return CID10↔CIAP2 close pairs present in the DB.

    Each tuple is ``(code_a, type_a, code_b, type_b, ab_code)``, matching
    the format used in validation notebooks. All pairs are semantically
    equivalent by definition (same AB programme, different coding system).
    One CID10 is picked per (ab_code, ciap2_code) to avoid duplicates.
    """
    from vir.terminology_mapping import load_mapping

    mapping = load_mapping()
    con = get_connection()

    in_db = set(
        con.execute("SELECT code, code_type FROM medical_terminologies")
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


def dimension_exists(dimension: str) -> bool:
    """Checks if a given dimension exists in the dimensions table."""
    con = get_connection()
    try:
        result = con.execute(
            "SELECT 1 FROM dimensions WHERE dimension = ? LIMIT 1", [dimension]
        ).fetchone()
    except Exception as e:
        if "Catalog Error" in e.args[0]:
            return False
    return result is not None
