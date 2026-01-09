import duckdb

from vir.config import DATA_DIR

PROJECT_DB = DATA_DIR / "virola.db"
DEFAULT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
MODEL_DIMENSIONS = 768


def get_connection(writeable=False):
    return duckdb.connect(PROJECT_DB, read_only=not writeable)


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
