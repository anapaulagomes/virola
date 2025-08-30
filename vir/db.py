import duckdb

EMBEDDINGS_DB = "bra_medter_embeddings.db"
DEFAULT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
MODEL_DIMENSIONS = 768


def get_connection(writeable=False):
    return duckdb.connect(EMBEDDINGS_DB, read_only=not writeable)


def create_embeddings_table_if_not_exists():
    con = get_connection(writeable=True)
    con.from_query(
        """
        CREATE SEQUENCE IF NOT EXISTS id_sequence START 1;
        CREATE TABLE IF NOT EXISTS medical_terminologies(
             id INTEGER DEFAULT nextval('id_sequence'),
             code VARCHAR,
             code_type VARCHAR,
             description VARCHAR,
             model VARCHAR,
             embedding FLOAT[] NOT NULL,
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
    con.execute(f"""
        INSERT INTO medical_terminologies (code, code_type, description, model, embedding)
        SELECT code, code_type, description, '{model or DEFAULT_MODEL}' AS model, embeddings FROM df_view
    """)


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
