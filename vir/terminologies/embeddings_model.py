import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import torch
    from transformers import AutoModel, AutoTokenizer

    from vir.config import PROCESSED_DATA_DIR
    from vir.db import (
        create_embeddings_table_if_not_exists,
        get_connection,
        get_embedding,
        insert,
    )
    from vir.helpers import get_or_download_model

    return (
        AutoModel,
        AutoTokenizer,
        get_connection,
        get_embedding,
        get_or_download_model,
        mo,
        np,
        pl,
        torch,
        PROCESSED_DATA_DIR,
        create_embeddings_table_if_not_exists,
        insert,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # Embeddings model

    TODO

    - [ ] Evaluate the length of the sentences to define max_length (tokens)
    - [ ] Check ABP008 (duplicated)
    """
    )
    return


@app.cell
def _(pl, PROCESSED_DATA_DIR):
    codes = pl.read_csv(f"{PROCESSED_DATA_DIR}/cid10_ciap2_ab.csv")
    codes
    return (codes,)


@app.cell
def _(torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    return batch_size, device


@app.cell
def _():
    model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    return (model_name,)


@app.cell
def _(
    AutoModel,
    AutoTokenizer,
    device,
    get_or_download_model,
    model_name,
    os,
    torch,
):
    tokenizer = AutoTokenizer.from_pretrained(get_or_download_model(model_name))
    model = AutoModel.from_pretrained(get_or_download_model(model_name))
    model = model.to(device)
    model.eval()  # inference only
    if device == "cpu":
        torch.set_num_threads(max(os.cpu_count() or 2, 2))
    embeddings_dimension = model.config.hidden_size
    return embeddings_dimension, model, tokenizer


@app.cell
def _(embeddings_dimension):
    embeddings_dimension
    return


@app.cell
def _(
    batch_size,
    device,
    embeddings_dimension,
    model,
    np,
    out_dtype,
    tokenizer,
    torch,
):
    @torch.inference_mode()
    def encode_batch(sentences):
        out = []
        for i in range(0, len(sentences), batch_size):
            chunk = sentences[i : i + batch_size].to_list()
            toks = tokenizer.batch_encode_plus(
                chunk, padding="max_length", max_length=25, truncation=True, return_tensors="pt"
            )
            toks = {k: v.to(device) for k, v in toks.items()}
            last_hidden = model(**toks).last_hidden_state
            cls = last_hidden[:, 0, :].to("cpu").numpy().astype(np.float32, copy=False)

            # L2 normalize
            norms = np.linalg.norm(cls, axis=1, keepdims=True) + 1e-12
            cls = cls / norms

            out.append(cls)
        return (
            np.concatenate(out, axis=0)
            if out
            else np.empty((0, embeddings_dimension), dtype=out_dtype)
        )

    return (encode_batch,)


@app.cell
def _(codes, encode_batch):
    all_embeddings = encode_batch(codes["description"])
    return (all_embeddings,)


@app.cell
def _(all_embeddings, codes, pl, PROCESSED_DATA_DIR, model_name):
    embeddings_df = pl.concat(
        [codes, pl.DataFrame(all_embeddings, schema=["embeddings"])], how="horizontal"
    )
    embeddings_df.write_parquet(
        f"{PROCESSED_DATA_DIR}/bra_medter_{model_name.replace('/', '__')}.parquet"
    )
    all_embeddings
    return


@app.cell
def _(mo):
    mo.md(r"""## DB""")
    return


@app.cell
def _(embeddings_df, create_embeddings_table_if_not_exists, insert):
    # un-comment to store in the db
    create_embeddings_table_if_not_exists()
    insert(embeddings_df)
    return


@app.cell
def _(get_embedding):
    # E511,beribéri,cid10
    # ABP008,beribéri,ab
    # ABP011,tabagismo,ab
    # F17,transtornos mentais e comportamentais devidos ao uso de fumo,cid10

    get_embedding("ABP011", "ab")  # using code and code type
    return


@app.cell
def _(get_connection):
    con = get_connection()
    return (con,)


@app.cell
def _(con, medical_terminologies, mo):
    _df = mo.sql(
        """
        SELECT
        model, embedding
        FROM medical_terminologies WHERE model = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext' AND code = 'E511' AND code_type = 'cid10';
        """,
        engine=con,
    )
    return


if __name__ == "__main__":
    app.run()
