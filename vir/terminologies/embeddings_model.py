import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import marimo as mo
    import numpy as np
    import polars as pl
    import torch
    from transformers import AutoModel, AutoTokenizer

    from vir.config import PROCESSED_DATA_DIR
    from vir.db import DEFAULT_MODEL, create_embeddings_table_if_not_exists, insert
    from vir.helpers import get_or_download_model

    return (
        AutoModel,
        AutoTokenizer,
        DEFAULT_MODEL,
        PROCESSED_DATA_DIR,
        create_embeddings_table_if_not_exists,
        get_or_download_model,
        insert,
        mo,
        np,
        os,
        pl,
        torch,
    )


@app.cell
def _(mo):
    args = mo.cli_args() if mo.app_meta().mode == "script" else {}
    return (args,)


@app.cell
def _(DEFAULT_MODEL, args):
    model_name = args.get("model", DEFAULT_MODEL)
    return (model_name,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Embeddings model

    TODO

    - [ ] Evaluate the length of the sentences to define max_length (tokens)

    Notes:

    - ABP008 is duplicated (confirmed in official source: https://integracao.esusab.ufsc.br/ledi/documentacao/estrutura_arquivos/dicionario-fai.html#listaciapcondicaoavaliada). To circunvent this, suffix with numbers was added to it. ABP008-1, ABP008-2.
    """
    )
    return


@app.cell
def _(PROCESSED_DATA_DIR, pl):
    codes = pl.read_csv(f"{PROCESSED_DATA_DIR}/cid10_ciap2_ab.csv")
    codes
    return (codes,)


@app.cell
def _(torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    return batch_size, device


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
        torch.set_num_threads(max(os.cpu_count() or os.getenv("NUMBER_OF_THREADS"), 2))
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
def _(PROCESSED_DATA_DIR, all_embeddings, codes, model_name, pl):
    embeddings_df = pl.concat(
        [codes, pl.DataFrame(all_embeddings, schema=["embedding"])], how="horizontal"
    )
    embeddings_df.write_parquet(
        f"{PROCESSED_DATA_DIR}/bra_medter_{model_name.replace('/', '__')}.parquet"
    )
    all_embeddings
    return (embeddings_df,)


@app.cell
def _(mo):
    mo.md(r"""## DB""")
    return


@app.cell
def _(embeddings_df):
    embeddings_df
    return


@app.cell
def _(embeddings_df, pl):
    (embeddings_df.group_by(["code", "code_type"]).len().filter(pl.col("len") > 1))
    return


@app.cell
def _(
    create_embeddings_table_if_not_exists,
    embeddings_df,
    insert,
    model_name,
):
    # un-comment to store in the db
    create_embeddings_table_if_not_exists()
    insert(embeddings_df, model_name)
    return


@app.cell
def _(mo):
    if mo.app_meta().mode == "script":
        print("✓ Embeddings database created successfully")
    return


if __name__ == "__main__":
    app.run()
