import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import torch
    import polars as pl
    from transformers import AutoTokenizer, AutoModel  
    from sklearn.manifold import TSNE
    import plotly.express as px
    return AutoModel, AutoTokenizer, TSNE, mo, np, pl, px, torch


@app.cell
def _(mo):
    mo.md(
        r"""
    # Embedding medical terminologies

    Model: [SapBERT](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext) - clinical and multilingual
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Medical terminologies: CID-10, CIAP-2, ABP

    * `abp_ciap2.csv`

    ```
    CIAP2 relacionada,Descrição AB,Código AB
    R96,Asma,ABP009
    T91,Beribéri,ABP008
    ```

    * `ciap2-cid10.csv`

    ```
    Código da CIAP2,Descrição da CIAP2,Código da CID10,Descrição da CID10
    A01,DOR GENERALIZADA /MÚLTIPLA,R52,DOR NÃO CLASSIFICADA EM OUTRA PARTE
    A02,ARREPIOS/ CALAFRIOS,R688,OUTROS SINTOMAS E SINAIS GERAIS ESPECIFICADOS
    A03,FEBRE,R50,FEBRE DE ORIGEM DESCONHECIDA E DE OUTRAS ORIGENS
    ```

    * `references/RepositorioTerminologia_202506/tb_cid.csv`

    ```
    "Coluna";"Tamanho";"Index";"Tipo"
    "CO_CID";"4";"0";"VARCHAR"
    "NO_CID";"100";"1";"VARCHAR"
    "ST_AGRAVO_NOTIFICACAO";"1";"2";"VARCHAR"
    "CO_SEXO";"1";"3";"VARCHAR"
    "ST_ESTADIO";"1";"4";"VARCHAR"
    "CO_IRRADIADOS";"3";"5";"NUMBER"
    "DT_COMPETENCIA";"6";"6";"VARCHAR"
    "CO_VERSAO";"8";"7";"VARCHAR"
    "ST_ACAO";"1";"8";"NUMBER"
    ```

    ```
    "A00";"C�lera";"1";"I";"0";"0";"202506";"04/2020D";"2"
    "A000";"C�lera devida a Vibrio Cholerae 01, bi�tipo Cholerae";"1";"I";"0";"0";"202506";"01/2017A";"2"
    "A001";"C�lera devida a Vibrio Cholerae 01, bi�tipo El Tor";"1";"I";"0";"0";"202506";"01/2017A";"2"
    "A009";"C�lera n�o especificada";"1";"I";"0";"0";"202506";"01/2017A";"2"
    "A01";"Febres tif�ide e paratif�ide";"0";"I";"0";"0";"202506";"01/2017A";"2"
    ```
    """
    )
    return


@app.cell
def _(pl):
    _exclude_columns = [
        "ST_AGRAVO_NOTIFICACAO",
        "CO_SEXO",
        "ST_ESTADIO",
        "CO_IRRADIADOS",
        "DT_COMPETENCIA",
        "CO_VERSAO",
        "ST_ACAO"
    ]
    cid10 = pl.read_csv(
        "references/RepositorioTerminologia_202506/tb_cid.csv",
        encoding="iso-8859-1",
        separator=";",
        has_header=False,
        new_columns=[
            "code",
            "description",
            "ST_AGRAVO_NOTIFICACAO",
            "CO_SEXO",
            "ST_ESTADIO",
            "CO_IRRADIADOS",
            "DT_COMPETENCIA",
            "CO_VERSAO",
            "ST_ACAO",
        ]
    )
    cid10 = cid10.drop(_exclude_columns)
    cid10 = cid10.with_columns(code_type=pl.lit("cid10"))
    cid10
    return (cid10,)


@app.cell
def _(pl):
    ciap2_cid10 = pl.read_csv("references/ciap2-cid10.csv")
    ciap2_cid10 = ciap2_cid10.rename({
        "Código da CIAP2": "code",
        "Código da CID10": "related_cid10",
        "Descrição da CIAP2": "description",
        "Descrição da CID10": "related_description_cid10",
    })
    ciap2_cid10 = ciap2_cid10.with_columns(code_type=pl.lit("ciap2"))
    abp_ciap2 = pl.read_csv("references/abp_ciap2.csv")
    abp_ciap2 = abp_ciap2.rename({
        "CIAP2 relacionada": "related_ciap2",
        "Código AB": "code",
        "Descrição AB": "description",
    })
    abp_ciap2 = abp_ciap2.with_columns(code_type=pl.lit("ab"))
    return abp_ciap2, ciap2_cid10


@app.cell
def _(abp_ciap2, ciap2_cid10):
    ciap2_cid10.columns, abp_ciap2.columns
    return


@app.cell
def _(abp_ciap2, ciap2_cid10, cid10, pl):
    codes = pl.concat([
        cid10,
        ciap2_cid10[["code", "description", "code_type"]],
        abp_ciap2[["code", "description", "code_type"]]
    ])

    codes
    return (codes,)


@app.cell
def _(mo):
    mo.md(r"""## Embeddings""")
    return


@app.cell
def _(AutoModel, AutoTokenizer, torch):
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    device = torch.device("cpu")
    model = model.to(device)
    return device, model, tokenizer


@app.cell
def _(codes, device, model, np, tokenizer):
    batch_size = 128  # TODO check
    all_embeddings = []

    for i in np.arange(0, len(codes["description"]), batch_size):
        toks = tokenizer.batch_encode_plus(
            codes["description"][i:i+batch_size].to_list(),
            padding="max_length",
            max_length=25, 
            truncation=True,
            return_tensors="pt"
        )
        toks_cpu = {k: v.to(device) for k, v in toks.items()}
        cls_rep = model(**toks_cpu)[0][:, 0, :]  # CLS token representation
        all_embeddings.append(cls_rep.cpu().detach().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return (all_embeddings,)


@app.cell
def _(all_embeddings):
    all_embeddings
    return


@app.cell
def _(all_embeddings):
    len(all_embeddings)
    return


@app.cell
def _(TSNE, all_embeddings):
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(all_embeddings)
    X_embedded
    return (X_embedded,)


@app.cell
def _(mo):
    mo.md(r"""## Viz""")
    return


@app.cell
def _(X_embedded, codes, pl):
    codes_tsne = pl.concat([codes, pl.DataFrame(X_embedded, schema=["c0", "c1"])], how="horizontal")
    codes_tsne
    return (codes_tsne,)


@app.cell
def _(codes_tsne, px):
    _fig = px.scatter(
        codes_tsne,
        x='c0',
        y='c1',
        color="code_type",
        hover_data={"code": True, "description": True}
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Validation""")
    return


@app.cell
def _(all_embeddings):
    all_embeddings
    return


@app.cell
def _(all_embeddings, codes, pl):
    codes_embeddings = pl.concat([codes, pl.DataFrame(all_embeddings, schema=["embeddings"])], how="horizontal")
    codes_embeddings
    return (codes_embeddings,)


@app.cell
def _():
    # codes_embeddings.write_parquet("data/processed/code_type_embeddings.parquet")
    return


@app.cell
def _(np):
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    return (cosine_sim,)


@app.cell
def _(ciap2_cid10):
    ciap2_cid10
    return


@app.cell
def _(ciap2_cid10, codes_embeddings, cosine_sim, pl):
    results = {}

    print("**ICD10 and CIAP2**")
    for row in ciap2_cid10.iter_rows(named=True):
        _left = codes_embeddings.filter(pl.col("code").eq(row["related_cid10"]), pl.col("code_type").eq("cid10"))
        _right = codes_embeddings.filter(pl.col("code").eq(row["code"]), pl.col("code_type").eq("ciap2"))

        results[f"{row['related_cid10']}_{row['code']}"] = cosine_sim(_left["embeddings"].first(), _right["embeddings"].first())

        print(
            f"{row['related_cid10']}_{row['code']}",
            results[f"{row['related_cid10']}_{row['code']}"],
            f"{_left['description'].first()} --- {_right['description'].first()}"
        )

    return (results,)


@app.cell
def _(px, results):
    px.histogram(list(results.values()))
    return


@app.cell
def _(np, results):
    np.percentile(list(results.values()), 50), np.percentile(list(results.values()), 75), np.percentile(list(results.values()), 90)
    return


if __name__ == "__main__":
    app.run()
