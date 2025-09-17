import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import polars as pl
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import StratifiedKFold

    return (
        LogisticRegression,
        SentenceTransformer,
        StratifiedKFold,
        classification_report,
        np,
        pl,
    )


@app.cell
def _(np):
    np.random.seed(42)
    label = "indicative"
    return (label,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Indicative codes for epidemiological surveillance 😷

    **Target codes**: preventive, screening, routine, administrative, or related to contact with the healthcare system.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Training""")
    return


@app.cell
def _(pl):
    codes = pl.read_csv("data/processed/cid10_ciap2_ab.csv")
    return (codes,)


@app.cell
def _():
    # marcar: exame, exames, preventiva, prescrição
    return


@app.cell
def _(codes, pl):
    df_words = codes.with_columns(
        pl.col("description")
        .str.replace_all(r"[^a-zA-ZáéíóúãõâêôçÁÉÍÓÚÃÕÂÊÔÇ]", " ")
        .str.split(" ")
        .alias("words")
    ).explode("words")

    df_words = df_words.with_columns(pl.col("words").str.strip_chars().str.to_lowercase())

    df_result = (
        df_words.filter(pl.col("words").ne(""))
        .group_by("words")
        .agg(
            [
                pl.col("description")
                .unique()
                .sort()
                .top_k(3)
                .str.concat("\n")
                .alias("descriptions"),
                pl.col("description").n_unique().alias("count"),  # quantas descrições únicas
            ]
        )
        .sort("count", descending=True)
    )
    df_result
    return


@app.cell
def _(mo):
    mo.md(r"""## Data""")
    return


@app.cell
def _(label, pl):
    df = pl.read_csv("data/Garbage code - AESOP 20 municípios - 75_ - por quantidade.csv")
    df = df.with_columns(
        [(df[label].fill_null("").str.strip_chars().str.to_lowercase() == "x").cast(pl.Int8)]
    )
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(mo):
    mo.md(r"""## Embeddings""")
    return


@app.cell
def _(df, label):
    texts = df["description"].to_list()
    labels = df[label].to_numpy()
    return labels, texts


@app.cell
def _(SentenceTransformer, texts):
    model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings, model


@app.cell
def _(StratifiedKFold):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return (kf,)


@app.cell
def _(mo):
    mo.md(r"""## Train & Evaluate""")
    return


@app.cell
def _(LogisticRegression, classification_report, embeddings, kf, labels):
    THRESHOLD = 0.7

    all_reports = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(embeddings, labels), 1):
        print(f"\n===== Fold {fold} =====")

        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_train, y_train)

        y_probs = clf.predict_proba(X_test)[:, 1]

        # convert to binary
        y_pred = (y_probs >= THRESHOLD).astype(int)

        # fold report
        report = classification_report(y_test, y_pred, digits=3, output_dict=True)
        all_reports.append(report)

        print(classification_report(y_test, y_pred, digits=3))
    return THRESHOLD, clf


@app.cell
def _(THRESHOLD, clf, model):
    def predict(text):
        _embeddings = model.encode(text, convert_to_numpy=True)
        _probs = clf.predict_proba(_embeddings)[:, 1]
        return _probs

    def classify(text):
        _probs = predict(text)
        _preds = (_probs >= THRESHOLD).astype(int)
        for _txt, _prob, _pred in zip(text, _probs, _preds):
            print(f"{_txt} → prob={_prob:.2f} → {'🚫' if _pred == 1 else '✅'}")

    return (classify,)


@app.cell
def _():
    some_texts = [
        "exame médico geral",
        "hipertensão essencial (primária)",
        "rastreamento de câncer de mama",
        "tosse",
        "diabetes não insulino-dependente",
        "obesidade",
        "gripe",
        "influenza (gripe) devida a vírus não identificado",
        "outros exames gerais",
        "exame ginecológico (geral) (de rotina)",
        "outros disp. assoc. incid. adversos e ñ espec. - dispos. p/fins terapêut. e aparelh. de reabilitação",
        "outros disp. assoc. incid. adversos e ñ espec. - dispos. p/fins diagnóst. ou de monitorização",
        "outros disp. assoc. incid. adversos e ñ espec. - dispos. diver., incl. combin. ñ class. noutra parte",
        "suporte familiar inadequado",
        "problema por doença familiar",
        "problema comportamental de familiar",
        "seqüelas de intoxicação por drogas, medicamentos e substâncias biológicas",
        "supervisão do uso de medicamentos anticoncepcionais",
        "outras drogas medicamentos e substancias biológicas e as não especificadas intenção não determinada",
    ]
    return (some_texts,)


@app.cell
def _(mo, some_texts):
    descriptions = mo.ui.text_area(
        label="Descriptions", value="\n".join(some_texts), full_width=True
    )
    descriptions
    return (descriptions,)


@app.cell
def _(classify, descriptions):
    classify(descriptions.value.splitlines())
    return


if __name__ == "__main__":
    app.run()
