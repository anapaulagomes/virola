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

    return (
        LogisticRegression,
        SentenceTransformer,
        classification_report,
        np,
        pl,
    )


@app.cell
def _(np):
    np.random.seed(42)
    label = "indicative"
    n_samples = 30
    return label, n_samples


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
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings, model


@app.cell
def _(embeddings, labels, n_samples, np):
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    pos_sample = np.random.choice(pos_idx, size=n_samples, replace=False)
    neg_sample = np.random.choice(neg_idx, size=n_samples, replace=False)
    train_idx = np.concatenate([pos_sample, neg_sample])

    X_train, y_train = embeddings[train_idx], labels[train_idx]
    X_test, y_test = embeddings, labels  # avaliar no dataset inteiro

    return X_test, X_train, y_test, y_train


@app.cell
def _(mo):
    mo.md(r"""## Train""")
    return


@app.cell
def _(LogisticRegression, X_train, y_train):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return (clf,)


@app.cell
def _(mo):
    mo.md(r"""## Evaluate""")
    return


@app.cell
def _(X_test, classification_report, clf, y_test):
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    return


@app.cell
def _(clf, model):
    novos_textos = [
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
    ]

    novos_embeddings = model.encode(novos_textos, convert_to_numpy=True)
    preds = clf.predict(novos_embeddings)

    for txt, pred in zip(novos_textos, preds):
        print(txt, "→", "🚫" if pred == 1 else "✅")
    return


if __name__ == "__main__":
    app.run()
