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
    import plotly.express as px
    import polars as pl
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, precision_recall_curve
    from sklearn.model_selection import StratifiedKFold, train_test_split

    return (
        LogisticRegression,
        SentenceTransformer,
        StratifiedKFold,
        average_precision_score,
        np,
        pl,
        precision_recall_curve,
        px,
        train_test_split,
    )


@app.cell
def _(np):
    seed_number = 42
    np.random.seed(seed_number)
    label = "indicative"
    return label, seed_number


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
    mo.md(r"""## Data""")
    return


@app.cell
def _(label, pl):
    df = pl.read_csv("data/indicative_codes_yes_no.csv")
    mapping = {"yes": 1, "no": 0}
    df = df.with_columns(
        [
            df[label]
            .fill_null("")
            .str.strip_chars()
            .str.to_lowercase()
            .map_elements(lambda answer: mapping.get(answer, -1), return_dtype=pl.Int8)
        ]
    )
    df
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""## Embeddings""")
    return


@app.cell
def _(SentenceTransformer):
    # model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
    # sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 0.77
    # paraphrase-multilingual-mpnet-base-v2 0.76
    # jinaai/jinaai-embeddings-v3 0.81
    return (model,)


@app.cell
def _(mo):
    mo.md(r"""## Train & Evaluate""")
    return


@app.cell
def _(df, label, pl, seed_number):
    positives = df.filter(pl.col(label) == 1)
    negatives = df.filter(pl.col(label) == 0)

    df_balanced = pl.concat([positives, negatives]).sample(
        fraction=1.0, seed=seed_number
    )  # embaralhar

    texts = df_balanced["description"].to_list()
    labels = df_balanced[label].to_numpy()
    return labels, texts


@app.cell
def _(model, texts):
    embeddings = model.encode(texts, convert_to_numpy=True)
    return (embeddings,)


@app.cell
def _(embeddings, labels, train_test_split):
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.25, stratify=labels, random_state=42
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(LogisticRegression, X_train, y_train):
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)
    return (clf,)


@app.cell
def _(
    X_test,
    average_precision_score,
    clf,
    mo,
    np,
    precision_recall_curve,
    px,
    y_test,
):
    def evaluate_model(classifier, X_test_data, y_test_data):
        """Avalia o modelo e retorna o melhor threshold e a figura da curva PR."""
        y_probs = classifier.predict_proba(X_test_data)[:, 1]
        prec, rec, thresholds = precision_recall_curve(y_test_data, y_probs)
        avg_prec = average_precision_score(y_test_data, y_probs)

        safe_thresholds = np.append(thresholds, 1.0)

        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_idx = np.argmax(f1_scores)

        best_threshold = safe_thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        print(f"Melhor threshold: {best_threshold:.3f}")
        print(f"F1 ótimo: {best_f1:.3f}")

        fig = px.line(
            x=rec,
            y=prec,
            title=f"Precision-Recall Curve (AP={avg_prec:.2f})",
            labels={"x": "Recall", "y": "Precision"},
        )
        fig.add_scatter(
            x=[rec[best_idx]],
            y=[prec[best_idx]],
            mode="markers+text",
            text=[f"Best thr={best_threshold:.2f}"],
            textposition="top center",
            marker=dict(size=10, color="red"),
            name="Best Threshold",
        )
        return best_threshold, fig

    BEST_THRESHOLD, pr_fig = evaluate_model(clf, X_test, y_test)

    mo.md(
        f"### Curva Precision-Recall\nO melhor threshold encontrado (maior F1-Score) é **{BEST_THRESHOLD:.3f}**."
    )
    pr_fig
    return (BEST_THRESHOLD,)


@app.cell
def _(
    LogisticRegression,
    StratifiedKFold,
    average_precision_score,
    embeddings,
    labels,
    mo,
    np,
):
    mo.md("""### Cross validation""")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(embeddings, labels)):
        X_train_fold, X_test_fold = embeddings[train_idx], embeddings[test_idx]
        y_train_fold, y_test_fold = labels[train_idx], labels[test_idx]

        clf_fold = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf_fold.fit(X_train_fold, y_train_fold)

        y_probs_fold = clf_fold.predict_proba(X_test_fold)[:, 1]
        score = average_precision_score(y_test_fold, y_probs_fold)
        fold_scores.append(score)
        print(f"Fold {fold + 1} - Average Precision: {score:.3f}")

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    mo.md(f"""**Result:**

    - **Average Precision Médio:** `{mean_score:.3f}`
    - **Desvio Padrão:** `{std_score:.3f}`
    """)
    return


@app.cell
def _(LogisticRegression, embeddings, labels, mo):
    mo.md("""### Treino do Modelo Final

    Após avaliar a performance, treinamos o classificador final usando **todos** os dados rotulados disponíveis. Este modelo será usado para fazer as predições em novos dados.
    """)
    final_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    final_clf.fit(embeddings, labels)
    print("Modelo final treinado com sucesso!")
    return (final_clf,)


@app.cell
def _(BEST_THRESHOLD, final_clf, model):
    # Funções utilitárias para predição
    def predict(text):
        _embeddings = model.encode(text, convert_to_numpy=True)
        _probs = final_clf.predict_proba(_embeddings)[:, 1]
        return _probs

    def classify(text):
        _probs = predict(text)
        _preds = (_probs >= BEST_THRESHOLD).astype(int)
        for _txt, _prob, _pred in zip(text, _probs, _preds):
            print(f"{_txt} → prob={_prob:.2f} → {'🚫' if _pred == 1 else '✅'}")

    return (classify,)


@app.cell
def _(mo):
    mo.md("""
    ## 6. Gerar CSV com Todas as Predições

    Clique no botão abaixo para executar o modelo final em todos os 15.000+ códigos e salvar um arquivo CSV com as probabilidades.

    **Atenção:** Este processo pode demorar alguns minutos.
    """)
    run_button = mo.ui.button(label="Gerar Predições e Salvar CSV")
    run_button
    return (run_button,)


@app.cell
def _(run_button):
    run_button.value
    return


@app.cell
def _(final_clf, mo, model, pl):
    # if run_button.value:
    with mo.status.spinner("Executando predições... Isso pode demorar alguns minutos.") as sp:
        # 1. Carregar todos os códigos
        sp.update("Carregando todos os códigos...")
        df_all_codes = pl.read_csv("data/processed/cid10_ciap2_ab.csv")
        all_texts = df_all_codes["description"].to_list()

        # 2. Gerar embeddings
        sp.update(f"Gerando embeddings para {len(all_texts)} códigos...")
        all_embeddings = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=True)

        # 3. Calcular probabilidades
        sp.update("Calculando probabilidades com o modelo final...")
        probabilities = final_clf.predict_proba(all_embeddings)[:, 1]

        # 4. Criar e salvar resultados
        sp.update("Salvando resultados em CSV...")
        df_results = df_all_codes.with_columns(
            pl.Series(name="indicative_probability", values=probabilities)
        )
        output_path = "data/processed/indicative_probabilities_final_model.csv"
        df_results.write_csv(output_path)

        # 5. Exibir confirmação
        mo.md(f"""
        ### ✅ Processo Concluído!

        Resultados salvos em: `{output_path}`

        **Amostra:**
        {df_results.head()}
        """)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Playground

    """
    )
    return


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
        "candidíase",
        "aborto",
        "gravidez",
        "infecção urinária",
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
