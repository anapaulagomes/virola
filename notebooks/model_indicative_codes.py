import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    import torch
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments
    from sklearn.metrics import average_precision_score, precision_recall_curve
    from sklearn.model_selection import train_test_split

    return (
        Dataset,
        Path,
        SetFitModel,
        Trainer,
        TrainingArguments,
        average_precision_score,
        datetime,
        go,
        np,
        pl,
        precision_recall_curve,
        px,
        torch,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Indicative codes for epidemiological surveillance 😷

    **Goal:** Classify 14,000+ diagnosis descriptions using a local, few-shot model.

    **Classes:**

    * Indicative: ✅ Indicative of disease, symptoms, or important for epidemiology.
    * Non-indicative: 🚫 Administrative, social issues, accidents, routine tasks, insignificant, preventive, screening, routine, administrative, or related to contact with the healthcare system.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Data
    """)
    return


@app.cell
def _(Path, np):
    seed_number = 42
    np.random.seed(seed_number)
    label_col = "indicative"

    # paths
    output_path = Path("data/processed/indicative_probabilities_setfit.csv")
    active_learning_labels_path = Path("data/active_learning_labels.csv")
    active_learning_metrics_path = Path("data/active_learning_metrics.csv")
    return (
        active_learning_labels_path,
        active_learning_metrics_path,
        label_col,
        output_path,
        seed_number,
    )


@app.cell
def _(active_learning_labels_path, label_col, pl, seed_number):
    df = pl.read_csv("data/indicative_codes_yes_no.csv")
    mapping = {"yes": 1, "no": 0}

    df = df.with_columns(
        [
            df[label_col]
            .fill_null("")
            .str.strip_chars()
            .str.to_lowercase()
            .map_elements(lambda answer: mapping.get(answer, -1), return_dtype=pl.Int8)
        ]
    )

    # merge with active learning labels if they exist
    if active_learning_labels_path.exists():
        df_al = pl.read_csv(active_learning_labels_path)
        df_al = df_al.with_columns(
            pl.col("indicative")
            .str.strip_chars()
            .str.to_lowercase()
            .map_elements(lambda answer: mapping.get(answer, -1), return_dtype=pl.Int8)
        )
        # update labels in df with active learning labels
        df = (
            df.join(
                df_al.select(["description", "indicative"]),
                on="description",
                how="left",
                suffix="_al",
            )
            .with_columns(
                pl.coalesce([pl.col("indicative_al"), pl.col("indicative")]).alias("indicative")
            )
            .drop("indicative_al")
        )
        print(f"📚 Merged {len(df_al)} active learning labels")

    df_labeled = df.filter(pl.col(label_col).is_in([0, 1]))
    df_all_codes = df.filter(pl.col(label_col).eq(-1))

    positives = df_labeled.filter(pl.col(label_col) == 1)
    negatives = df_labeled.filter(pl.col(label_col) == 0)

    print(
        f"✅ Labeled dataset loaded: "
        f"{len(positives)} indicatives + {len(negatives)} non-indicatives = {len(df_labeled)} total"
    )

    df_labeled = df_labeled.sample(fraction=1.0, seed=seed_number)
    texts = df_labeled["description"].to_list()
    labels = df_labeled[label_col].to_numpy()
    return df_all_codes, df_labeled, labels, texts


@app.cell
def _(Dataset, labels, texts, train_test_split):
    # create train/test split (80/20) for valid evaluation
    # using stratify to ensure equal distribution of classes in train and test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )

    print(f"Train set: {len(train_texts)} samples")
    print(f"Test set (Holdout): {len(test_texts)} samples")

    # convert to HuggingFace Dataset format
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})
    return test_dataset, test_labels, train_dataset


@app.cell
def _(mo):
    mo.md(r"""
    ## Model Initialization & Training
    """)
    return


@app.cell
def _(SetFitModel):
    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        use_differentiable_head=True,
        head_params={"out_features": 2},  # indicative or non-indicative
    )
    return (model,)


@app.cell
def _(Trainer, TrainingArguments, model, train_dataset):
    args = TrainingArguments(
        batch_size=32,
        num_epochs=1,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        # sampling_strategy="unique",
        num_iterations=20,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        metric="accuracy",
    )

    print("⏳ Training model...")
    trainer.train()
    print("✅ Training complete!")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Evaluation
    """)
    return


@app.cell
def _(
    average_precision_score,
    model,
    np,
    precision_recall_curve,
    px,
    test_dataset,
    test_labels,
    torch,
):
    # get probabilities on the holdout set
    y_pred_probs = model.predict_proba(test_dataset["text"])[:, 1]
    if isinstance(y_pred_probs, torch.Tensor):
        y_pred_probs = y_pred_probs.detach().cpu().numpy()

    _precision, _recall, thresholds = precision_recall_curve(test_labels, y_pred_probs)
    _avg_prec = average_precision_score(test_labels, y_pred_probs)

    safe_thresholds = np.append(thresholds, 1.0)  # match shapes of prec/rec

    _f1_scores = 2 * (_precision * _recall) / (_precision + _recall + 1e-8)
    _best_idx = np.argmax(_f1_scores)
    _best_f1 = _f1_scores[_best_idx]
    best_threshold = safe_thresholds[_best_idx]

    print("Results on Test Set:")
    print("---------------------")
    print(f"Average Precision (AP): {_avg_prec:.3f}")
    print(f"Optimal Threshold:      {best_threshold:.3f}")
    print(f"Best F1 Score:          {_best_f1:.3f}")
    print(f"Precision at Best:      {_precision[_best_idx]:.3f}")
    print(f"Recall at Best:         {_recall[_best_idx]:.3f}")

    _fig = px.line(
        x=_recall,
        y=_precision,
        title=f"Precision-Recall Curve (AP={_avg_prec:.2f})",
        labels={"x": "Recall", "y": "Precision"},
        markers=True,
    )
    _fig.add_scatter(
        x=[_recall[_best_idx]],
        y=[_precision[_best_idx]],
        mode="markers+text",
        text=[f"Best F1={_best_f1:.2f}"],
        textposition="top right",
        marker=dict(size=12, color="red", symbol="star"),
        name="Optimal threshold",
    )
    _fig
    return best_threshold, y_pred_probs


@app.cell
def _(mo):
    mo.md(r"""
    ## Prediction

    For ICD-10, ICPC-2, and AB (Brazilian primary care legacy code).
    """)
    return


@app.cell
def _(best_threshold, df_all_codes, model, pl, torch):
    print("⏳ Running batch prediction...")
    probs = model.predict_proba(df_all_codes["description"].to_list())[:, 1]
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()

    _preds = (probs >= best_threshold).astype(int)

    df_results = df_all_codes.with_columns(
        [
            pl.Series(name="indicative_prob", values=probs),
            pl.Series(name="predicted_label", values=_preds),
        ]
    )

    df_results = df_results.with_columns(
        pl.when(pl.col("predicted_label") == 1)
        .then(pl.lit("indicative"))
        .otherwise(pl.lit("non-indicative"))
        .alias("category_name")
    )

    print("✅ Prediction complete.")
    return df_results, probs


@app.cell
def _(mo):
    mo.md(r"""
    ## Validation
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Uncertainty check (Active Learning)

    These are the rows where the model is roughly 50/50.
    """)
    return


@app.cell
def _(df_results, mo, pl):
    uncertain_rows = (
        df_results.filter(pl.col("indicative_prob").is_between(0.50, 0.51))
        .sort("indicative_prob")
        .head(10)
    )

    mo.ui.table(uncertain_rows.select(["description", "indicative_prob", "category_name"]))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Active Learning 🎯

    Scientifically sound approach:
    - 70% uncertainty sampling (near decision boundary)
    - 30% random sampling (for diversity)
    - Metadata tracking for reproducibility
    """)
    return


@app.cell
def _(mo):
    # configuration
    al_batch_size = mo.ui.slider(
        start=5, stop=30, value=10, step=1, label="Batch size", show_value=True
    )
    al_uncertainty_ratio = mo.ui.slider(
        start=0.5,
        stop=1.0,
        value=0.7,
        step=0.1,
        label="Uncertainty ratio (vs random)",
        show_value=True,
    )
    mo.hstack([al_batch_size, al_uncertainty_ratio], justify="start")
    return al_batch_size, al_uncertainty_ratio


@app.cell
def _(
    al_batch_size,
    al_uncertainty_ratio,
    best_threshold,
    df_all_codes,
    df_labeled,
    np,
    pl,
    seed_number,
):
    # exclude already labeled samples from pool
    _already_labeled = set(df_labeled["description"].to_list())
    _pool = df_all_codes.filter(~pl.col("description").is_in(_already_labeled))

    # calculate uncertainty scores for all pool samples
    _pool_with_scores = _pool.with_columns(
        (pl.col("indicative") - best_threshold).abs().alias("distance_to_boundary")
    )

    # mixed sampling strategy
    _n_uncertain = int(al_batch_size.value * al_uncertainty_ratio.value)
    _n_random = al_batch_size.value - _n_uncertain

    # uncertainty sampling: closest to decision boundary
    _uncertain_candidates = _pool_with_scores.sort("distance_to_boundary").head(_n_uncertain)

    # random sampling: from remaining pool
    np.random.seed(seed_number)
    _remaining_pool = _pool_with_scores.filter(
        ~pl.col("description").is_in(_uncertain_candidates["description"])
    )
    _random_candidates = _remaining_pool.sample(n=min(_n_random, len(_remaining_pool)))

    # combine both strategies
    al_candidates = pl.concat([_uncertain_candidates, _random_candidates]).with_columns(
        pl.Series(
            name="sampling_strategy",
            values=["uncertainty"] * len(_uncertain_candidates)
            + ["random"] * len(_random_candidates),
        )
    )

    print(
        f"📊 Selected {_n_uncertain} uncertain + {_n_random} random = {len(al_candidates)} total"
    )
    return (al_candidates,)


@app.cell
def _(al_candidates, mo):
    # create labeling interface
    _widgets = []
    for _row in al_candidates.iter_rows(named=True):
        desc = _row["description"]
        prob = _row["indicative"]
        strategy = _row["sampling_strategy"]
        icon = "🎲" if strategy == "random" else "⚖️"

        radio = mo.ui.radio(
            options={"yes": "✅ Indicative", "no": "🚫 Non-Indicative", "skip": "⏭️ Skip"},
            value="skip",
            label=f"{icon} **[{prob:.1%}]** {desc}",
        )
        _widgets.append(radio)

    al_labeling_form = mo.ui.form(
        mo.ui.array(_widgets),  # Usar array ao invés de vstack
        submit_button_label="💾 Save Labels",
        bordered=True,
    )
    al_labeling_form
    return (al_labeling_form,)


@app.cell
def _(
    active_learning_labels_path,
    active_learning_metrics_path,
    al_candidates,
    al_labeling_form,
    average_precision_score,
    best_threshold,
    datetime,
    mo,
    pl,
    test_labels,
    y_pred_probs,
):
    if al_labeling_form.value:
        _labeled_samples = []
        _timestamp = datetime.now().isoformat()

        # collect labeled samples
        for _idx, _row in enumerate(al_candidates.iter_rows(named=True)):
            label_value = al_labeling_form.value[_idx]
            if label_value and label_value != "skip":
                _labeled_samples.append(
                    {
                        "description": _row["description"],
                        "indicative": label_value,
                        "indicative_prob": _row["indicative"],
                        "distance_to_boundary": _row["distance_to_boundary"],
                        "sampling_strategy": _row["sampling_strategy"],
                        "timestamp": _timestamp,
                    }
                )

        if _labeled_samples:
            _df_new = pl.DataFrame(_labeled_samples)

            # append to existing labels
            if active_learning_labels_path.exists():
                _existing = pl.read_csv(active_learning_labels_path)
                _df_new = pl.concat([_existing, _df_new])

            _df_new.write_csv(active_learning_labels_path)

            # track iteration metrics
            _iteration = (
                pl.read_csv(active_learning_metrics_path).height
                if active_learning_metrics_path.exists()
                else 0
            )
            _metrics_row = pl.DataFrame(
                {
                    "iteration": [_iteration + 1],
                    "n_samples_added": [len(_labeled_samples)],
                    "total_labeled": [len(_df_new)],
                    "test_avg_precision": [average_precision_score(test_labels, y_pred_probs)],
                    "best_threshold": [best_threshold],
                    "timestamp": [_timestamp],
                }
            )

            if active_learning_metrics_path.exists():
                _existing_metrics = pl.read_csv(active_learning_metrics_path)
                _metrics_row = pl.concat([_existing_metrics, _metrics_row])

            _metrics_row.write_csv(active_learning_metrics_path)

            mo.md(
                f"""
            ✅ **Saved {len(_labeled_samples)} labels** (iteration {_iteration + 1})

            Files updated:
            - `{active_learning_labels_path}`
            - `{active_learning_metrics_path}`

            **Next:** Restart the notebook to retrain with new labels
            """
            )
        else:
            mo.md("⚠️ No labels selected (all skipped)")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Learning Curve

    Track model performance across active learning iterations.
    """)
    return


@app.cell
def _(active_learning_metrics_path, go, mo, pl):
    if active_learning_metrics_path.exists():
        _metrics_df = pl.read_csv(active_learning_metrics_path)

        _fig = go.Figure()
        _fig.add_trace(
            go.Scatter(
                x=_metrics_df["total_labeled"],
                y=_metrics_df["test_avg_precision"],
                mode="lines+markers",
                name="Average Precision",
                marker=dict(size=10),
                line=dict(width=2),
            )
        )
        _fig.update_layout(
            title="Active Learning Progress",
            xaxis_title="Total Labeled Samples",
            yaxis_title="Test Set Average Precision",
            yaxis=dict(range=[0, 1]),
            hovermode="x unified",
        )

        mo.vstack(
            [
                _fig,
                mo.md(f"**Current iteration:** {_metrics_df.height}"),
                mo.ui.table(
                    _metrics_df.select(
                        [
                            "iteration",
                            "n_samples_added",
                            "total_labeled",
                            "test_avg_precision",
                            "best_threshold",
                        ]
                    )
                ),
            ]
        )
    else:
        mo.md("No active learning iterations yet. Label some samples to start tracking progress.")
    return


@app.cell
def _(df_results, mo, pl):
    mo.md("""### Sanity check
    Verify that the "High Confidence" predictions make sense.
    """)

    top_disease = (
        df_results.filter(pl.col("predicted_label") == 1)
        .sort("indicative_prob", descending=True)
        .head(5)
    )
    top_admin = (
        df_results.filter(pl.col("predicted_label") == 0)
        .sort("indicative_prob", descending=False)
        .head(5)
    )

    mo.vstack(
        [
            mo.md("Model is MOST CONFIDENT these are **indicative**:"),
            mo.ui.table(top_disease.select(["description", "indicative_prob"])),
            mo.md("---"),
            mo.md("Model is MOST CONFIDENT these are **non-indicative**:"),
            mo.ui.table(top_admin.select(["description", "indicative_prob"])),
        ]
    )
    return


@app.cell
def _(df_results, output_path):
    df_results.write_csv(output_path)
    print(f"💾 Results saved to: {output_path}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Playground
    """)
    return


@app.cell
def _(best_threshold, model, probs, torch):
    def check_texts(text_list):
        _probs = model.predict_proba(text_list)[:, 1]
        if isinstance(probs, torch.Tensor):
            _probs = _probs.detach().cpu().numpy()
        for txt, p in zip(text_list, _probs):
            icon = "✅ Indicative" if p >= best_threshold else "🚫 Non-Indicative"
            print(f"[{p:.1%}] {icon} : {txt}")

    return (check_texts,)


@app.cell
def _(mo):
    text_input = mo.ui.text_area(
        label="Type descriptions to test (one per line):",
        value="Diabetes Tipo 2\nRenovação de receita\nDor no peito\nConsulta médica",
        full_width=True,
    )
    text_input
    return (text_input,)


@app.cell
def _(check_texts, text_input):
    if text_input.value:
        check_texts(text_input.value.splitlines())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Analyzing False Positives
    """)
    return


@app.cell
def _(best_threshold, mo, pl, test_dataset, test_labels, y_pred_probs):
    _preds = (y_pred_probs >= best_threshold).astype(int)
    _df_test = pl.DataFrame(
        {
            "text": test_dataset["text"],
            "true_label": test_labels,  # 0 or 1 from manual labels
            "pred_label": _preds,  # what the model predicted
            "prob": y_pred_probs,
        }
    )

    false_positives = _df_test.filter(
        (pl.col("true_label") == 0) & (pl.col("pred_label") == 1)
    ).sort("prob", descending=True)

    mo.ui.table(false_positives)
    return


if __name__ == "__main__":
    app.run()
