import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import plotly.graph_objects as go
    import numpy as np
    from sklearn.decomposition import PCA
    import hdbscan

    from vir.config import PROCESSED_DATA_DIR
    return PCA, PROCESSED_DATA_DIR, go, hdbscan, mo, np, pl


@app.cell
def _(PROCESSED_DATA_DIR, pl):
    df = pl.read_parquet(PROCESSED_DATA_DIR / "processed_aesop_raw_dados_2017_20250223_20municipios.parquet")
    df
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""## Reduce dimentionality""")
    return


@app.cell
def _(PCA, df):
    pca = PCA(n_components=10)
    pca.fit_transform(df)
    prop_var = pca.explained_variance_ratio_
    eigenvalues = pca.explained_variance_
    return eigenvalues, pca, prop_var


@app.cell
def _(eigenvalues):
    eigenvalues
    return


@app.cell
def _(go, np, pca, prop_var):
    _fig = go.Figure(
        data=go.Scatter(
            x=np.arange(pca.n_components_) + 1,
            y=prop_var,
            mode='lines+markers',
            marker=dict(color='red'),
            line=dict(color='red'),
            name='Variance'
        )
    )

    _fig.update_layout(
        title='Figure 1: Scree Plot',
        title_font=dict(size=8),
        yaxis_title='Proportion of Variance',
        yaxis_title_font=dict(size=8),
        xaxis_title='Principal Component',
        font=dict(size=8)
    )
    _fig
    return


@app.cell(hide_code=True)
def _(PCA, df):
    reduced_df = PCA(n_components=3).fit_transform(df)
    reduced_df
    return (reduced_df,)


@app.cell
def _(df, hdbscan, reduced_df):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=20,    # ajusta p/ seu caso
        min_samples=5,          # mais robustez a outliers
        metric='euclidean'
    )
    labels = clusterer.fit_predict(reduced_df)

    df_clusters = df.clone()
    df_clusters = df_clusters.with_columns(cluster=labels)
    return (df_clusters,)


@app.cell
def _(df_clusters):
    df_clusters["cluster"].value_counts()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
