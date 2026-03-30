import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go

    return go, mo, np, px


@app.cell
def _():
    from umap import UMAP

    from vir.db import get_all_embeddings

    return UMAP, get_all_embeddings


@app.cell
def _(UMAP):
    umap_reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    return (umap_reducer,)


@app.cell
def _(get_all_embeddings, mo, np):
    @mo.cache
    def load_embeddings():
        embeddings_data = get_all_embeddings()
        if not embeddings_data:
            return None, None, None, None

        embeddings = np.array([item["embedding"] for item in embeddings_data])
        codes = [item["code"] for item in embeddings_data]
        code_types = [item["code_type"] for item in embeddings_data]
        descriptions = [item["description"] for item in embeddings_data]

        return embeddings, codes, code_types, descriptions

    embeddings, codes, code_types, descriptions = load_embeddings()
    return code_types, codes, descriptions, embeddings


@app.cell
def _(embeddings, mo, umap_reducer):
    @mo.cache
    def compute_umap(embeddings):
        """Compute UMAP embeddings for visualization."""
        if embeddings is None:
            return None
        return umap_reducer.fit_transform(embeddings)

    umap_coords = compute_umap(embeddings)
    return (umap_coords,)


@app.cell
def _(codes, color_mapping, mo):
    mo.md(
        f"""
    ## 🏥 Medical Terminology Explorer

    Explore medical terminologies using semantic search and UMAP visualization.

    **Database Info:**

    - Total terms: {len(codes) if codes else 0}
    - Code types: {len(color_mapping)}

    **Available code types:** {", ".join(color_mapping.keys()) if color_mapping else "None"}

    **Instructions:**

    1. Type medical terms, symptoms, or conditions in the search box above
    2. The visualization shows:

       - All terms in light gray
       - Top 10 most similar terms colored by code type
       - Query matches highlighted in red with star symbol

    3. Hover over points to see details
    """
    )
    return


@app.cell
def _(px):
    color_mapping = {
        "cid10": px.colors.qualitative.Plotly[0],
        "ciap2": px.colors.qualitative.Plotly[1],
        "ab": px.colors.qualitative.Plotly[2],
    }
    return (color_mapping,)


@app.cell
def _(mo):
    search_input = mo.ui.text(
        value="",
        placeholder="Enter medical terms, symptoms, or conditions...",
        label="🔍 Medical Terminology Search",
        full_width=True,
    )
    search_input
    return (search_input,)


@app.cell
def _(mo):
    top_k = mo.ui.slider(
        start=10, stop=100, step=10, value=10, label="Top matches", show_value=True
    )
    top_k
    return (top_k,)


@app.cell
def _(
    code_types,
    codes,
    color_mapping,
    descriptions,
    embeddings,
    encode_text,
    go,
    np,
    search_input,
    top_k,
    umap_coords,
):
    def compute_similarities_and_plot(
        search_text, embeddings, umap_coords, codes, code_types, descriptions
    ):
        if not search_text.strip() or embeddings is None:
            return go.Figure(), []

        query_embedding = encode_text(search_text.strip())
        similarities = np.dot(embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][: top_k.value]

        # Create results table data
        results_data = []
        for idx in top_indices:
            results_data.append(
                {
                    "code": codes[idx],
                    "code_type": code_types[idx],
                    "description": descriptions[idx],
                    "score": f"{similarities[idx]:.3f}",
                }
            )

        fig = go.Figure().add_annotation(
            text="",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.add_trace(
            go.Scatter(
                x=umap_coords[:, 0],
                y=umap_coords[:, 1],
                mode="markers",
                marker=dict(size=6, color="lightgray", opacity=0.6),
                text=[f"{code}: {desc}" for code, desc in zip(codes, descriptions)],
                hovertemplate="<b>%{text}</b><br>UMAP: (%{x:.2f}, %{y:.2f})<extra></extra>",
                name="All terms",
                # showlegend=False
            )
        )

        # top similar points with colors by code type
        for index, idx in enumerate(top_indices):
            code_type = code_types[idx]
            color = color_mapping.get(code_type, "gray")
            print(f"{codes[idx]}: {descriptions[idx]}<br>Similarity: {similarities[idx]:.3f}")

            fig.add_trace(
                go.Scatter(
                    x=[umap_coords[idx, 0]],
                    y=[umap_coords[idx, 1]],
                    mode="markers",
                    marker=dict(
                        # size=marker_size,
                        color=color,
                        # symbol=marker_symbol,
                    ),
                    # text=f"{codes[idx]}: {descriptions[idx]}<br>Similarity: {similarities[idx]:.3f}",
                    hovertemplate=f"<b>{descriptions[idx]}</b>Similarity: {similarities[idx]:.3f}",
                    name=code_type,
                    showlegend=False,
                )
            )

        # legend for code types
        for code_type, color in color_mapping.items():
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=8, color=color),
                    name=code_type,
                    showlegend=True,
                )
            )

        fig.update_layout(
            title=f"Medical Terminology Search: '{search_text}'",
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            hovermode="closest",
            width=800,
            height=600,
            margin=dict(l=50, r=50, t=80, b=50),
        )
        return fig, results_data

    if search_input.value:
        plot_fig, results_data = compute_similarities_and_plot(
            search_input.value, embeddings, umap_coords, codes, code_types, descriptions
        )
    else:
        plot_fig = go.Figure().add_annotation(
            text="Enter a search term above to explore medical terminology",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        results_data = []
    return plot_fig, results_data


@app.cell
def _(mo, plot_fig):
    mo.ui.plotly(plot_fig)
    return


@app.cell
def _(mo, results_data):
    if results_data:
        mo.md(f"""
        ## 📊 Search Results Table

        Top {len(results_data)} most similar medical terminologies:
        """)

        # Create table using mo.ui.table
        table_data = []
        for i, result in enumerate(results_data, 1):
            table_data.append(
                {
                    "Rank": i,
                    "Code": result["code"],
                    "Code Type": result["code_type"].upper(),
                    "Description": result["description"],
                    "Similarity Score": result["score"],
                }
            )

        mo.ui.table(data=table_data, selection=None, pagination=True, page_size=10)
    else:
        mo.md("Enter a search term above to see results in the table below.")
    return


if __name__ == "__main__":
    app.run()
