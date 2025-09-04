from pathlib import Path

import typer
from loguru import logger

app = typer.Typer()


@app.command()
def main(
    processed_path: Path,
    label: str = None,
    model: str = "hdbscan",
):
    if label is None:
        label = ""
    match model:
        case "hdbscan":
            from vir.modeling.clustering.hdbscan import main as hdbscan_main

            hdbscan_main(processed_path, label)
        case _:
            raise ValueError(f"Unknown model: {model}")
    logger.success("Clustering completed.")


if __name__ == "__main__":
    app()
