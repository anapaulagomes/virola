from pathlib import Path

import typer
from loguru import logger

app = typer.Typer()


@app.command()
def main(
    processed_path: Path,
    label: str = None,
    model: str = None,
):
    if label is None:
        label = ""
    match model:
        case _:
            raise ValueError(f"Unknown model: {model}")
    logger.warning("Clustering disabled. This is a placeholder.")


if __name__ == "__main__":
    app()
