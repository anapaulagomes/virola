from pathlib import Path

import polars as pl
import typer

app = typer.Typer()


@app.command("df")
def main(
    input_path: Path,
    function: str = "head",
):
    df = pl.read_parquet(input_path)
    print(getattr(df, function)())


if __name__ == "__main__":
    app()
