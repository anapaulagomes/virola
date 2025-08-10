from pathlib import Path

import polars as pl
import typer

app = typer.Typer()


@app.command("df")
def main(
    input_path: Path,
    function: str = "head",
    attribute: str = None,
):
    df = pl.read_parquet(input_path)
    if attribute:
        print(getattr(df, attribute))
    else:
        with pl.Config(tbl_cols=-1):
            print(getattr(df, function)())


if __name__ == "__main__":
    app()
