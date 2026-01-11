"""Main CLI for Virola - Multi-view clustering for epidemiological surveillance."""

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from vir.config import PROJ_ROOT
from vir.dataset import app as helper_app
from vir.db import DEFAULT_MODEL
from vir.features import CITY_NAMES

app = typer.Typer(
    name="vir",
    help="Virola: Multi-view clustering for epidemiological surveillance",
    add_completion=False,
)


setup_app = typer.Typer(help="Setup and initialization commands")
prepare_app = typer.Typer(help="Data preparation pipeline")
view_app = typer.Typer(help="Analysis views (clinical, demographic, temporal)")
model_app = typer.Typer(help="Modeling commands")

app.add_typer(setup_app, name="setup")
app.add_typer(prepare_app, name="prepare")
app.add_typer(view_app, name="view")
app.add_typer(model_app, name="model")
app.add_typer(helper_app, name="helpers")


def run_marimo_notebook(notebook_path: Path, args: list[str]) -> int:
    """
    Run a marimo notebook in script mode with arguments.

    Args:
        notebook_path: Path to marimo notebook
        args: List of command line arguments to pass

    Returns:
        Exit code from marimo process
    """

    # Use marimo run to execute the notebook with arguments
    # This properly handles script mode and argument parsing
    cmd = ["python", str(notebook_path)]
    if args:
        cmd.extend(args)

    result = subprocess.run(cmd)
    return result.returncode


# ============================================================================
# Setup Commands
# ============================================================================


@setup_app.command("terminologies")
def setup_terminologies():
    """Create terminology files from source data."""
    from vir.terminologies.terminologies import main as run_terminologies

    run_terminologies()
    typer.secho("✓ Terminology files created successfully", fg=typer.colors.GREEN)


@setup_app.command("embeddings")
def setup_embeddings(
    model: Optional[str] = typer.Option(
        None, "--model", help=f"Transformer model name (uses {DEFAULT_MODEL} if not specified)"
    ),
):
    """Generate medical code embeddings database."""
    notebook = PROJ_ROOT / "vir/terminologies/embeddings_model.py"
    args = []
    if model:
        args.extend(["--model", model])

    exit_code = run_marimo_notebook(notebook, args)
    if exit_code == 0:
        typer.secho("✓ Embeddings database created successfully", fg=typer.colors.GREEN)
    sys.exit(exit_code)


# ============================================================================
# Prepare Commands
# ============================================================================


@prepare_app.command("clean")
def prepare_clean(
    city: Optional[str] = typer.Option(
        None, "--city", help=f"City name to filter: {', '.join(CITY_NAMES)}"
    ),
    input_file: Optional[Path] = typer.Option(None, "--input", help="Input parquet file"),
    output_dir: Optional[Path] = typer.Option(None, "--output", help="Output directory"),
):
    """Clean and standardize raw data."""
    notebook = PROJ_ROOT / "vir/modeling/data_cleaning.py"
    args = []
    if city:
        args.extend(["--city", city])
    if input_file:
        args.extend(["--input", str(input_file)])
    if output_dir:
        args.extend(["--output", str(output_dir)])

    exit_code = run_marimo_notebook(notebook, args)
    if exit_code == 0:
        typer.secho("✓ Data cleaning completed successfully", fg=typer.colors.GREEN)
    sys.exit(exit_code)


@prepare_app.command("features")
def prepare_features(
    dataset_filename: Path = typer.Argument(..., help="Path to cleaned dataset parquet file"),
    level: str = typer.Option(
        "records", "--level", help="Processing level: 'records' or 'diagnosis'"
    ),
):
    """Extract features from cleaned data."""
    cmd = ["python", "-m", "vir.modeling.feature_engineering", str(dataset_filename)]
    if level != "records":
        cmd.extend(["--level", level])

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode == 0:
        typer.secho("✓ Feature extraction completed successfully", fg=typer.colors.GREEN)
    sys.exit(result.returncode)


# ============================================================================
# View Commands
# ============================================================================


@view_app.command("clinical")
def view_clinical(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset file path or name"),
    year_min: int = typer.Option(2022, "--year-min", help="Minimum year (inclusive)"),
    year_max: int = typer.Option(2024, "--year-max", help="Maximum year (inclusive)"),
):
    """Run clinical view analysis using semantic similarity."""
    notebook = PROJ_ROOT / "notebooks/view_clinical.py"
    args = [
        "--dataset",
        dataset,
        "--year-min",
        str(year_min),
        "--year-max",
        str(year_max),
    ]

    exit_code = run_marimo_notebook(notebook, args)
    sys.exit(exit_code)


@view_app.command("demographic")
def view_demographic(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset file path or name"),
    year_min: int = typer.Option(2022, "--year-min", help="Minimum year (inclusive)"),
    year_max: int = typer.Option(2024, "--year-max", help="Maximum year (inclusive)"),
):
    """Run demographic view analysis."""
    notebook = PROJ_ROOT / "notebooks/view_demographic.py"
    args = [
        "--dataset",
        dataset,
        "--year-min",
        str(year_min),
        "--year-max",
        str(year_max),
    ]

    exit_code = run_marimo_notebook(notebook, args)
    sys.exit(exit_code)


@view_app.command("temporal")
def view_temporal(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset file path or name"),
    year_min: int = typer.Option(2022, "--year-min", help="Minimum year (inclusive)"),
    year_max: int = typer.Option(2024, "--year-max", help="Maximum year (inclusive)"),
    min_weeks: int = typer.Option(12, "--min-weeks", help="Minimum weeks of data required"),
):
    """Run temporal view analysis using time-series patterns."""

    notebook = PROJ_ROOT / "notebooks/view_temporal.py"
    args = [
        "--dataset",
        dataset,
        "--year-min",
        str(year_min),
        "--year-max",
        str(year_max),
        "--min-weeks",
        str(min_weeks),
    ]

    exit_code = run_marimo_notebook(notebook, args)
    sys.exit(exit_code)


# ============================================================================
# Model Commands
# ============================================================================


@model_app.command("indicative-codes")
def model_indicative_codes(
    input_file: Path = typer.Option(..., "--input", help="Input CSV file with labeled data"),
    output_file: Path = typer.Option(..., "--output", help="Output CSV file for probabilities"),
):
    """Train indicative codes classifier using SetFit."""
    notebook = PROJ_ROOT / "notebooks/model_indicative_codes.py"
    args = [
        "--input",
        str(input_file),
        "--output",
        str(output_file),
    ]

    exit_code = run_marimo_notebook(notebook, args)
    if exit_code == 0:
        typer.secho(
            f"✓ Model trained successfully. Results saved to: {output_file}",
            fg=typer.colors.GREEN,
        )
    sys.exit(exit_code)


@model_app.command("snf")
def model_snf(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset file path or name"),
    year_min: int = typer.Option(2022, "--year-min", help="Minimum year (inclusive)"),
    year_max: int = typer.Option(2024, "--year-max", help="Maximum year (inclusive)"),
    snf_k: int = typer.Option(10, "--snf-k", help="SNF parameter k"),
    n_clusters: int = typer.Option(200, "--n-clusters", help="Number of clusters"),
    output_dir: Path = typer.Option(None, "--output", help="Output directory for results"),
):
    """Run Similarity Network Fusion clustering."""
    notebook = PROJ_ROOT / "notebooks/model_similarity_network_fusion.py"
    args = [
        "--dataset",
        dataset,
        "--year-min",
        str(year_min),
        "--year-max",
        str(year_max),
        "--snf-k",
        str(snf_k),
        "--n-clusters",
        str(n_clusters),
        "--output",
        str(output_dir),
    ]

    exit_code = run_marimo_notebook(notebook, args)
    if exit_code == 0:
        typer.secho(
            f"✓ SNF clustering completed. Results saved to: {output_dir}",
            fg=typer.colors.GREEN,
        )
    sys.exit(exit_code)


if __name__ == "__main__":
    app()
