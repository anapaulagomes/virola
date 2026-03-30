"""Main CLI for Virola - Multi-view clustering for epidemiological surveillance."""

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from vir.config import CLUSTERING_DIR, DEFAULT_MODEL, PROJ_ROOT
from vir.dataset import app as helper_app
from vir.features import CITY_NAMES
from vir.helpers import get_n_jobs

app = typer.Typer(
    name="vir",
    help="Virola: Multi-view clustering for epidemiological surveillance",
    add_completion=False,
)


setup_app = typer.Typer(help="Setup and initialization commands")
prepare_app = typer.Typer(help="Data preparation pipeline")
view_app = typer.Typer(help="Analysis views (clinical, demographic, temporal)")
model_app = typer.Typer(help="Modeling commands")
pipeline_app = typer.Typer(help="End-to-end pipeline commands")
results_app = typer.Typer(help="Query and enrich saved runs")

app.add_typer(setup_app, name="setup")
app.add_typer(prepare_app, name="prepare")
app.add_typer(view_app, name="view")
app.add_typer(model_app, name="model")
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(results_app, name="results")
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


# ============================================================================
# View Commands
# ============================================================================


@view_app.command("clinical")
def view_clinical(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset file path or name"),
    year_min: int = typer.Option(2022, "--year-min", help="Minimum year (inclusive)"),
    year_max: int = typer.Option(2024, "--year-max", help="Maximum year (inclusive)"),
    no_indicative: bool = typer.Option(
        False, "--no-indicative", help="Include all codes (skip indicative filtering)"
    ),
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
    if no_indicative:
        args.append("--no-indicative")

    exit_code = run_marimo_notebook(notebook, args)
    sys.exit(exit_code)


@view_app.command("demographic")
def view_demographic(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset file path or name"),
    year_min: int = typer.Option(2022, "--year-min", help="Minimum year (inclusive)"),
    year_max: int = typer.Option(2024, "--year-max", help="Maximum year (inclusive)"),
    no_indicative: bool = typer.Option(
        False, "--no-indicative", help="Include all codes (skip indicative filtering)"
    ),
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
    if no_indicative:
        args.append("--no-indicative")

    exit_code = run_marimo_notebook(notebook, args)
    sys.exit(exit_code)


@view_app.command("temporal")
def view_temporal(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset file path or name"),
    year_min: int = typer.Option(2022, "--year-min", help="Minimum year (inclusive)"),
    year_max: int = typer.Option(2024, "--year-max", help="Maximum year (inclusive)"),
    min_weeks: int = typer.Option(12, "--min-weeks", help="Minimum weeks of data required"),
    no_indicative: bool = typer.Option(
        False, "--no-indicative", help="Include all codes (skip indicative filtering)"
    ),
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
    if no_indicative:
        args.append("--no-indicative")

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
    notes: Optional[str] = typer.Option(
        None, "--notes", help="Free-text note for this run (e.g. 'run for paper')"
    ),
    no_indicative: bool = typer.Option(
        False, "--no-indicative", help="Include all codes (skip indicative filtering)"
    ),
):
    """Run Similarity Network Fusion clustering. Results are saved to virola.db."""
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
    ]
    if notes:
        args.extend(["--notes", notes])
    if no_indicative:
        args.append("--no-indicative")

    exit_code = run_marimo_notebook(notebook, args)
    if exit_code == 0:
        typer.secho("✓ SNF clustering completed. Run saved to virola.db.", fg=typer.colors.GREEN)
    sys.exit(exit_code)


# ============================================================================
# Pipeline Commands
# ============================================================================


@pipeline_app.command("run")
def pipeline_run(
    city: str = typer.Argument(..., help=f"City name: {', '.join(CITY_NAMES)}"),
    year_min: int = typer.Option(2022, "--year-min", help="Minimum year (inclusive)"),
    year_max: int = typer.Option(2024, "--year-max", help="Maximum year (inclusive)"),
    min_weeks: int = typer.Option(12, "--min-weeks", help="Minimum weeks for temporal view"),
    snf_k: int = typer.Option(10, "--snf-k", help="SNF parameter k"),
    n_clusters: int = typer.Option(200, "--n-clusters", help="Number of clusters"),
    skip_clean: bool = typer.Option(False, "--skip-clean", help="Skip data cleaning step"),
    no_indicative: bool = typer.Option(
        False, "--no-indicative", help="Include all codes (skip indicative filtering)"
    ),
    notes: Optional[str] = typer.Option(
        None, "--notes", help="Free-text note attached to the SNF run (e.g. 'run for paper')"
    ),
):
    """Run complete pipeline for a city: clean -> views -> model snf."""

    if city not in CITY_NAMES:
        typer.secho(
            f"✗ Invalid city name. Available cities: {', '.join(CITY_NAMES)}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    typer.secho(f"\n{'=' * 60}", fg=typer.colors.CYAN)
    typer.secho(f"Running pipeline for: {city}", fg=typer.colors.CYAN, bold=True)
    typer.secho(f"{'=' * 60}\n", fg=typer.colors.CYAN)

    # Step 1: Clean data
    if not skip_clean:
        typer.secho("Step 1/5: Cleaning data...", fg=typer.colors.BLUE, bold=True)
        notebook = PROJ_ROOT / "vir/modeling/data_cleaning.py"
        args = ["--city", city]
        exit_code = run_marimo_notebook(notebook, args)
        if exit_code != 0:
            typer.secho("✗ Data cleaning failed", fg=typer.colors.RED)
            sys.exit(exit_code)
        typer.secho("✓ Data cleaning completed\n", fg=typer.colors.GREEN)
    else:
        typer.secho("Step 1/5: Skipping data cleaning\n", fg=typer.colors.YELLOW)

    # Construct dataset name based on city (use normalized name for file matching)
    dataset = city.lower().replace(" ", "_")

    indicative_args = ["--no-indicative"] if no_indicative else []

    # Step 2: Clinical view
    typer.secho("Step 2/5: Running clinical view analysis...", fg=typer.colors.BLUE, bold=True)
    notebook = PROJ_ROOT / "notebooks/view_clinical.py"
    args = [
        "--dataset",
        dataset,
        "--year-min",
        str(year_min),
        "--year-max",
        str(year_max),
        *indicative_args,
    ]
    exit_code = run_marimo_notebook(notebook, args)
    if exit_code != 0:
        typer.secho("✗ Clinical view failed", fg=typer.colors.RED)
        sys.exit(exit_code)
    typer.secho("✓ Clinical view completed\n", fg=typer.colors.GREEN)

    # Step 3: Demographic view
    typer.secho("Step 3/5: Running demographic view analysis...", fg=typer.colors.BLUE, bold=True)
    notebook = PROJ_ROOT / "notebooks/view_demographic.py"
    args = [
        "--dataset",
        dataset,
        "--year-min",
        str(year_min),
        "--year-max",
        str(year_max),
        *indicative_args,
    ]
    exit_code = run_marimo_notebook(notebook, args)
    if exit_code != 0:
        typer.secho("✗ Demographic view failed", fg=typer.colors.RED)
        sys.exit(exit_code)
    typer.secho("✓ Demographic view completed\n", fg=typer.colors.GREEN)

    # Step 4: Temporal view
    typer.secho("Step 4/5: Running temporal view analysis...", fg=typer.colors.BLUE, bold=True)
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
        *indicative_args,
    ]
    exit_code = run_marimo_notebook(notebook, args)
    if exit_code != 0:
        typer.secho("✗ Temporal view failed", fg=typer.colors.RED)
        sys.exit(exit_code)
    typer.secho("✓ Temporal view completed\n", fg=typer.colors.GREEN)

    # Step 5: SNF clustering
    typer.secho("Step 5/5: Running SNF clustering...", fg=typer.colors.BLUE, bold=True)
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
        str(CLUSTERING_DIR),
        *indicative_args,
    ]
    if notes:
        args.extend(["--notes", notes])
    exit_code = run_marimo_notebook(notebook, args)
    if exit_code != 0:
        typer.secho("✗ SNF clustering failed", fg=typer.colors.RED)
        sys.exit(exit_code)
    typer.secho("✓ SNF clustering completed\n", fg=typer.colors.GREEN)

    typer.secho(f"\n{'=' * 60}", fg=typer.colors.GREEN)
    typer.secho(f"✓ Pipeline completed successfully for {city}!", fg=typer.colors.GREEN, bold=True)
    typer.secho(f"{'=' * 60}\n", fg=typer.colors.GREEN)
    typer.secho(f"Results saved to: {CLUSTERING_DIR}", fg=typer.colors.CYAN)


# ============================================================================
# Results Commands
# ============================================================================


@results_app.command("list")
def results_list():
    """List all saved runs."""
    from vir.db import fetch_runs

    runs = fetch_runs()
    if runs.is_empty():
        typer.secho("No runs found.", fg=typer.colors.YELLOW)
        return
    typer.echo(runs)


@results_app.command("build-profiles")
def results_build_profiles(
    run_id: int = typer.Argument(..., help="run_id to build profiles for"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset file path or name"),
    year_min: int = typer.Option(2022, "--year-min", help="Minimum year (inclusive)"),
    year_max: int = typer.Option(2024, "--year-max", help="Maximum year (inclusive)"),
    no_indicative: bool = typer.Option(
        False, "--no-indicative", help="Include all codes (skip indicative filtering)"
    ),
):
    """Build cluster profiles (stats + temporal + demographic) for a run."""
    from vir.db import fetch_cluster_assignments
    from vir.notebook_widgets import (
        load_and_prepare_dataframe,
        load_indicative_codes,
        load_terminology_labels,
    )
    from vir.results import build_cluster_profiles

    terminology_labels = load_terminology_labels()
    exclude_codes, _ = load_indicative_codes(enabled=not no_indicative)
    df = load_and_prepare_dataframe(
        file_path=dataset,
        terminology_labels=terminology_labels,
        exclude_codes=exclude_codes,
        year_min=year_min,
        year_max=year_max,
    )

    assignments = fetch_cluster_assignments(run_id)
    common_ids = assignments["terminology_id"].to_list()
    fused_labels = assignments["cluster_id"].to_numpy()

    build_cluster_profiles(run_id=run_id, df=df, common_ids=common_ids, fused_labels=fused_labels)
    typer.secho(f"✓ Profiles built for run {run_id}.", fg=typer.colors.GREEN)


@results_app.command("llm-descriptions")
def results_llm_descriptions(
    run_id: int = typer.Argument(..., help="run_id to generate LLM descriptions for"),
):
    """Generate LLM descriptions for all clusters in a run (skips already-described)."""
    from vir.results import populate_llm_descriptions

    populate_llm_descriptions(run_id)
    typer.secho(f"✓ LLM descriptions populated for run {run_id}.", fg=typer.colors.GREEN)


@results_app.command("explain")
def results_explain(
    run_id: int = typer.Argument(..., help="run_id to compute explanations for"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset file path or name"),
    year_min: int = typer.Option(2022, "--year-min", help="Minimum year (inclusive)"),
    year_max: int = typer.Option(2024, "--year-max", help="Maximum year (inclusive)"),
    min_weeks: int = typer.Option(12, "--min-weeks", help="Minimum weeks for temporal view"),
    no_indicative: bool = typer.Option(
        False, "--no-indicative", help="Include all codes (skip indicative filtering)"
    ),
):
    """Compute per-cluster explanations: primary view driver and central code.

    For each cluster stores in cluster_profiles.explanation:
    - view_similarity: mean intra-cluster similarity per view (clinical, demographic, temporal)
    - primary_driver: which view contributed most to this grouping
    - central_code: most representative code (closest to all others in the cluster)

    Requires build-profiles to have been run first.
    """

    from vir.db import fetch_cluster_assignments
    from vir.notebook_widgets import (
        extract_dataset_name,
        load_and_prepare_dataframe,
        load_indicative_codes,
        load_terminology_labels,
    )
    from vir.results import explain_clusters
    from vir.views import ClinicalView, DemographicView, TemporalView

    typer.secho(f"Loading data for run {run_id}...", fg=typer.colors.BLUE)

    terminology_labels = load_terminology_labels()
    exclude_codes, _ = load_indicative_codes(enabled=not no_indicative)
    df = load_and_prepare_dataframe(
        file_path=dataset,
        terminology_labels=terminology_labels,
        exclude_codes=exclude_codes,
        year_min=year_min,
        year_max=year_max,
    )
    dataset_name = extract_dataset_name(dataset)

    # Reconstruct common_ids and fused_labels from DB
    assignments = fetch_cluster_assignments(run_id)
    common_ids = assignments["terminology_id"].to_list()
    fused_labels = assignments["cluster_id"].to_numpy()

    typer.secho("Computing view distance matrices (using cache)...", fg=typer.colors.BLUE)

    n_jobs = get_n_jobs()

    clinical_view = ClinicalView(df=df, dataset_name=dataset_name)
    demographic_view = DemographicView(df=df, dataset_name=dataset_name)
    temporal_view = TemporalView(
        df=df,
        dataset_name=dataset_name,
        params={"min_weeks": min_weeks, "n_jobs": n_jobs},
    )

    clinical_aligned = clinical_view.align_to_common_ids(common_ids)
    demographic_aligned = demographic_view.align_to_common_ids(common_ids)
    temporal_aligned = temporal_view.align_to_common_ids(common_ids)

    typer.secho("Computing explanations...", fg=typer.colors.BLUE)

    explain_clusters(
        run_id=run_id,
        common_ids=common_ids,
        fused_labels=fused_labels,
        clinical_distance=clinical_aligned,
        demographic_distance=demographic_aligned,
        temporal_distance=temporal_aligned,
    )
    typer.secho(f"✓ Explanations computed for run {run_id}.", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
