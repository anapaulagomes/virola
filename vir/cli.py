"""Main CLI for Virola - Multi-view clustering for epidemiological surveillance."""

import subprocess
import sys
from pathlib import Path
from typing import Optional

import polars as pl
import typer

from vir.config import (
    AFFINITY_ZSCORE_PERMS,
    CLUSTERING_DIR,
    DEFAULT_MODEL,
    INTERIM_DATA_DIR,
    PROJ_ROOT,
    VALIDATION_DIR,
)
from vir.dataset import app as helper_app
from vir.db import (
    delete_run,
    fetch_cluster_assignments,
    fetch_runs,
)
from vir.features import CITY_NAMES
from vir.helpers import extract_dataset_name, get_n_jobs
from vir.notebook_widgets import load_and_prepare_dataframe
from vir.views import ClinicalView, DemographicView, TemporalView

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
validate_app = typer.Typer(help="Validation commands")

app.add_typer(setup_app, name="setup")
app.add_typer(prepare_app, name="prepare")
app.add_typer(view_app, name="view")
app.add_typer(model_app, name="model")
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(results_app, name="results")
app.add_typer(validate_app, name="validate")
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
    force: bool = typer.Option(
        False, "--force", help="Reprocess from raw even if cached output exists"
    ),
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
    if force:
        args.append("--force")

    exit_code = run_marimo_notebook(notebook, args)
    if exit_code == 0:
        typer.secho("✓ Data cleaning completed successfully", fg=typer.colors.GREEN)
    sys.exit(exit_code)


@prepare_app.command("process")
def prepare_process(
    dataset: str = typer.Option(
        ..., "--dataset", "-d", help="Path to a cleaned interim parquet file"
    ),
):
    """Prepare a cleaned interim dataset: aggregate, join terminology, filter codes."""
    notebook = PROJ_ROOT / "vir/modeling/data_preparation.py"
    args = ["--dataset", dataset]

    exit_code = run_marimo_notebook(notebook, args)
    if exit_code == 0:
        typer.secho("✓ Data preparation completed successfully", fg=typer.colors.GREEN)
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
    """Run complete pipeline for a city: clean -> process -> views -> model snf."""

    if city not in CITY_NAMES:
        typer.secho(
            f"✗ Invalid city name. Available cities: {', '.join(CITY_NAMES)}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    typer.secho(f"\n{'=' * 60}", fg=typer.colors.CYAN)
    typer.secho(f"Running pipeline for: {city}", fg=typer.colors.CYAN, bold=True)
    typer.secho(f"{'=' * 60}\n", fg=typer.colors.CYAN)

    city_slug = city.lower().replace(" ", "_")

    # Step 1: Clean data
    if not skip_clean:
        typer.secho("Step 1/6: Cleaning data...", fg=typer.colors.BLUE, bold=True)
        notebook = PROJ_ROOT / "vir/modeling/data_cleaning.py"
        args = ["--city", city]
        exit_code = run_marimo_notebook(notebook, args)
        if exit_code != 0:
            typer.secho("✗ Data cleaning failed", fg=typer.colors.RED)
            sys.exit(exit_code)
        typer.secho("✓ Data cleaning completed\n", fg=typer.colors.GREEN)
    else:
        typer.secho("Step 1/6: Skipping data cleaning\n", fg=typer.colors.YELLOW)

    # Step 2: Process data (terminology join + code filtering)
    matching_interim = sorted(INTERIM_DATA_DIR.glob(f"cleaned_*{city_slug}*.parquet"))
    if not matching_interim:
        typer.secho(
            f"✗ No cleaned interim file found for city '{city}' in {INTERIM_DATA_DIR}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    interim_file = matching_interim[-1]

    typer.secho("Step 2/6: Processing data...", fg=typer.colors.BLUE, bold=True)
    notebook = PROJ_ROOT / "vir/modeling/data_preparation.py"
    args = ["--dataset", str(interim_file)]
    exit_code = run_marimo_notebook(notebook, args)
    if exit_code != 0:
        typer.secho("✗ Data preparation failed", fg=typer.colors.RED)
        sys.exit(exit_code)
    typer.secho("✓ Data preparation completed\n", fg=typer.colors.GREEN)

    # Construct dataset pattern based on city for view notebooks
    dataset = city_slug

    indicative_args = ["--no-indicative"] if no_indicative else []

    # Step 3: Clinical view
    typer.secho("Step 3/6: Running clinical view analysis...", fg=typer.colors.BLUE, bold=True)
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

    # Step 4: Demographic view
    typer.secho("Step 4/6: Running demographic view analysis...", fg=typer.colors.BLUE, bold=True)
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

    # Step 5: Temporal view
    typer.secho("Step 5/6: Running temporal view analysis...", fg=typer.colors.BLUE, bold=True)
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

    # Step 6: SNF clustering
    typer.secho("Step 6/6: Running SNF clustering...", fg=typer.colors.BLUE, bold=True)
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
def results_list(
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Maximum rows to show (default: all)"
    ),
):
    """List all saved runs."""

    runs = fetch_runs()
    if runs.is_empty():
        typer.secho("No runs found.", fg=typer.colors.YELLOW)
        return

    if limit is not None:
        runs = runs.head(limit)

    with pl.Config(tbl_rows=len(runs)):
        typer.echo(runs)


@results_app.command("build-profiles")
def results_build_profiles(
    run_id: int = typer.Argument(..., help="run_id to build profiles for"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset file path or name"),
    year_min: int = typer.Option(2022, "--year-min", help="Minimum year (inclusive)"),
    year_max: int = typer.Option(2024, "--year-max", help="Maximum year (inclusive)"),
):
    """Build cluster profiles (stats + temporal + demographic) for a run."""
    from vir.results import build_cluster_profiles

    df = load_and_prepare_dataframe(
        file_path=dataset,
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
):
    """Compute per-cluster explanations: primary view driver and central code.

    For each cluster stores in cluster_profiles.explanation:
    - view_similarity: mean intra-cluster similarity per view (clinical, demographic, temporal)
    - primary_driver: which view contributed most to this grouping
    - central_code: most representative code (closest to all others in the cluster)

    Requires build-profiles to have been run first.
    """

    from vir.results import explain_clusters

    typer.secho(f"Loading data for run {run_id}...", fg=typer.colors.BLUE)

    df = load_and_prepare_dataframe(
        file_path=dataset,
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


@results_app.command("delete")
def results_delete(
    run_id: Optional[int] = typer.Option(None, "--id", help="run_id to delete"),
    city: Optional[str] = typer.Option(
        None, "--city", help=f"Delete all runs for a city: {', '.join(CITY_NAMES)}"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Delete a run by id, or all runs for a city.

    Removes cluster assignments, profiles, multi-K and Leiden sweep rows,
    plus the parquet sidecar in data/runs/.
    """
    if run_id is None and city is None:
        typer.secho("✗ Provide --id or --city.", fg=typer.colors.RED)
        raise typer.Exit(1)
    if run_id is not None and city is not None:
        typer.secho("✗ --id and --city are mutually exclusive.", fg=typer.colors.RED)
        raise typer.Exit(1)
    if city is not None and city not in CITY_NAMES:
        typer.secho(f"✗ Invalid city. Available: {', '.join(CITY_NAMES)}", fg=typer.colors.RED)
        raise typer.Exit(1)

    if city is not None:
        target_run_ids = fetch_runs().filter(pl.col("city") == city).get_column("run_id").to_list()
    else:
        target_run_ids = [run_id]

    if not target_run_ids:
        typer.secho("No runs to delete.", fg=typer.colors.YELLOW)
        return

    typer.echo(f"Will delete {len(target_run_ids)} run(s): {target_run_ids}")
    if not yes and not typer.confirm("Proceed?", default=False):
        typer.secho("Aborted.", fg=typer.colors.YELLOW)
        return

    deleted_count = 0
    not_found = []
    for target_id in target_run_ids:
        if delete_run(target_id):
            deleted_count += 1
        else:
            not_found.append(target_id)

    if not_found:
        typer.secho(f"Run(s) not found: {not_found}", fg=typer.colors.YELLOW)
    typer.secho(f"✓ Deleted {deleted_count} run(s).", fg=typer.colors.GREEN)


@validate_app.command("ablation")
def validate_ablation(
    city: Optional[str] = typer.Option(
        None, "--city", help=f"City name to run ablation for: {', '.join(CITY_NAMES)}"
    ),
    permutations: int = typer.Option(
        AFFINITY_ZSCORE_PERMS,
        "--permutations",
        help=f"Permutations for affinity Z-score (default: {AFFINITY_ZSCORE_PERMS})",
    ),
    output_directory: Optional[Path] = typer.Option(
        VALIDATION_DIR / "ablation",
        "--output",
        help=f"Output directory (default: {VALIDATION_DIR / 'ablation'})",
    ),
):
    """Run ablation study across selected cities."""
    notebook = PROJ_ROOT / "notebooks/validation/validation_ablation.py"
    args = []
    if city:
        args.extend(["--city", city])
    if permutations:
        args.extend(["--permutations", str(permutations)])
    if output_directory:
        args.extend(["--output", str(output_directory)])

    exit_code = run_marimo_notebook(notebook, args)
    if exit_code == 0:
        typer.secho("✓ Ablation study complete.", fg=typer.colors.GREEN)
    sys.exit(exit_code)


if __name__ == "__main__":
    app()
