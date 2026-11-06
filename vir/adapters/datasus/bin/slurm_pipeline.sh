#!/bin/bash
#SBATCH --job-name=virola_pipeline
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err

# Usage (from the repo root; Slurm does not create the log directory):
#   mkdir -p logs
#   sbatch vir/adapters/datasus/bin/slurm_pipeline.sh
#   sbatch --export=ALL,EXTRA_ARGS="--skip-clean" vir/adapters/datasus/bin/slurm_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# sbatch runs a spooled copy of this script, so BASH_SOURCE does not point at the repo;
# submit from the repo root and SLURM_SUBMIT_DIR is the project root.
PROJ_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
CITIES_FILE="$PROJ_ROOT/vir/adapters/datasus/bin/cities.txt"

export NUMBER_OF_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# Override data directory for HPC shared filesystem if needed
# export DATA_DIR=/path/to/shared/data

mkdir -p "$PROJ_ROOT/logs"

echo "========================================================"
echo "Job:     ${SLURM_JOB_ID:-local}"
echo "CPUs:    $NUMBER_OF_THREADS"
echo "Node:    ${SLURMD_NODENAME:-local}"
echo "Started: $(date)"
echo "========================================================"

cd "$PROJ_ROOT"

if [[ -f "$PROJ_ROOT/.env" ]]; then
    set -a
    source "$PROJ_ROOT/.env"
    set +a
fi
source "${UV_PROJECT_ENVIRONMENT:-$PROJ_ROOT/.venv}/bin/activate"

FAILED=()

while IFS= read -r city || [[ -n "$city" ]]; do
    [[ -z "$city" ]] && continue

    echo ""
    echo "------------------------------------------------------------"
    echo "Running: $city  ($(date))"
    echo "------------------------------------------------------------"

    if uv run --active vir pipeline run "$city" ${EXTRA_ARGS:-}; then
        echo "✓ $city done"
    else
        echo "✗ $city failed" >&2
        FAILED+=("$city")
    fi
done < "$CITIES_FILE"

echo ""
echo "========================================================"
echo "Finished: $(date)"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed cities: ${FAILED[*]}" >&2
    exit 1
fi

echo "All cities completed successfully."
