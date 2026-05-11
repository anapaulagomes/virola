#!/bin/bash
#SBATCH --job-name=virola_pipeline
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err

# Usage:
#   sbatch bin/slurm_pipeline.sh
#   sbatch --export=ALL,EXTRA_ARGS="--skip-clean" bin/slurm_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CITIES_FILE="$SCRIPT_DIR/cities.txt"

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
