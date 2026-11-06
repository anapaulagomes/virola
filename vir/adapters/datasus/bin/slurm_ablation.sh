#!/bin/bash
#SBATCH --job-name=virola_ablation
#SBATCH --partition=main
#SBATCH --array=0-9
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ablation_%A_%a.out
#SBATCH --error=logs/ablation_%A_%a.err

# Usage (from the repo root; Slurm does not create the log directory):
#   mkdir -p logs
#   sbatch vir/adapters/datasus/bin/slurm_ablation.sh
#   sbatch --export=ALL,EXTRA_ARGS="--permutations 5000" vir/adapters/datasus/bin/slurm_ablation.sh
#
# The --array range must match the number of lines in vir/adapters/datasus/bin/cities.txt.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# sbatch runs a spooled copy of this script, so BASH_SOURCE does not point at the repo;
# submit from the repo root and SLURM_SUBMIT_DIR is the project root.
PROJ_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
CITIES_FILE="$PROJ_ROOT/vir/adapters/datasus/bin/cities.txt"

export NUMBER_OF_THREADS="${SLURM_CPUS_PER_TASK:-4}"

mkdir -p "$PROJ_ROOT/logs"
mkdir -p "$PROJ_ROOT/data/validation/ablation"

cd "$PROJ_ROOT"

if [[ -f "$PROJ_ROOT/.env" ]]; then
    set -a
    source "$PROJ_ROOT/.env"
    set +a
fi
source "${UV_PROJECT_ENVIRONMENT:-$PROJ_ROOT/.venv}/bin/activate"

mapfile -t CITIES < "$CITIES_FILE"

if [[ "$SLURM_ARRAY_TASK_ID" -ge "${#CITIES[@]}" ]]; then
    echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID exceeds cities.txt length (${#CITIES[@]})" >&2
    exit 1
fi

CITY="${CITIES[$SLURM_ARRAY_TASK_ID]}"
PERMUTATIONS="${PERMUTATIONS:-1000}"
OUTPUT_DIR="${OUTPUT_DIR:-data/validation/ablation}"

echo "========================================================"
echo "Job:     ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
echo "City:    $CITY"
echo "CPUs:    $NUMBER_OF_THREADS"
echo "Node:    ${SLURMD_NODENAME:-local}"
echo "Started: $(date)"
echo "========================================================"

if uv run --active vir validate ablation \
    --city "$CITY" \
    --permutations "$PERMUTATIONS" \
    --output "$OUTPUT_DIR" \
    ${EXTRA_ARGS:-}; then
    echo "✓ $CITY done at $(date)"
else
    echo "✗ $CITY failed at $(date)" >&2
    exit 1
fi
