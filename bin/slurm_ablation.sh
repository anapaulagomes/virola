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

# Usage:
#   sbatch bin/slurm_ablation.sh
#   sbatch --export=ALL,EXTRA_ARGS="--permutations 5000" bin/slurm_ablation.sh
#
# The --array range must match the number of lines in bin/cities.txt.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CITIES_FILE="$SCRIPT_DIR/cities.txt"

export NUMBER_OF_THREADS="${SLURM_CPUS_PER_TASK:-4}"

mkdir -p "$PROJ_ROOT/logs"
mkdir -p "$PROJ_ROOT/data/validation/ablation"

cd "$PROJ_ROOT"

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
