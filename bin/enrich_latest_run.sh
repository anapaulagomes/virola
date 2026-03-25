#!/usr/bin/env bash
# Run build-profiles, explain, and llm-descriptions for a run.
#
# Usage:
#   bin/enrich_latest_run.sh
#   bin/enrich_latest_run.sh --dataset data/interim/cleaned_...belo_horizonte.parquet
#   bin/enrich_latest_run.sh --run-id 7
#   bin/enrich_latest_run.sh --run-id 7 --dataset data/interim/cleaned_...caucaia.parquet
#   bin/enrich_latest_run.sh --year-min 2022 --year-max 2024
#
# Run selection (in order of precedence):
#   --run-id   use the specified run
#   --dataset  infer dataset_name from the filename and pick the latest run for that dataset
#   (neither)  pick the latest run overall
#
# If --dataset is omitted the script infers the path from the dataset_name
# stored in the run record by globbing data/interim/ for a file
# matching *municipios_<dataset_name>.parquet.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET=""
RUN_ID_ARG=""
YEAR_MIN=""
YEAR_MAX=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset|-d) DATASET="$2"; shift 2 ;;
        --run-id|-r)  RUN_ID_ARG="$2"; shift 2 ;;
        --year-min)   YEAR_MIN="$2"; shift 2 ;;
        --year-max)   YEAR_MAX="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Fetch run metadata from the DB — one field per line to avoid
# whitespace-splitting issues with ANSI output from loguru/tqdm.
{
    IFS= read -r RUN_ID
    IFS= read -r STORED_YEAR_MIN
    IFS= read -r STORED_YEAR_MAX
    IFS= read -r DATASET_NAME
} < <(RUN_ID_ARG="$RUN_ID_ARG" DATASET="$DATASET" uv run --active python -c "
import os; from vir.db import find_run
run_id = int(os.environ['RUN_ID_ARG']) if os.environ.get('RUN_ID_ARG') else None
row = find_run(run_id=run_id, dataset_path=os.environ.get('DATASET') or None)
print(row['run_id']); print(row['year_min']); print(row['year_max']); print(row['dataset_name'])
" 2>/dev/null)

echo "Latest run: id=${RUN_ID}  dataset=${DATASET_NAME}  years=${STORED_YEAR_MIN}-${STORED_YEAR_MAX}"

YEAR_MIN="${YEAR_MIN:-$STORED_YEAR_MIN}"
YEAR_MAX="${YEAR_MAX:-$STORED_YEAR_MAX}"

# Infer dataset path from the name stored in the run if not provided
if [[ -z "$DATASET" ]]; then
    DATASET=$(DATASET_NAME="$DATASET_NAME" PROJ_ROOT="$PROJ_ROOT" uv run --active python -c "
import os
from pathlib import Path
interim = Path(os.environ['PROJ_ROOT']) / 'data' / 'interim'
matches = sorted(interim.glob(f\"*municipios_{os.environ['DATASET_NAME']}.parquet\"))
if not matches:
    raise SystemExit(f\"No dataset file found for '{os.environ['DATASET_NAME']}' in {interim}. Pass --dataset explicitly.\")
print(matches[-1])
")
    echo "Dataset:     $DATASET"
fi

if [[ ! -f "$DATASET" ]]; then
    echo "✗ Dataset file not found: $DATASET" >&2
    exit 1
fi

echo ""
echo "1/3 Building cluster profiles..."
uv run --active vir results build-profiles "$RUN_ID" \
    --dataset "$DATASET" --year-min "$YEAR_MIN" --year-max "$YEAR_MAX"

echo ""
echo "2/3 Computing explanations..."
uv run --active vir results explain "$RUN_ID" \
    --dataset "$DATASET" --year-min "$YEAR_MIN" --year-max "$YEAR_MAX"

echo ""
echo "3/3 Generating LLM descriptions..."
uv run --active vir results llm-descriptions "$RUN_ID"

echo ""
echo "✓ Run $RUN_ID enriched."
