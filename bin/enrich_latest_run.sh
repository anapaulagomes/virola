#!/usr/bin/env bash
# Run build-profiles, explain, and llm-descriptions for a run.
#
# Usage:
#   bin/enrich_latest_run.sh
#   bin/enrich_latest_run.sh all
#   bin/enrich_latest_run.sh --dataset data/interim/cleaned_...belo_horizonte.parquet
#   bin/enrich_latest_run.sh --run-id 7
#   bin/enrich_latest_run.sh --run-id 7 --dataset data/interim/cleaned_...caucaia.parquet
#   bin/enrich_latest_run.sh --year-min 2022 --year-max 2024
#
# When "all" is passed as the first argument, enrich the latest run for every
# city that has at least one run in the database.
#
# Run selection (in order of precedence):
#   --run-id   use the specified run
#   --dataset  infer dataset_name from the filename and pick the latest run for that dataset
#   (neither)  pick the latest run overall
#
# If --dataset is omitted the script infers the path from the dataset_name
# stored in the run record by globbing data/processed/ for a file
# matching prepared_<dataset_name>.parquet.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "${DATA_DIR:-}" && -f "$PROJ_ROOT/.env" ]]; then
    eval "$(grep -E '^DATA_DIR=' "$PROJ_ROOT/.env")"
    export DATA_DIR
fi

DATA_BASE_DIR="${DATA_DIR:-$PROJ_ROOT/data}"

enrich_run() {
    local DATASET=""
    local RUN_ID_ARG=""
    local YEAR_MIN=""
    local YEAR_MAX=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dataset|-d) DATASET="$2"; shift 2 ;;
            --run-id|-r)  RUN_ID_ARG="$2"; shift 2 ;;
            --year-min)   YEAR_MIN="$2"; shift 2 ;;
            --year-max)   YEAR_MAX="$2"; shift 2 ;;
            *) echo "Unknown argument: $1" >&2; return 1 ;;
        esac
    done

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

    if [[ -z "$DATASET" ]]; then
        PROCESSED_DIR="$DATA_BASE_DIR/processed"
        DATASET=$(find "$PROCESSED_DIR" -maxdepth 1 -name "prepared_${DATASET_NAME}.parquet" | sort | tail -1)
        if [[ -z "$DATASET" ]]; then
            echo "✗ No prepared dataset file found for '${DATASET_NAME}' in ${PROCESSED_DIR}. Run 'vir prepare process' first or pass --dataset explicitly." >&2
            return 1
        fi
        echo "Dataset:     $DATASET"
    fi

    if [[ ! -f "$DATASET" ]]; then
        echo "✗ Dataset file not found: $DATASET" >&2
        return 1
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
}

if [[ "${1:-}" == "all" ]]; then
    shift
    EXTRA_ARGS=("$@")

    RUNS=$(uv run --active python -c "
from vir.db import fetch_runs
runs = fetch_runs()
latest = runs.group_by('city').first()
for row in latest.iter_rows(named=True):
    print(f\"{row['run_id']}|{row['city']}\")
" 2>/dev/null)

    if [[ -z "$RUNS" ]]; then
        echo "✗ No runs found in the database." >&2
        exit 1
    fi

    FAILED=()
    while IFS='|' read -r run_id city; do
        [[ -z "$run_id" ]] && continue
        echo ""
        echo "============================================================"
        echo "Enriching run ${run_id} for: $city"
        echo "============================================================"
        if enrich_run --run-id "$run_id" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; then
            echo "✓ $city done"
        else
            echo "✗ $city failed" >&2
            FAILED+=("$city")
        fi
    done <<< "$RUNS"

    if [[ ${#FAILED[@]} -gt 0 ]]; then
        echo "" >&2
        echo "Failed cities: ${FAILED[*]}" >&2
        exit 1
    fi
    echo ""
    echo "All cities enriched successfully."
else
    enrich_run "$@"
fi
