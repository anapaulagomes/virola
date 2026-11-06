#!/usr/bin/env bash
# Usage: vir/adapters/datasus/bin/run_pipeline.sh <city|all> [extra vir pipeline run flags...]
#
# Examples:
#   vir/adapters/datasus/bin/run_pipeline.sh all
#   vir/adapters/datasus/bin/run_pipeline.sh "Porto Alegre"
#   vir/adapters/datasus/bin/run_pipeline.sh all --skip-clean --n-clusters 150
#   vir/adapters/datasus/bin/run_pipeline.sh "Manaus" --year-min 2023

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
CITIES_FILE="$SCRIPT_DIR/cities.txt"

if [[ -f "$PROJ_ROOT/.env" ]]; then
    set -a
    source "$PROJ_ROOT/.env"
    set +a
fi
source "${UV_PROJECT_ENVIRONMENT:-$PROJ_ROOT/.venv}/bin/activate"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <city|all> [extra flags...]" >&2
    echo "Cities available in $CITIES_FILE:" >&2
    cat "$CITIES_FILE" >&2
    exit 1
fi

TARGET="$1"
shift
EXTRA_ARGS=("$@")

run_city() {
    local city="$1"
    echo ""
    echo "============================================================"
    echo "Running pipeline for: $city"
    echo "============================================================"
    uv run --active vir pipeline run "$city" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
}

if [[ "$TARGET" == "all" ]]; then
    FAILED=()
    while IFS= read -r city || [[ -n "$city" ]]; do
        [[ -z "$city" ]] && continue
        if run_city "$city"; then
            echo "✓ $city done"
        else
            echo "✗ $city failed" >&2
            FAILED+=("$city")
        fi
    done < "$CITIES_FILE"

    if [[ ${#FAILED[@]} -gt 0 ]]; then
        echo "" >&2
        echo "Failed cities: ${FAILED[*]}" >&2
        exit 1
    fi
    echo ""
    echo "All cities completed successfully."
else
    if ! grep -qxF "$TARGET" "$CITIES_FILE"; then
        echo "Error: '$TARGET' not found in $CITIES_FILE" >&2
        echo "Available cities:" >&2
        cat "$CITIES_FILE" >&2
        exit 1
    fi
    run_city "$TARGET"
fi
