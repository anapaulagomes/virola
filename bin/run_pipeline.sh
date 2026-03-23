#!/usr/bin/env bash
# Usage: bin/run_pipeline.sh <city|all> [extra vir pipeline run flags...]
#
# Examples:
#   bin/run_pipeline.sh all
#   bin/run_pipeline.sh "Porto Alegre"
#   bin/run_pipeline.sh all --skip-clean --n-clusters 150
#   bin/run_pipeline.sh "Manaus" --year-min 2023

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CITIES_FILE="$SCRIPT_DIR/cities.txt"

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
