"""Generate the top-volume codes CSV for an epidemiologist to label.

Adapter-agnostic: operates on the canonical row (cleaned interim parquet shape).
"""

import polars as pl

from vir.data_filters import filter_most_representative_rows


def build_indicative_template(
    cleaned: pl.DataFrame,
    terminology_labels: pl.DataFrame,
    existing_labels: pl.DataFrame | None = None,
    coverage: float = 0.9,
) -> pl.DataFrame:
    """Return the top-volume codes covering ``coverage`` of total volume.

    Each row carries the code identifier, its human-readable description (left-joined
    from ``terminology_labels``), the cumulative-volume percentile, and an
    ``indicative`` column to be filled in by an epidemiologist. When
    ``existing_labels`` is provided, prior annotations are preserved on a
    left-join.

    Output schema: ``code, code_type, description, indicative, quantity, cum_pct``.
    Sorted by ``quantity`` descending.
    """
    top = filter_most_representative_rows(
        cleaned, group_by=["code", "code_type"], threshold=coverage
    )

    descriptions = terminology_labels.select(["code", "code_type", "description"])
    annotated = top.join(descriptions, on=["code", "code_type"], how="left")

    if existing_labels is not None:
        labels = existing_labels.select(["code", "code_type", "indicative"])
        annotated = annotated.join(labels, on=["code", "code_type"], how="left")
    else:
        annotated = annotated.with_columns(indicative=pl.lit(None, dtype=pl.Utf8))

    return annotated.select(
        "code", "code_type", "description", "indicative", "quantity", "cum_pct"
    )
