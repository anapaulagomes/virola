"""Data preparation transformations applied to cleaned interim datasets."""

from typing import List, Optional, Tuple

import polars as pl

from vir.adapters import ADAPTERS
from vir.data_filters import filter_skip_patterns


def _default_skip_patterns() -> list[tuple[str, str]]:
    """Union of every registered adapter's ``skip_patterns``.

    The framework default is the union because each pattern is ``code_type``-scoped:
    DATASUS' ``("cid10", "^Z")`` never fires on an AKTIN row (whose ``code_type``
    is ``icd10gm``) and vice versa. When the call site knows which adapter is
    active it should pass ``adapter.skip_patterns`` explicitly.
    """
    return [pattern for adapter in ADAPTERS.values() for pattern in adapter.skip_patterns]


def prepare_dataframe(
    dataframe: pl.DataFrame,
    terminology_labels: pl.DataFrame,
    exclude_codes: Optional[List[str]] = None,
    skip_patterns: Optional[List[Tuple]] = None,
    keep_city: bool = False,
) -> pl.DataFrame:
    """
    Apply the non-parametric transformations to a freshly cleaned interim dataframe.

    Steps:
    - Select the canonical input columns (year, week, sex, age_group, code_type, code,
      quantity, plus city when ``keep_city=True``). Adapter-specific extras carried
      in the interim parquet are dropped here.
    - Aggregate quantities by (year, week, sex, age_group, code_type, code)
      If ``keep_city=True``: aggregates by (city, year, week, sex, age_group, code_type, code)
    - Normalise code_type values (cid → cid10, ciap → ciap2)
    - Join with terminology labels
    - Exclude non-indicative codes (when provided)
    - Filter skip patterns (e.g. ICD-10 Z-chapter)

    This function is intentionally year-agnostic so the result can be saved to
    data/processed/ and re-used by views that apply different year ranges.

    Args:
        dataframe: Interim DataFrame loaded from data/interim/
        terminology_labels: DataFrame with id, code, code_type, label columns
        exclude_codes: Optional list of labels to exclude (format: ``code:code_type``)
        skip_patterns: ``(code_type, regex)`` pairs to drop. ``None`` falls back to
            the union of every registered adapter's ``skip_patterns``; pass ``[]``
            to disable all pattern filters.
        keep_city: If True, retain the ``city`` column (used for cross-city analysis)

    Returns:
        Prepared DataFrame with columns:
        year, week, sex, age_group, code_type, code, quantity,
        terminology_id, terminology_label

        If ``keep_city=True``, includes the ``city`` column as well.

    Raises:
        polars.exceptions.ColumnNotFoundError: if the interim dataframe is missing
            any canonical input column.
    """
    grouping_columns = ["year", "week", "sex", "age_group", "code_type", "code"]
    if keep_city:
        grouping_columns = ["city", *grouping_columns]

    dataframe = dataframe.select([*grouping_columns, "quantity"])

    dataframe = dataframe.group_by(grouping_columns).agg(pl.col("quantity").sum())

    dataframe = dataframe.with_columns(
        code_type=pl.col("code_type").str.to_lowercase().replace({"cid": "cid10", "ciap": "ciap2"})
    )
    terminology_join = terminology_labels.select(["id", "code", "code_type", "label"])
    dataframe = dataframe.join(terminology_join, on=["code", "code_type"], how="left").rename(
        {"id": "terminology_id", "label": "terminology_label"}
    )

    if exclude_codes:
        dataframe = dataframe.filter(~pl.col("terminology_label").is_in(exclude_codes))

    resolved_patterns = _default_skip_patterns() if skip_patterns is None else skip_patterns
    if resolved_patterns:
        dataframe = filter_skip_patterns(dataframe, resolved_patterns)

    return dataframe
