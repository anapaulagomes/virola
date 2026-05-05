"""Data preparation transformations applied to cleaned interim datasets."""

from typing import List, Optional, Tuple

import polars as pl

from vir.config import SKIP_PATTERNS
from vir.data_filters import filter_skip_patterns


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
    - Drop administrative columns (unique_id, state, region, population_range)
      Note: ``city`` is dropped by default, set ``keep_city=True`` to retain it
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
        skip_patterns: Override SKIP_PATTERNS; pass [] to disable all pattern filters
        keep_city: If True, retain the ``city`` column (used for cross-city analysis)

    Returns:
        Prepared DataFrame with columns:
        year, week, sex, age_group, code_type, code, quantity,
        terminology_id, terminology_label

        If ``keep_city=True``, includes the ``city`` column as well.
    """
    columns_to_drop = ["unique_id", "state", "region", "population_range"]
    if not keep_city:
        columns_to_drop.append("city")
    dataframe = dataframe.drop([c for c in columns_to_drop if c in dataframe.columns])

    group_by_cols = ["year", "week", "sex", "age_group", "code_type", "code"]
    if keep_city:
        group_by_cols.insert(0, "city")
    dataframe = dataframe.group_by(group_by_cols).agg(pl.col("quantity").sum())

    dataframe = dataframe.with_columns(
        code_type=pl.col("code_type").str.to_lowercase().replace({"cid": "cid10", "ciap": "ciap2"})
    )
    dataframe = dataframe.join(terminology_labels, on=["code", "code_type"], how="left").rename(
        {"id": "terminology_id", "label": "terminology_label"}
    )

    if exclude_codes:
        dataframe = dataframe.filter(~pl.col("terminology_label").is_in(exclude_codes))

    resolved_patterns = SKIP_PATTERNS if skip_patterns is None else skip_patterns
    if resolved_patterns:
        dataframe = filter_skip_patterns(dataframe, resolved_patterns)

    return dataframe
