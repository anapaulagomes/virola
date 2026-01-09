"""Widgets and data loading functions for notebooks."""

from typing import List, Optional, Tuple

import marimo as mo
import polars as pl

from vir.config import (
    INDICATIVE_ENABLED,
    INDICATIVE_MODEL_RESULTS,
    INDICATIVE_MODEL_THRESHOLD,
    INTERIM_DATA_DIR,
)
from vir.data_filters import filter_min_temporal_coverage
from vir.db import DEFAULT_MODEL, fetch_terminologies


def load_terminology_labels(model: str = DEFAULT_MODEL) -> pl.DataFrame:
    """
    Load terminology labels from database.

    Args:
        model: Model name for embeddings (default: DEFAULT_MODEL)

    Returns:
        DataFrame with id, code, code_type, label columns
    """
    return fetch_terminologies(model)


def load_indicative_codes(
    threshold: float = INDICATIVE_MODEL_THRESHOLD, enabled: bool = INDICATIVE_ENABLED
) -> Tuple[List[str], bool]:
    """
    Load indicative codes from CSV file.

    Args:
        threshold: Probability threshold for filtering codes
        enabled: Whether to filter by indicative codes

    Returns:
        Tuple of (list of code labels, enabled flag)
    """
    include_codes = []
    if not enabled:
        return include_codes, False

    indicative_codes = pl.read_csv(INDICATIVE_MODEL_RESULTS)
    for row in indicative_codes.iter_rows(named=True):
        if row["indicative_prob"] >= threshold:
            include_codes.append(f"{row['code']}:{row['code_type']}")

    return include_codes, enabled


def create_file_dropdown(
    default_pattern: str = "cleaned_aesop_raw_dados_2017_20250223_20municipios_salvador.parquet",
) -> mo.ui.dropdown:
    """
    Create dropdown for selecting dataset files.

    Args:
        default_pattern: Pattern to match for default selection

    Returns:
        Marimo dropdown widget
    """
    all_options = [str(file.resolve()) for file in INTERIM_DATA_DIR.glob("**/cleaned_*.parquet")]
    default = [option for option in all_options if default_pattern in option]

    return mo.ui.dropdown(
        options=all_options,
        label="Choose dataset",
        searchable=True,
        value=default[0] if default else None,
    )


def extract_dataset_name(file_path: str) -> str:
    """
    Extract dataset name from file path.

    Args:
        file_path: Full path to dataset file

    Returns:
        Dataset name (text between "municipios_" and ".parquet")
    """
    if "municipios" in file_path and ".parquet" in file_path:
        start_idx = file_path.find("municipios") + 11  # len("municipios_") = 11
        end_idx = file_path.find(".parquet")
        return file_path[start_idx:end_idx]
    return "unknown"


def load_and_prepare_dataframe(
    file_path: str,
    terminology_labels: pl.DataFrame,
    include_codes: Optional[List[str]] = None,
    year_min: Optional[int] = 2022,
    year_max: Optional[int] = 2024,
) -> pl.DataFrame:
    """
    Load and prepare dataframe for view analysis.

    Standard preprocessing:
    - Drop unnecessary columns (unique_id, state, city, region, population_range)
    - Group by year, week, sex, age_group, code_type, code
    - Join with terminology labels
    - Filter by year range
    - Filter by indicative codes (if provided)

    Args:
        file_path: Path to parquet file
        terminology_labels: DataFrame with terminology labels
        include_codes: Optional list of codes to filter (format: "code:code_type")
        year_min: Minimum year (inclusive), None for no filter
        year_max: Maximum year (inclusive), None for no filter

    Returns:
        Prepared DataFrame with columns:
        - year, week, sex, age_group, code_type, code, quantity
        - terminology_id, terminology_label
    """
    df = pl.read_parquet(file_path)

    # drop unnecessary columns
    df = df.drop(["unique_id", "state", "city", "region", "population_range"])

    # guarantee that there are no orphans from dropped columns
    df = df.group_by(["year", "week", "sex", "age_group", "code_type", "code"]).agg(
        pl.col("quantity").sum()
    )

    df = df.join(terminology_labels, on=["code", "code_type"]).rename(
        {"id": "terminology_id", "label": "terminology_label"}
    )

    # filter by year range
    if year_min is not None and year_max is not None:
        df = df.filter(pl.col("year").is_between(year_min, year_max))
    elif year_min is not None:
        df = df.filter(pl.col("year").ge(year_min))
    elif year_max is not None:
        df = df.filter(pl.col("year").le(year_max))

    # filter by indicative codes
    if include_codes:
        df = df.filter(pl.col("terminology_label").is_in(include_codes))

    # ensure temporal representation
    df = filter_min_temporal_coverage(df)

    return df
