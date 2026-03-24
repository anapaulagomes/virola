"""Widgets and data loading functions for notebooks."""

from typing import List, Optional, Tuple

import marimo as mo
import numpy as np
import polars as pl
from snf import compute as snf_compute

from vir.config import (
    INDICATIVE_CODES_FILE,
    INDICATIVE_ENABLED,
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
    enabled: bool = INDICATIVE_ENABLED,
) -> Tuple[List[str], bool]:
    """
    Load non-indicative codes to be excluded from analysis.

    Args:
        enabled: Whether to apply indicative filtering (default True).
                 Pass False to skip filtering (include all codes).

    Returns:
        Tuple of (list of code labels to exclude, enabled flag).
        When enabled=False, returns ([], False) meaning no filter is applied.
    """
    if not enabled:
        return [], False

    indicative_codes = pl.read_csv(INDICATIVE_CODES_FILE)
    exclude_codes = [
        f"{row['code']}:{row['code_type']}"
        for row in indicative_codes.filter(pl.col("indicative") == "no").iter_rows(named=True)
    ]

    return exclude_codes, True


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
    exclude_codes: Optional[List[str]] = None,
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
    - Exclude non-indicative codes (if provided)

    Args:
        file_path: Path to parquet file
        terminology_labels: DataFrame with terminology labels
        exclude_codes: Optional list of code labels to exclude (format: "code:code_type")
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

    print(f"Raw: {df.shape}")

    # filter by year range
    if year_min is not None and year_max is not None:
        df = df.filter(pl.col("year").is_between(year_min, year_max))
    elif year_min is not None:
        df = df.filter(pl.col("year").ge(year_min))
    elif year_max is not None:
        df = df.filter(pl.col("year").le(year_max))

    print(f"After temporal filter: {df.shape}")

    # guarantee that there are no orphans from dropped columns
    df = df.group_by(["year", "week", "sex", "age_group", "code_type", "code"]).agg(
        pl.col("quantity").sum()
    )
    print(f"After quantity aggregation: {df.shape}")

    df = df.with_columns(
        code_type=pl.col("code_type").str.to_lowercase().replace({"cid": "cid10", "ciap": "ciap2"})
    )
    df = df.join(terminology_labels, on=["code", "code_type"], how="left").rename(
        {"id": "terminology_id", "label": "terminology_label"}
    )
    print(f"After joining terminologies: {df.shape}")

    # exclude non-indicative codes
    if exclude_codes:
        df = df.filter(~pl.col("terminology_label").is_in(exclude_codes))

    print(f"After exclusion of non-indicative codes: {df.shape}")

    # ensure temporal representation
    df = filter_min_temporal_coverage(df)
    print(f"After ensuring temporal representation: {df.shape}")

    return df


def normalize_distance(D: np.ndarray) -> np.ndarray:
    """Normalise a distance matrix to [0, 1] with zero diagonal.

    Used before computing affinity matrices so that all views enter SNF
    on comparable scales regardless of their natural distance range.
    """
    D = D.copy()
    np.fill_diagonal(D, 0.0)
    max_val = D.max()
    return D / max_val if max_val > 0 else D


def distance_to_affinity(dist: np.ndarray, K: int = 20, mu: float = 0.5) -> np.ndarray:
    """Convert a distance matrix to a normalised affinity matrix.

    Applies SNF's adaptive Gaussian kernel (local scaling based on K-NN
    neighbourhood) then normalises the result to [0, 1] so all views
    contribute equally to fusion.

    Args:
        dist: Square distance matrix.
        K: Number of nearest neighbours for local scale estimation.
        mu: Scaling factor for the adaptive bandwidth.

    Returns:
        Square affinity matrix with zero diagonal, values in [0, 1].
    """
    W = snf_compute.affinity_matrix(normalize_distance(dist), K=K, mu=mu)
    np.fill_diagonal(W, 0.0)
    max_val = W.max()
    return W / max_val if max_val > 0 else W
