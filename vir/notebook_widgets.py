"""Widgets and data loading functions for notebooks."""

from pathlib import Path
from typing import List, Optional, Tuple

import marimo as mo
import numpy as np
import polars as pl
from snf import compute as snf_compute

from vir.config import (
    DEFAULT_MODEL,
    INDICATIVE_ENABLED,
    PROCESSED_DATA_DIR,
)
from vir.data_filters import filter_min_temporal_coverage
from vir.db import fetch_terminologies

_PATH_NOT_PROVIDED = object()


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
    path: Optional[Path] = _PATH_NOT_PROVIDED,
    enabled: bool = INDICATIVE_ENABLED,
) -> Tuple[List[str], bool]:
    """Load non-indicative codes to be excluded from analysis.

    Args:
        path: Path to the labeling CSV. Pass ``adapter.indicative_codes_file``
            from the active adapter — that is ``None`` for adapters without a
            labeling CSV (e.g. AKTIN), in which case filtering is skipped.
            Required: omitting the argument raises ``ValueError`` so adapter
            context can't be silently dropped.
        enabled: When ``False`` filtering is unconditionally disabled.

    Returns:
        Tuple of ``(list of code labels to exclude, enabled flag)``.
        Returns ``([], False)`` when ``enabled=False`` or ``path is None``.

    Raises:
        ValueError: if ``enabled=True`` and ``path`` is omitted. The caller must
            be explicit about which adapter's labeling CSV to read.
    """
    if not enabled:
        return [], False

    if path is _PATH_NOT_PROVIDED:
        raise ValueError(
            "load_indicative_codes(): pass `path=adapter.indicative_codes_file` "
            "or an explicit CSV path."
        )

    if path is None:
        return [], False

    indicative_codes = pl.read_csv(path)
    exclude_codes = [
        f"{row['code']}:{row['code_type']}"
        for row in indicative_codes.filter(pl.col("indicative").eq("no")).iter_rows(named=True)
    ]

    return exclude_codes, True


def create_file_dropdown(
    default_pattern: str = "prepared_base_aesop_raw_extracao_01042026_salvador.parquet",
) -> mo.ui.dropdown:
    """
    Create dropdown for selecting prepared dataset files.

    Args:
        default_pattern: Pattern to match for default selection

    Returns:
        Marimo dropdown widget
    """
    all_options = [
        str(file.resolve()) for file in PROCESSED_DATA_DIR.glob("**/prepared_*.parquet")
    ]
    default = [option for option in all_options if default_pattern in option]

    return mo.ui.dropdown(
        options=all_options,
        label="Choose dataset",
        searchable=True,
        value=default[0] if default else None,
    )


def load_and_prepare_dataframe(
    file_path: str,
    year_min: Optional[int] = 2022,
    year_max: Optional[int] = 2024,
) -> pl.DataFrame:
    """
    Load a prepared dataframe and apply year filtering and temporal coverage.

    Reads a pre-processed parquet file from data/processed/ (produced by the
    ``vir prepare process`` step) and applies the two parametric filters that
    depend on the user's year selection:

    - Year range filter (year_min / year_max)
    - Minimum temporal coverage (filter_min_temporal_coverage)

    Args:
        file_path: Path to a prepared parquet file
        year_min: Minimum year (inclusive), None for no lower bound
        year_max: Maximum year (inclusive), None for no upper bound

    Returns:
        Filtered DataFrame ready for view analysis
    """
    dataframe = pl.read_parquet(file_path)

    if year_min is not None and year_max is not None:
        dataframe = dataframe.filter(pl.col("year").is_between(year_min, year_max))
    elif year_min is not None:
        dataframe = dataframe.filter(pl.col("year").ge(year_min))
    elif year_max is not None:
        dataframe = dataframe.filter(pl.col("year").le(year_max))

    dataframe = filter_min_temporal_coverage(dataframe)

    return dataframe


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
