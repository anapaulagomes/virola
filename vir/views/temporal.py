"""Temporal view: time series similarity using MSM distance."""

from functools import cached_property

import numpy as np
import polars as pl
from aeon.distances import pairwise_distance

from vir.config import MIN_WEEKS
from vir.views.base import BaseView


class TemporalView(BaseView):
    """
    Temporal view: captures temporal patterns and cyclic behavior of diagnoses over time.

    - Input: Diagnosis × week matrix (aggregated time series)
    - Preprocessing: Filter for min_weeks coverage, fill missing weeks with 0
    - Distance metric: MSM (Move-Split-Merge) distance
    - Output: Distance matrix

    The MSM distance is appropriate for comparing time series with elastic matching.
    """

    view_type = "temporal"

    @cached_property
    def _pivot_df(self) -> pl.DataFrame:
        """
        Create and cache the diagnosis × week pivot table.

        Only includes diagnoses that appear in at least min_weeks.

        Returns:
            Polars DataFrame with terminology_id and weekly columns
        """
        min_weeks = self.params.get("min_weeks", MIN_WEEKS)

        # Filter for diagnoses with sufficient temporal coverage
        more_than_min_weeks = (
            self.df.group_by("terminology_id")
            .agg(pl.col("week").count().alias("week_count"))
            .filter(pl.col("week_count").ge(min_weeks))
        )

        # Aggregate by week
        df_week = (
            self.df.filter(pl.col("terminology_id").is_in(more_than_min_weeks["terminology_id"]))
            .group_by(["terminology_id", "week"])
            .agg(pl.col("quantity").sum().alias("total"))
        )

        # Create ordered week columns (1-52)
        ordered_weeks = [str(week) for week in range(1, 53)]

        # Pivot and fill missing weeks with 0
        pivoted = (
            df_week.pivot(
                index="terminology_id",
                on="week",
                values="total",
            )
            .fill_null(0)
            .select(["terminology_id"] + ordered_weeks)
            .sort("terminology_id")
        )

        return pivoted

    def get_terminology_ids(self) -> list:
        """
        Get sorted list of terminology IDs in the temporal matrix.

        Returns:
            Ordered list of terminology_id values
        """
        return self._pivot_df["terminology_id"].to_list()

    def compute(self) -> np.ndarray:
        """
        Compute temporal distance matrix using MSM distance.

        Applies Z-score normalization per APPROACH.md to ensure
        low-volume seasonal diseases match high-volume seasonal diseases.

        Returns:
            Distance matrix (NxN numpy array)
        """
        # Get time series matrix
        temporal_matrix = self._create_temporal_matrix()

        # Apply Z-score normalization (mean=0, std=1 per row)
        # This ensures pattern comparison independent of absolute volume
        temporal_matrix_normalized = self.normalize_zscore(temporal_matrix)

        # Get MSM parameters
        c = self.params.get("c", 1.0)
        n_jobs = self.params.get("n_jobs", -1)

        # Compute pairwise MSM distance on normalized data
        distance_matrix = pairwise_distance(
            temporal_matrix_normalized, method="msm", n_jobs=n_jobs, c=c
        )

        return distance_matrix.astype(np.float32)

    def _create_temporal_matrix(self) -> np.ndarray:
        """
        Create diagnosis × week matrix.

        Returns:
            Matrix where rows are diagnoses, columns are weeks (1-52)
        """
        pivoted_df = self._pivot_df
        # Get all columns except terminology_id (these are the week columns)
        week_cols = [col for col in pivoted_df.columns if col != "terminology_id"]
        return pivoted_df.select(week_cols).to_numpy()

    def get_temporal_pivot(self) -> pl.DataFrame:
        """
        Get temporal pivot table for analysis.

        Returns:
            Polars DataFrame with diagnosis × week data
        """
        return self._pivot_df

    def get_time_series(self) -> np.ndarray:
        """
        Get raw time series matrix for visualization or further analysis.

        Returns:
            Matrix of time series (rows are diagnoses, columns are weeks)
        """
        return self._create_temporal_matrix()
