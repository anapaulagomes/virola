"""Temporal view: time series similarity using MSM distance."""

from functools import cached_property

import numpy as np
import polars as pl
from aeon.distances import pairwise_distance

from vir.config import MIN_WEEKS
from vir.views.base import BaseView


class TemporalView(BaseView):
    """
    Temporal view: captures temporal patterns and cyclic behaviour of diagnoses over time.

    Series are represented as year_week sequences (chronological order), preserving
    surges, outbreaks and secular trends that would be lost by aggregating across years.

    - Input: Diagnosis × year_week matrix (chronological time series)
    - Preprocessing: filter for min_weeks coverage, fill missing year_weeks with 0,
                     Z-score normalisation per row
    - Distance metric: MSM (Move-Split-Merge)
    - Output: Distance matrix
    """

    view_type = "temporal"

    @cached_property
    def _pivot_df(self) -> pl.DataFrame:
        """
        Build and cache the diagnosis × year_week pivot table.

        Columns are sequential integers (1, 2, 3, …) representing chronological
        year_week order.  The human-readable label for each column is stored in
        ``self._col_labels`` as a list of "YYYY-WW" strings.

        Only diagnoses that appear in at least ``min_weeks`` distinct year_weeks
        are included.

        Returns:
            Polars DataFrame: rows = terminology_id, columns = sequential ints.
        """
        min_weeks = self.params.get("min_weeks", MIN_WEEKS)

        # Build chronologically-ordered list of (year, week) pairs present in data
        ordered_yw = self.df.select(["year", "week"]).unique().sort(["year", "week"])

        # Sequential integer index for each year_week  (1-based)
        ordered_yw = ordered_yw.with_row_index(name="yw_index", offset=1)

        # Store human-readable labels for downstream use (visualisation, export)
        self._col_labels: list[str] = [
            f"{r['year']}-{r['week']:02d}" for r in ordered_yw.iter_rows(named=True)
        ]

        # Aggregate raw data to (terminology_id, year, week) → total
        df_yw = (
            self.df.group_by(["terminology_id", "year", "week"])
            .agg(pl.col("quantity").sum().alias("total"))
            .join(ordered_yw.select(["year", "week", "yw_index"]), on=["year", "week"])
        )

        # Filter: keep only diagnoses with sufficient temporal coverage
        sufficient = (
            df_yw.group_by("terminology_id")
            .agg(pl.col("yw_index").n_unique().alias("yw_count"))
            .filter(pl.col("yw_count") >= min_weeks)
        )
        df_yw = df_yw.filter(pl.col("terminology_id").is_in(sufficient["terminology_id"]))

        # Pivot: rows = terminology_id, columns = yw_index (as string for pivot)
        seq_cols = [str(i) for i in ordered_yw["yw_index"].to_list()]

        pivoted = (
            df_yw.with_columns(pl.col("yw_index").cast(pl.Utf8))
            .pivot(
                index="terminology_id",
                on="yw_index",
                values="total",
            )
            .fill_null(0)
            .select(["terminology_id"] + seq_cols)
            .sort("terminology_id")
        )

        return pivoted

    def get_terminology_ids(self) -> list:
        """Ordered list of terminology_ids in the temporal matrix."""
        return self._pivot_df["terminology_id"].to_list()

    def get_col_labels(self) -> list[str]:
        """Human-readable 'YYYY-WW' label for each column of the time series matrix."""
        _ = self._pivot_df  # ensure cached property is built
        return self._col_labels

    def get_temporal_pivot(self) -> pl.DataFrame:
        """Diagnosis × year_week pivot table (columns are sequential integers)."""
        return self._pivot_df

    def get_time_series(self) -> np.ndarray:
        """Raw time series matrix — rows are diagnoses, columns are year_weeks."""
        return self._create_temporal_matrix()

    def compute(self) -> np.ndarray:
        """
        Compute pairwise MSM distance matrix on Z-score normalised series.

        Z-score normalisation ensures that low-volume seasonal diseases are
        compared by shape rather than magnitude.

        Returns:
            float32 distance matrix (N × N).
        """
        temporal_matrix = self._create_temporal_matrix()
        temporal_matrix_normalized = self.normalize_zscore(temporal_matrix)

        c = self.params.get("c", 1.0)
        n_jobs = self.params.get("n_jobs", -1)

        distance_matrix = pairwise_distance(
            temporal_matrix_normalized, method="msm", n_jobs=n_jobs, c=c
        )
        return distance_matrix.astype(np.float32)

    def _create_temporal_matrix(self) -> np.ndarray:
        """Return the numeric time-series matrix (excludes terminology_id column)."""
        pivoted_df = self._pivot_df
        value_cols = [col for col in pivoted_df.columns if col != "terminology_id"]
        return pivoted_df.select(value_cols).to_numpy()
