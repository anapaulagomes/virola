"""Temporal view: time series similarity using MSM distance."""

from functools import cached_property

import numpy as np
import polars as pl
from aeon.distances import pairwise_distance

from vir.config import MIN_WEEKS, TEMPORAL_MODE
from vir.helpers import get_n_jobs
from vir.views.base import BaseView


class TemporalView(BaseView):
    """
    Temporal view: captures the annual seasonal pattern of each diagnosis.

    Two modes are supported (controlled by the ``mode`` param, default ``"annual"``):

    ``"annual"``
        Each diagnosis is represented as a 52-point week-of-year profile — the
        mean quantity per calendar week across all years in the data.  This gives
        a clean seasonal signal regardless of how many years are available, and
        avoids the false-similarity problem that arises when z-scoring long sparse
        chronological series (where two unrelated rare diseases look identical).

    ``"chronological"``
        Original behaviour: full chronological year_week sequence.  Preserves
        multi-year trends and outbreak timing but is noisy for short datasets
        (< 5 years) and produces flat MSM distances for sparse codes.

    Normalization: minmax per row (row min → 0, row max → 1).  Preserves the
    sparsity structure — zero weeks stay zero — while removing magnitude
    differences between high- and low-frequency diagnoses.  This prevents
    two sparse-but-unrelated diseases from appearing identical after normalization
    (the failure mode of z-score on near-zero series).

    Distance: MSM (Move-Split-Merge).
    """

    view_type = "temporal"

    @cached_property
    def _mode(self) -> str:
        return self.params.get("mode", TEMPORAL_MODE)

    @cached_property
    def _pivot_df(self) -> pl.DataFrame:
        """Build and cache the diagnosis × time pivot table.

        In ``"annual"`` mode: columns are week numbers 1–52, values are
        mean counts per week-of-year across all years.

        In ``"chronological"`` mode: columns are sequential integers
        representing ordered (year, week) pairs; human-readable labels
        are stored in ``self._col_labels``.

        Only diagnoses with at least ``min_weeks`` distinct time columns
        are retained.
        """
        if self._mode == "annual":
            return self._build_annual_pivot()
        return self._build_chronological_pivot()

    def _build_annual_pivot(self) -> pl.DataFrame:
        """52-point week-of-year profile (mean across years)."""
        min_weeks = self.params.get("min_weeks", MIN_WEEKS)

        # Mean quantity per (terminology_id, week_of_year)
        df_week = self.df.group_by(["terminology_id", "week"]).agg(
            pl.col("quantity").mean().alias("total")
        )

        # Filter: keep only diagnoses present in enough distinct weeks
        sufficient = (
            df_week.group_by("terminology_id")
            .agg(pl.col("week").n_unique().alias("week_count"))
            .filter(pl.col("week_count") >= min_weeks)
        )
        df_week = df_week.filter(pl.col("terminology_id").is_in(sufficient["terminology_id"]))

        # All 52 week columns (fill missing weeks with 0)
        all_weeks = [str(w) for w in range(1, 53)]
        self._col_labels: list[str] = [f"W{w:02d}" for w in range(1, 53)]

        tid_dtype = self.df["terminology_id"].dtype

        pivoted = (
            df_week.with_columns(pl.col("week").cast(pl.Utf8))
            .pivot(index="terminology_id", on="week", values="total")
            .fill_null(0.0)
            .with_columns(pl.col("terminology_id").cast(tid_dtype))
            .sort("terminology_id")
        )

        # Ensure all 52 columns exist in order, adding zeros for absent weeks
        for col in all_weeks:
            if col not in pivoted.columns:
                pivoted = pivoted.with_columns(pl.lit(0.0).alias(col))

        return pivoted.select(["terminology_id"] + all_weeks)

    def _build_chronological_pivot(self) -> pl.DataFrame:
        """Full chronological year_week sequence (original behaviour)."""
        min_weeks = self.params.get("min_weeks", MIN_WEEKS)

        ordered_yw = self.df.select(["year", "week"]).unique().sort(["year", "week"])
        ordered_yw = ordered_yw.with_row_index(name="yw_index", offset=1)

        self._col_labels = [
            f"{r['year']}-{r['week']:02d}" for r in ordered_yw.iter_rows(named=True)
        ]

        df_yw = (
            self.df.group_by(["terminology_id", "year", "week"])
            .agg(pl.col("quantity").sum().alias("total"))
            .join(ordered_yw.select(["year", "week", "yw_index"]), on=["year", "week"])
        )

        sufficient = (
            df_yw.group_by("terminology_id")
            .agg(pl.col("yw_index").n_unique().alias("yw_count"))
            .filter(pl.col("yw_count") >= min_weeks)
        )
        df_yw = df_yw.filter(pl.col("terminology_id").is_in(sufficient["terminology_id"]))

        seq_cols = [str(i) for i in ordered_yw["yw_index"].to_list()]

        return (
            df_yw.with_columns(pl.col("yw_index").cast(pl.Utf8))
            .pivot(index="terminology_id", on="yw_index", values="total")
            .fill_null(0)
            .select(["terminology_id"] + seq_cols)
            .sort("terminology_id")
        )

    def get_terminology_ids(self) -> list:
        """Ordered list of terminology_ids in the temporal matrix."""
        return self._pivot_df["terminology_id"].to_list()

    def get_col_labels(self) -> list[str]:
        """Human-readable label for each column of the time series matrix."""
        _ = self._pivot_df  # ensure cached property is built
        return self._col_labels

    def get_temporal_pivot(self) -> pl.DataFrame:
        """Diagnosis × time pivot table."""
        return self._pivot_df

    def get_time_series(self) -> np.ndarray:
        """Raw time series matrix — rows are diagnoses, columns are time points."""
        return self._create_temporal_matrix()

    def compute(self) -> np.ndarray:
        """
        Compute pairwise MSM distance on minmax-normalised series.

        Minmax normalization scales each row to [0, 1], preserving the
        sparsity structure (zero weeks stay zero) while removing magnitude
        differences.  This prevents sparse, unrelated diagnoses from
        appearing identical after normalization — the failure mode of
        z-score on near-zero series.

        Returns:
            float32 distance matrix (N × N).
        """
        temporal_matrix = self._create_temporal_matrix()
        temporal_matrix_normalized = self.normalize_minmax(temporal_matrix)

        c = self.params.get("c", 1.0)
        n_jobs = self.params.get("n_jobs", get_n_jobs())

        distance_matrix = pairwise_distance(
            temporal_matrix_normalized, method="msm", n_jobs=n_jobs, c=c
        )
        return distance_matrix.astype(np.float32)

    def _create_temporal_matrix(self) -> np.ndarray:
        """Return the numeric time-series matrix (excludes terminology_id column)."""
        pivoted_df = self._pivot_df
        value_cols = [col for col in pivoted_df.columns if col != "terminology_id"]
        return pivoted_df.select(value_cols).to_numpy()
