"""Demographic view: age/sex distribution similarity."""

from functools import cached_property

import numpy as np
import polars as pl
from scipy.spatial.distance import pdist, squareform

from vir.views.base import BaseView


class DemographicView(BaseView):
    """
    Demographic view: captures similarities in age- and sex-specific patient distributions.

    Follows APPROACH.md specification:
    - Input: Diagnosis × demographic bin matrix (age/sex strata)
    - Preprocessing: Pseudocount smoothing + row-wise normalization to probability distributions
    - Distance metric: Chi-square distance
    - Rationale: Appropriate for comparing compositional histogram data
    """

    view_type = "demographic"

    @cached_property
    def _pivot_df(self) -> pl.DataFrame:
        """
        Create and cache the diagnosis × demographic bin pivot table.

        The table is sorted by 'terminology_id' to ensure consistent row ordering
        across all computations.

        Returns:
            A Polars DataFrame pivoted to show diagnoses against demographic bins.
        """
        df_demographics = (
            self.df.with_columns(
                (pl.col("sex").cast(pl.Utf8) + "-" + pl.col("age_group").cast(pl.Utf8)).alias(
                    "sex_age"
                )
            )
            .group_by(["terminology_id", "sex_age"])
            .agg(pl.col("quantity").sum().alias("total"))
        )

        pivoted = (
            df_demographics.pivot(
                index="terminology_id",
                on="sex_age",
                values="total",
            )
            .fill_null(0)  # TODO check this
            .sort("terminology_id")
        )

        return pivoted

    def get_terminology_ids(self) -> list:
        """
        Get the sorted list of terminology IDs corresponding to the matrix rows.

        Returns:
            Ordered list of terminology_id values
        """
        return self._pivot_df["terminology_id"].to_list()

    def compute(self) -> np.ndarray:
        """
        Compute demographic distance matrix using chi-square distance.

        Returns:
            Distance matrix (NxN numpy array)
        """
        # Step 1: Create demographic pivot (diagnosis × age/sex bins)
        pivot_matrix = self._create_demographic_pivot()

        # Step 2: Apply preprocessing (pseudocount + probability normalization)
        pseudocount = self.params.get("pseudocount", 0.5)
        normalized_matrix = self.normalize_probability(pivot_matrix, pseudocount=pseudocount)

        # Step 3: Compute chi-square distance
        distance_matrix = self._compute_chisquare_distance(normalized_matrix)

        # Step 4: Validate the result
        self._validate_distance_matrix(distance_matrix)

        return distance_matrix

    def _validate_distance_matrix(self, matrix: np.ndarray):
        """
        Validate distance matrix for numerical issues.

        Args:
            matrix: Distance matrix to validate

        Raises:
            ValueError: If matrix contains invalid values
        """
        if np.any(np.isnan(matrix)):
            raise ValueError("Distance matrix contains NaN values")

        if np.any(np.isinf(matrix)):
            raise ValueError("Distance matrix contains infinite values")

        if np.any(matrix < 0):
            raise ValueError("Distance matrix contains negative values")

        # Check symmetry
        if not np.allclose(matrix, matrix.T):
            raise ValueError("Distance matrix is not symmetric")

    def _create_demographic_pivot(self) -> np.ndarray:
        """
        Create diagnosis × demographic bin matrix from the cached pivot table.

        Returns:
            Matrix where rows are diagnoses, columns are age/sex bins
        """
        pivoted_df = self._pivot_df
        feature_cols = [col for col in pivoted_df.columns if col != "terminology_id"]
        return pivoted_df.select(feature_cols).to_numpy()

    def _compute_chisquare_distance(self, data: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        """
        Compute pairwise chi-square distance between probability distributions.

        Chi-square distance for histograms:
        d_chi2(p, q) = sqrt(sum((p_i - q_i)^2 / (p_i + q_i)))

        This is computed efficiently using scipy.spatial.distance.pdist.

        Args:
            data: Matrix of probability distributions (rows are samples)
            eps: Small value to prevent division by zero

        Returns:
            Pairwise distance matrix (NxN)
        """

        def chi2_metric(p: np.ndarray, q: np.ndarray) -> float:
            """The Chi-square distance formula specified in APPROACH.md."""
            diff_sq = (p - q) ** 2
            sum_pq = p + q

            mask = sum_pq > eps
            if not np.any(mask):
                return 0.0

            chi_sq = np.sum(diff_sq[mask] / sum_pq[mask])
            return np.sqrt(np.clip(chi_sq, 0, None))

        # pdist computes the condensed distance matrix (1D array)
        distance_vector = pdist(data.astype(np.float64), metric=chi2_metric)

        # squareform converts it to a symmetric NxN matrix
        distance_matrix = squareform(distance_vector).astype(np.float32)

        # Final validation for any numerical instabilities
        if np.any(~np.isfinite(distance_matrix)):
            max_finite = np.max(distance_matrix[np.isfinite(distance_matrix)], initial=0)
            distance_matrix = np.where(np.isfinite(distance_matrix), distance_matrix, max_finite)

        return distance_matrix

    def get_demographic_pivot(self) -> pl.DataFrame:
        """
        Get demographic pivot table for analysis.

        Returns:
            Polars DataFrame with diagnosis × demographic bins
        """
        return self._pivot_df

    def get_normalized_distributions(self) -> np.ndarray:
        """
        Get normalized probability distributions for analysis.

        Returns:
            Matrix of probability distributions (rows sum to 1)
        """
        pivot_matrix = self._create_demographic_pivot()
        pseudocount = self.params.get("pseudocount", 0.5)
        return self.normalize_probability(pivot_matrix, pseudocount=pseudocount)
