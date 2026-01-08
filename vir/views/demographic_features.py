"""
Alternative demographic view using engineered features.

Instead of using full age/sex distributions, create interpretable features:
- Female ratio
- Age group concentrations (children, adults, elderly)
- Deviation from population baseline
"""

from functools import cached_property

import numpy as np
import polars as pl
from scipy.spatial.distance import pdist, squareform

from vir.views.base import BaseView


class DemographicFeaturesView(BaseView):
    """
    Demographic view using engineered features instead of full distributions.

    Features extracted:
    - female_ratio: % of cases that are female
    - child_ratio: % of cases in children (0-17 years)
    - adult_ratio: % of cases in adults (18-64 years)
    - elderly_ratio: % of cases in elderly (65+ years)
    - young_adult_ratio: % of cases in young adults (18-39 years)
    - reproductive_age_female_ratio: % of cases in females 15-49 years

    These features capture demographic patterns more directly.
    """

    view_type = "demographic_features"

    # Age group mappings (adjust based on your data)
    AGE_GROUP_MAPPING = {
        "child": ["0-4", "5-9", "10-14", "15-17", "0-17", "0-18"],
        "young_adult": ["18-24", "18-39", "20-29", "30-39"],
        "adult": ["18-64", "40-49", "50-59", "60-64"],
        "elderly": ["65+", "65-74", "75+", "65-100"],
        "reproductive_female": [
            "15-49",
            "15-17",
            "18-24",
            "25-29",
            "30-34",
            "35-39",
            "40-44",
            "45-49",
        ],
    }

    @cached_property
    def _features_df(self) -> pl.DataFrame:
        """
        Extracts and caches demographic features for each diagnosis using efficient Polars operations.
        This fixes the performance bottleneck and incorrect calculations of the original implementation.
        """
        # A small epsilon to add to denominators to avoid division by zero errors.
        epsilon = 1e-9

        # Calculate total quantity per diagnosis.
        total_per_diag = self.df.group_by("terminology_id").agg(
            pl.col("quantity").sum().alias("total")
        )

        # If there's no data, return an empty frame with the correct schema.
        if total_per_diag.is_empty() or total_per_diag["total"].sum() == 0:
            schema = {
                "terminology_id": pl.Int64,
                "female_ratio": pl.Float32,
                "child_ratio": pl.Float32,
                "adult_ratio": pl.Float32,
                "elderly_ratio": pl.Float32,
                "young_adult_ratio": pl.Float32,
                "reproductive_female_ratio": pl.Float32,
            }
            return pl.DataFrame({k: [] for k in schema}, schema=schema)

        # Calculate counts for all features in a single pass over the data.
        feature_counts = self.df.group_by("terminology_id").agg(
            pl.when(pl.col("sex") == "Feminino")
            .then(pl.col("quantity"))
            .otherwise(0)
            .sum()
            .alias("female_count"),
            pl.when(pl.col("age_group").is_in(self.AGE_GROUP_MAPPING["child"]))
            .then(pl.col("quantity"))
            .otherwise(0)
            .sum()
            .alias("child_count"),
            pl.when(pl.col("age_group").is_in(self.AGE_GROUP_MAPPING["adult"]))
            .then(pl.col("quantity"))
            .otherwise(0)
            .sum()
            .alias("adult_count"),
            pl.when(pl.col("age_group").is_in(self.AGE_GROUP_MAPPING["elderly"]))
            .then(pl.col("quantity"))
            .otherwise(0)
            .sum()
            .alias("elderly_count"),
            pl.when(pl.col("age_group").is_in(self.AGE_GROUP_MAPPING["young_adult"]))
            .then(pl.col("quantity"))
            .otherwise(0)
            .sum()
            .alias("young_adult_count"),
            pl.when(
                (pl.col("sex") == "Feminino")
                & (pl.col("age_group").is_in(self.AGE_GROUP_MAPPING["reproductive_female"]))
            )
            .then(pl.col("quantity"))
            .otherwise(0)
            .sum()
            .alias("reproductive_female_count"),
        )

        # Join the feature counts with the total counts.
        features_df = total_per_diag.join(
            feature_counts, on="terminology_id", how="left"
        ).fill_null(0)

        # Compute the final feature ratios.
        features_df = features_df.with_columns(
            (pl.col("female_count") / (pl.col("total") + epsilon)).alias("female_ratio"),
            (pl.col("child_count") / (pl.col("total") + epsilon)).alias("child_ratio"),
            (pl.col("adult_count") / (pl.col("total") + epsilon)).alias("adult_ratio"),
            (pl.col("elderly_count") / (pl.col("total") + epsilon)).alias("elderly_ratio"),
            (pl.col("young_adult_count") / (pl.col("total") + epsilon)).alias("young_adult_ratio"),
            (pl.col("reproductive_female_count") / (pl.col("total") + epsilon)).alias(
                "reproductive_female_ratio"
            ),
        ).sort("terminology_id")

        return features_df

    def get_terminology_ids(self) -> pl.Series:
        """
        Get the sorted list of terminology IDs corresponding to the matrix rows.

        Returns:
            A Polars Series containing the sorted terminology_id values.
        """
        return self._features_df["terminology_id"]

    def compute(self) -> np.ndarray:
        """
        Compute distance matrix using demographic features.

        Returns:
            Euclidean distance matrix on feature space
        """
        features = self._extract_features()
        distance_matrix = self._compute_euclidean_distance(features)
        self._validate_distance_matrix(distance_matrix)
        return distance_matrix

    def _extract_features(self) -> np.ndarray:
        """
        Extract demographic features for each diagnosis.

        Returns:
            Feature matrix (N diagnoses × 6 features)
        """
        feature_cols = [
            "female_ratio",
            "child_ratio",
            "adult_ratio",
            "elderly_ratio",
            "young_adult_ratio",
            "reproductive_female_ratio",
        ]
        return self._features_df.select(feature_cols).to_numpy().astype(np.float32)

    def _compute_euclidean_distance(self, features: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Euclidean distance using scipy for efficiency.
        """
        distance_vector = pdist(features.astype(np.float64), metric="euclidean")
        return squareform(distance_vector).astype(np.float32)

    def _validate_distance_matrix(self, matrix: np.ndarray):
        """Validate distance matrix."""
        if np.any(np.isnan(matrix)):
            raise ValueError("Distance matrix contains NaN values")

        if np.any(np.isinf(matrix)):
            raise ValueError("Distance matrix contains infinite values")

        if np.any(matrix < 0):
            raise ValueError("Distance matrix contains negative values")

        if not np.allclose(matrix, matrix.T):
            raise ValueError("Distance matrix is not symmetric")

    def get_features(self) -> pl.DataFrame:
        """
        Get feature matrix with labels.

        Returns:
            DataFrame with features for each diagnosis
        """
        return self._features_df.select(
            [
                "terminology_id",
                "female_ratio",
                "child_ratio",
                "adult_ratio",
                "elderly_ratio",
                "young_adult_ratio",
                "reproductive_female_ratio",
            ]
        )
