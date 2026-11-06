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

    # Age group mappings based on actual data (from vir.features.AGE_GROUP)
    # AGE_GROUP = ["< 1 ano", "1 a 15 anos", "16 a 59 anos", "Maior que 60 anos"]
    AGE_GROUP_MAPPING = {
        "child": ["< 1 ano", "1 a 15 anos"],  # 0-15 years
        "young_adult": [
            "16 a 59 anos"
        ],  # 16-59 years (overlaps with adult, but needed for feature)
        "adult": ["16 a 59 anos"],  # 16-59 years
        "elderly": ["Maior que 60 anos"],  # 60+ years
        "reproductive_female": [
            "16 a 59 anos"
        ],  # Approximation: 16-59 years (can't distinguish 15-49 exactly)
    }

    # Feature definitions for dynamic feature generation
    _FEATURE_DEFINITIONS = {
        "female": {
            "sex_condition": "Feminino",
            "age_groups": None,
        },
        "child": {
            "sex_condition": None,
            "age_groups": "child",
        },
        "adult": {
            "sex_condition": None,
            "age_groups": "adult",
        },
        "elderly": {
            "sex_condition": None,
            "age_groups": "elderly",
        },
        "young_adult": {
            "sex_condition": None,
            "age_groups": "young_adult",
        },
        "reproductive_female": {
            "sex_condition": "Feminino",
            "age_groups": "reproductive_female",
        },
    }

    @cached_property
    def _features_df(self) -> pl.DataFrame:
        """
        Extracts and caches demographic features for each diagnosis using efficient Polars operations.
        Features are dynamically generated from _FEATURE_DEFINITIONS for better maintainability.
        """
        epsilon = 1e-9

        # Calculate total quantity per diagnosis
        total_per_diag = self.df.group_by("terminology_id").agg(
            pl.col("quantity").sum().alias("total")
        )

        # Handle empty data case
        if total_per_diag.is_empty() or total_per_diag["total"].sum() == 0:
            schema = {
                "terminology_id": pl.Int64,
                **{f"{name}_ratio": pl.Float32 for name in self._FEATURE_DEFINITIONS.keys()},
            }
            return pl.DataFrame({k: [] for k in schema}, schema=schema)

        # Calculate all feature counts in one aggregation
        feature_counts = self.df.group_by("terminology_id").agg(self._calculate_all_counts())

        # Join counts with totals
        features_df = total_per_diag.join(
            feature_counts, on="terminology_id", how="left"
        ).fill_null(0)

        # Calculate all ratios
        features_df = self._add_ratio_columns(features_df, epsilon)

        return features_df.sort("terminology_id")

    def get_terminology_ids(self) -> pl.Series:
        """
        Get the sorted list of terminology IDs corresponding to the matrix rows.

        Returns:
            A Polars Series containing the sorted terminology_id values.
        """
        return self._features_df["terminology_id"]

    @classmethod
    def _get_feature_columns(cls) -> list[str]:
        """Get list of demographic feature column names."""
        return [f"{name}_ratio" for name in cls._FEATURE_DEFINITIONS.keys()]

    def _build_count_expression(self, feature_name: str, feature_def: dict) -> pl.Expr:
        """Build a polars expression for counting a specific feature."""
        condition = pl.lit(True)

        if feature_def["sex_condition"]:
            condition = condition & (pl.col("sex") == feature_def["sex_condition"])

        if feature_def["age_groups"]:
            age_list = self.AGE_GROUP_MAPPING[feature_def["age_groups"]]
            condition = condition & pl.col("age_group").is_in(age_list)

        return (
            pl.when(condition)
            .then(pl.col("quantity"))
            .otherwise(0)
            .sum()
            .alias(f"{feature_name}_count")
        )

    def _calculate_all_counts(self) -> list[pl.Expr]:
        """Generate count aggregation expressions for all features."""
        return [
            self._build_count_expression(name, definition)
            for name, definition in self._FEATURE_DEFINITIONS.items()
        ]

    def _add_ratio_columns(self, features_df: pl.DataFrame, epsilon: float) -> pl.DataFrame:
        """Add ratio columns for all count columns."""
        ratio_expressions = []
        for feature_name in self._FEATURE_DEFINITIONS.keys():
            count_col = f"{feature_name}_count"
            ratio_col = f"{feature_name}_ratio"
            ratio_expressions.append(
                (pl.col(count_col) / (pl.col("total") + epsilon)).alias(ratio_col)
            )
        return features_df.with_columns(ratio_expressions)

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
        return self._features_df.select(self._get_feature_columns()).to_numpy().astype(np.float32)

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
        return self._features_df.select(["terminology_id"] + self._get_feature_columns())
