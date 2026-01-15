"""Base class for all views with automatic disk caching."""

import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import polars as pl
from diskcache import Cache

from vir.config import VIEWS_CACHE_DIR, get_indicative_label


class BaseView(ABC):
    """
    Base class for all views with automatic caching using diskcache.

    Each view computes a distance or similarity matrix for diagnoses.
    Expensive computations are automatically cached based on input data and parameters.
    """

    view_type: str = "base"  # Override in subclass

    def __init__(
        self,
        df: pl.DataFrame,
        dataset_name: str,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the view.

        Args:
            df: Input dataframe with diagnosis data
            dataset_name: Name of the dataset (for cache naming)
            params: Parameters for this view (distance metric, preprocessing options, etc.)
        """
        self.df = df
        self.dataset_name = dataset_name
        self.params = params or {}

        # Initialize cache directory for this view type
        cache_dir = VIEWS_CACHE_DIR / self.view_type
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = Cache(str(cache_dir))

    def get_matrix(self, force_recompute: bool = False) -> np.ndarray:
        """
        Get distance/similarity matrix (cached or computed).

        Args:
            force_recompute: If True, recompute even if cached

        Returns:
            Distance or similarity matrix (NxN numpy array)
        """
        cache_key = self._get_cache_key()

        if not force_recompute and self.cache.get(cache_key):
            print(f"Loading {self.view_type} matrix from cache...")
            return self.cache[cache_key]

        print(f"Computing {self.view_type} matrix...")
        matrix = self.compute()

        # Save to cache
        self.cache[cache_key] = matrix
        print(f"Saved to cache with key: {cache_key}")

        return matrix

    @abstractmethod
    def compute(self) -> np.ndarray:
        """
        Compute the distance/similarity matrix.

        Must be implemented by subclasses.

        Returns:
            Distance or similarity matrix (NxN numpy array)
        """
        raise NotImplementedError

    def _get_cache_key(self) -> str:
        """
        Generate cache key from dataset name, params, and data hash.

        Returns:
            Cache key string
        """
        # Sort params for consistent key generation
        param_str = "_".join(f"{k}={v}" for k, v in sorted(self.params.items()))

        # Hash dataframe content (only relevant columns)
        df_hash = self._hash_dataframe()

        # Get indicative filtering label
        indic_label = get_indicative_label()

        # Combine into cache key
        cache_key = f"{self.dataset_name}_{param_str}_{df_hash[:8]}_{indic_label}"
        return cache_key

    def _hash_dataframe(self) -> str:
        """
        Create hash of dataframe content.

        Uses terminology_id and quantity columns to detect data changes.

        Returns:
            Hash string
        """
        # Use relevant columns for hashing
        cols_to_hash = ["terminology_id"]
        if "quantity" in self.df.columns:
            cols_to_hash.append("quantity")

        # Create deterministic hash
        df_bytes = self.df.select(cols_to_hash).write_csv().encode()
        return hashlib.sha256(df_bytes).hexdigest()

    def clear_cache(self):
        """Clear all cached matrices for this view."""
        self.cache.clear()
        print(f"Cleared cache for {self.view_type} view")

    # Normalization methods (common to all views)

    @staticmethod
    def normalize_zscore(data: np.ndarray) -> np.ndarray:
        """
        Z-score normalization: standardize each row to mean=0, std=1.

        Args:
            data: Input matrix (rows are samples)

        Returns:
            Normalized matrix
        """
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        return (data - mean) / std

    @staticmethod
    def normalize_minmax(data: np.ndarray) -> np.ndarray:
        """
        Min-max normalization: scale each row to [0, 1].

        Args:
            data: Input matrix (rows are samples)

        Returns:
            Normalized matrix
        """
        min_val = np.min(data, axis=1, keepdims=True)
        max_val = np.max(data, axis=1, keepdims=True)
        # Avoid division by zero
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)
        return (data - min_val) / range_val

    @staticmethod
    def normalize_probability(data: np.ndarray, pseudocount: float = 1e-6) -> np.ndarray:
        """
        Normalize each row to probability distribution (sum to 1).

        Applies pseudocount smoothing to handle zeros.

        Args:
            data: Input matrix (rows are samples)
            pseudocount: Small value added to mitigate zero inflation

        Returns:
            Normalized matrix where each row sums to 1
        """
        # Add pseudocount
        data_smoothed = data + pseudocount

        # Normalize to probability distribution
        row_sums = np.sum(data_smoothed, axis=1, keepdims=True)
        return data_smoothed / row_sums

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache info (size, entries, etc.)
        """
        return {
            "view_type": self.view_type,
            "cache_dir": str(self.cache.directory),
            "size_mb": self.cache.volume() / (1024 * 1024),
            "entries": len(self.cache),
        }

    @abstractmethod
    def get_terminology_ids(self) -> list:
        """
        Get the ordered list of terminology IDs in the distance matrix.

        Must be implemented by subclasses to ensure proper alignment.

        Returns:
            Ordered list of terminology IDs corresponding to matrix rows/columns
        """
        raise NotImplementedError

    def align_to_common_ids(
        self, common_ids: list, distance_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Align distance matrix to a common set of terminology IDs.

        Reorders and filters the matrix to match the provided ID list.
        Missing IDs are filled with maximum distance (indicating no similarity).

        Args:
            common_ids: Ordered list of terminology IDs to align to
            distance_matrix: Matrix to align. If None, uses get_matrix()

        Returns:
            Aligned distance matrix matching common_ids order
        """
        if distance_matrix is None:
            distance_matrix = self.get_matrix()

        current_ids = self.get_terminology_ids()
        n_target = len(common_ids)

        # Create mapping from current IDs to matrix indices
        current_id_to_idx = {tid: idx for idx, tid in enumerate(current_ids)}

        # Create aligned matrix (initialize with max distance)
        max_dist = np.nanmax(distance_matrix)
        aligned_matrix = np.full((n_target, n_target), max_dist, dtype=np.float32)

        # Fill in values for common IDs
        for i, id_i in enumerate(common_ids):
            if id_i not in current_id_to_idx:
                continue
            idx_i = current_id_to_idx[id_i]

            for j, id_j in enumerate(common_ids):
                if id_j not in current_id_to_idx:
                    continue
                idx_j = current_id_to_idx[id_j]

                aligned_matrix[i, j] = distance_matrix[idx_i, idx_j]

        return aligned_matrix

    def to_affinity(
        self,
        distance_matrix: Optional[np.ndarray] = None,
        k: int = 20,
        mu: float = 0.5,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        Convert distance matrix to affinity (similarity) matrix using adaptive Gaussian kernel.

        Uses local scaling based on k-nearest neighbors, following SNF methodology.
        This applies the same transformation as stable_affinity_from_distance() from
        the SNF pipeline.

        Args:
            distance_matrix: Distance matrix to convert. If None, uses get_matrix()
            k: Number of nearest neighbors for adaptive sigma computation
            mu: Scaling factor for sigma to prevent numerical underflow
            eps: Small epsilon to prevent division by zero

        Returns:
            Affinity matrix (NxN numpy array, range [0, 1])
        """
        if distance_matrix is None:
            distance_matrix = self.get_matrix()

        # Handle NaN/Inf values
        distance_matrix = np.nan_to_num(
            distance_matrix,
            nan=0.0,
            posinf=np.nanmax(distance_matrix),
            neginf=0.0,
        )

        # Ensure perfect symmetry
        distance_matrix = (distance_matrix + distance_matrix.T) / 2.0
        np.fill_diagonal(distance_matrix, 0.0)

        # Normalize distances to [0, 1]
        distance_matrix = distance_matrix / (np.nanmax(distance_matrix) + eps)

        # Compute adaptive local scale sigma (k-NN based)
        sorted_distances = np.sort(distance_matrix, axis=1)
        mean_neighbor_distance = np.vstack(sorted_distances[:, 1 : k + 1].mean(axis=1) + eps)
        sigma = (mean_neighbor_distance + mean_neighbor_distance.T + distance_matrix) / 3.0
        sigma = np.clip(mu * sigma, eps, None)

        # Apply Gaussian kernel
        affinity_matrix = np.exp(-(distance_matrix**2) / (2.0 * sigma**2))
        affinity_matrix = np.nan_to_num(affinity_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(affinity_matrix, 0.0)

        return affinity_matrix
