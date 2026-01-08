"""Base class for all views with automatic disk caching."""

import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import polars as pl
from diskcache import Cache

from vir.config import VIEWS_CACHE_DIR


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

        if not force_recompute and cache_key in self.cache:
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

        # Combine into cache key
        cache_key = f"{self.dataset_name}_{param_str}_{df_hash[:8]}"
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
