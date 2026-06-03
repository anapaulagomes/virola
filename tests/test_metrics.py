import numpy as np
import pytest
from snf import metrics as snf_metrics

from vir import metrics as vir_metrics


def _symmetric_affinity(n_samples, seed, negative=False):
    generator = np.random.default_rng(seed)
    if negative:
        matrix = generator.normal(size=(n_samples, n_samples))
    else:
        matrix = generator.random((n_samples, n_samples))
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _labels_with_all_clusters(n_samples, n_clusters, seed):
    generator = np.random.default_rng(seed + 7)
    labels = generator.integers(0, n_clusters, size=n_samples)
    labels[:n_clusters] = np.arange(n_clusters)
    generator.shuffle(labels)
    return labels


class TestSilhouetteScoreMatchesSnf:
    @pytest.mark.parametrize("n_samples", [60, 200])
    @pytest.mark.parametrize("n_clusters", [3, 10, 40])
    def test_matches_on_positive_affinities(self, n_samples, n_clusters):
        affinity = _symmetric_affinity(n_samples, seed=n_samples + n_clusters)
        labels = _labels_with_all_clusters(n_samples, n_clusters, seed=n_clusters)
        assert np.isclose(
            vir_metrics.silhouette_score(affinity, labels),
            snf_metrics.silhouette_score(affinity, labels),
        )

    def test_matches_with_negative_affinities(self):
        affinity = _symmetric_affinity(150, seed=1, negative=True)
        labels = _labels_with_all_clusters(150, 8, seed=2)
        assert np.isclose(
            vir_metrics.silhouette_score(affinity, labels),
            snf_metrics.silhouette_score(affinity, labels),
        )

    def test_matches_with_singleton_cluster(self):
        affinity = _symmetric_affinity(100, seed=3)
        labels = _labels_with_all_clusters(100, 6, seed=4)
        labels[labels == 5] = 0
        labels[0] = 5
        assert np.bincount(labels)[5] == 1
        assert np.isclose(
            vir_metrics.silhouette_score(affinity, labels),
            snf_metrics.silhouette_score(affinity, labels),
        )

    def test_does_not_mutate_input(self):
        affinity = _symmetric_affinity(50, seed=9)
        before = affinity.copy()
        vir_metrics.silhouette_score(affinity, _labels_with_all_clusters(50, 4, seed=9))
        assert np.array_equal(affinity, before)


class TestAffinityZscoreMatchesSnf:
    def test_identical_zscore_for_fixed_seed(self):
        affinity = _symmetric_affinity(180, seed=11)
        labels = _labels_with_all_clusters(180, 12, seed=12)
        assert np.isclose(
            vir_metrics.affinity_zscore(affinity, labels, n_perms=100, seed=42),
            snf_metrics.affinity_zscore(affinity, labels, n_perms=100, seed=42),
        )


class TestSilhouetteScoreValidation:
    def test_raises_on_single_cluster(self):
        affinity = _symmetric_affinity(20, seed=5)
        with pytest.raises(ValueError):
            vir_metrics.silhouette_score(affinity, np.zeros(20, dtype=int))
