"""Vectorized drop-in replacements for ``snf.metrics`` affinity-silhouette functions.

These reproduce ``snf.metrics.silhouette_score`` and ``snf.metrics.affinity_zscore``
numerically (identical within floating-point tolerance, and with a bit-identical
permutation sequence for a fixed seed), but replace the per-cluster Python loop in
snf's ``_silhouette_samples`` with matrix operations: a single ``affinity @ membership``
matmul against a one-hot cluster-membership matrix yields every sample's affinity to
every cluster at once. This is ~10x faster on the large affinity matrices used by the
ablation study, is architecture-independent (no multiprocessing), and benefits from
BLAS threading where available (e.g. SLURM ``--cpus-per-task``).
"""

import numpy as np
from sklearn.utils import check_random_state


def _silhouette_samples(affinity_matrix, labels):
    affinity = np.array(affinity_matrix, dtype=np.float64, copy=True)
    np.fill_diagonal(affinity, 0.0)

    unique_labels, encoded_labels = np.unique(labels, return_inverse=True)
    n_labels = len(unique_labels)
    n_samples = affinity.shape[0]
    if not 1 < n_labels < n_samples:
        raise ValueError(
            f"Number of labels is {n_labels}. Valid values are 2 to n_samples - 1 (inclusive)"
        )

    sample_indices = np.arange(n_samples)
    cluster_sizes = np.bincount(encoded_labels, minlength=n_labels).astype(np.float64)

    membership = np.zeros((n_samples, n_labels), dtype=np.float64)
    membership[sample_indices, encoded_labels] = 1.0

    # affinity_to_cluster[i, k] = sum of affinity from sample i to members of
    # cluster k. The diagonal is zeroed, so a sample never counts itself.
    affinity_to_cluster = affinity @ membership

    own_cluster_sum = affinity_to_cluster[sample_indices, encoded_labels]
    intra_denominator = cluster_sizes[encoded_labels] - 1.0
    safe_denominator = np.where(intra_denominator > 0, intra_denominator, 1.0)
    intra_cluster_affinity = np.where(
        intra_denominator > 0, own_cluster_sum / safe_denominator, 0.0
    )

    mean_affinity_to_cluster = affinity_to_cluster / cluster_sizes[np.newaxis, :]
    mean_affinity_to_cluster[sample_indices, encoded_labels] = -np.inf
    best_other_cluster_affinity = mean_affinity_to_cluster.max(axis=1)
    # snf initializes the inter-cluster affinity at 0 and accumulates it via
    # ``maximum`` over the other clusters, which floors it at 0 (this only
    # matters when every other-cluster mean affinity is negative).
    inter_cluster_affinity = np.maximum(0.0, best_other_cluster_affinity)

    with np.errstate(divide="ignore", invalid="ignore"):
        silhouette = (intra_cluster_affinity - inter_cluster_affinity) / np.maximum(
            intra_cluster_affinity, inter_cluster_affinity
        )
    silhouette[cluster_sizes[encoded_labels] == 1] = 0.0
    return silhouette


def silhouette_score(affinity_matrix, labels):
    """Affinity silhouette score; matches ``snf.metrics.silhouette_score``."""
    return float(np.mean(_silhouette_samples(affinity_matrix, labels)))


def affinity_zscore(affinity_matrix, labels, n_perms=1000, seed=None):
    """Permutation z-score of the affinity silhouette.

    Matches ``snf.metrics.affinity_zscore``: it mirrors snf's RNG usage
    (``check_random_state`` plus ``rs.permutation`` once per permutation), so for a
    fixed ``seed`` the permutation sequence is identical to snf's and the resulting
    z-score matches within floating point.
    """
    random_state = check_random_state(seed)
    permuted_scores = np.empty(n_perms, dtype=np.float64)
    for permutation_index in range(n_perms):
        permuted_scores[permutation_index] = silhouette_score(
            affinity_matrix, random_state.permutation(labels)
        )
    observed_score = silhouette_score(affinity_matrix, labels)
    return float((observed_score - permuted_scores.mean()) / permuted_scores.std())
