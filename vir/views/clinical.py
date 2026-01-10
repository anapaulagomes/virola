"""Clinical view: semantic/ontological similarity from embeddings."""

from typing import Any, Dict

import numpy as np
import polars as pl

from vir.config import logger
from vir.db import (
    DEFAULT_MODEL,
    create_dimensions_table_if_not_exists,
    dimension_exists,
    get_connection,
    insert_clinical_dimensions,
    read_dimension,
)
from vir.views.base import BaseView


class ClinicalView(BaseView):
    """
    Clinical view: captures semantic and ontological similarity between diagnoses.

    - Input: Diagnosis embeddings (cambridgeltl/SapBERT-from-PubMedBERT-fulltext)
    - Embedding construction: Mean pooling of token embeddings (precomputed in DB)
    - Preprocessing: Row-wise L2 normalization (done during embedding generation)
    - Distance metric: Cosine distance
    - Output: Distance matrix

    Note: The database stores cosine SIMILARITY (range [0, 1], higher = more similar).
    This class converts to cosine DISTANCE using: distance = 1 - similarity
    """

    view_type = "clinical"

    def get_terminology_ids(self) -> list:
        """
        Get the sorted list of terminology IDs in the clinical matrix.

        Returns:
            Ordered list of terminology_id values
        """
        return sorted(self.df["terminology_id"].unique().to_list())

    def compute(self) -> np.ndarray:
        """
        Compute clinical distance matrix from precomputed embeddings.

        Loads cosine similarity from database and converts to distance.

        Returns:
            Distance matrix (NxN numpy array)
        """
        # get unique terminology IDs from input dataframe
        terminology_ids = sorted(self.df["terminology_id"].unique().to_list())
        n_diagnoses = len(terminology_ids)

        # create mapping from terminology_id to matrix index
        id_to_idx = {tid: idx for idx, tid in enumerate(terminology_ids)}

        clinical_df = self._get_or_create_clinical_dimension()
        clinical_df = clinical_df.filter(
            pl.col("terminology_id_1").is_in(terminology_ids)
            & pl.col("terminology_id_2").is_in(terminology_ids)
        )

        similarity_matrix = np.zeros((n_diagnoses, n_diagnoses), dtype=np.float32)

        # fill similarity matrix from database values
        for row in clinical_df.iter_rows(named=True):
            id1 = row["terminology_id_1"]
            id2 = row["terminology_id_2"]
            similarity = row["value"]

            idx1 = id_to_idx[id1]
            idx2 = id_to_idx[id2]

            # symmetric matrix
            similarity_matrix[idx1, idx2] = similarity
            similarity_matrix[idx2, idx1] = similarity

        # set diagonal to 1 (perfect similarity with self)
        np.fill_diagonal(similarity_matrix, 1.0)

        # convert similarity to distance: distance = 1 - similarity
        distance_matrix = 1.0 - similarity_matrix

        return distance_matrix

    def get_similarity_matrix(self) -> np.ndarray:
        """
        Get cosine similarity matrix (not distance).

        Returns:
            Similarity matrix (NxN numpy array, range [0, 1])
        """
        distance_matrix = self.get_matrix()
        return 1.0 - distance_matrix

    def get_embeddings(self) -> pl.DataFrame:
        """
        Get embeddings for diagnoses in the dataset.

        Returns:
            DataFrame with terminology_id and embedding columns
        """

        terminology_ids = self.df["terminology_id"].unique().to_list()
        model_name = self.params.get("model", DEFAULT_MODEL)

        con = get_connection()
        result = con.execute(
            """
            SELECT id, code, code_type, description, embedding
            FROM medical_terminologies
            WHERE id = ANY($ids) AND model = $model
            ORDER BY id
            """,
            {"ids": terminology_ids, "model": model_name},
        )

        return result.pl()

    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Get statistics about embeddings.

        Returns:
            Dictionary with embedding statistics
        """
        embeddings_df = self.get_embeddings()
        embeddings = np.array([emb for emb in embeddings_df["embedding"].to_list()])

        return {
            "n_diagnoses": len(embeddings),
            "embedding_dim": len(embeddings[0]) if len(embeddings) > 0 else 0,
            "model": self.params.get("model", DEFAULT_MODEL),
            "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
            "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
        }

    def _get_or_create_clinical_dimension(self):
        if dimension_exists("clinical") is False:
            logger.info("Creating dimensions table...")
            create_dimensions_table_if_not_exists()
            logger.info("Inserting clinical dimensions...")
            insert_clinical_dimensions()
        return read_dimension("clinical")
