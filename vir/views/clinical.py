"""Clinical view: semantic/ontological similarity from embeddings."""

from typing import Any, Dict

import numpy as np
import polars as pl

from vir.config import logger
from vir.db import (
    create_dimensions_table_if_not_exists,
    dimension_exists,
    get_connection,
    insert_clinical_dimensions,
    read_dimension,
)
from vir.views.base import BaseView


def _required_model(params: dict) -> str:
    model = params.get("model")
    if not model:
        raise ValueError(
            "ClinicalView requires `model` in params (e.g. "
            "`params={'model': adapter.default_model}`)."
        )
    return model


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
        terminology_ids = sorted(self.df["terminology_id"].unique().to_list())
        n_diagnoses = len(terminology_ids)

        if n_diagnoses == 0:
            raise ValueError(
                "No terminology IDs found in the dataset. "
                "Make sure the database has embeddings loaded (run: vir setup embeddings) "
                "and that the data file contains codes matching the medical_terminologies table."
            )

        id_to_idx = {tid: idx for idx, tid in enumerate(terminology_ids)}

        clinical_df = self._get_or_create_clinical_dimension(terminology_ids)

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
        model_name = _required_model(self.params)

        # TODO move it out
        con = get_connection()
        result = con.execute(
            """
            SELECT mt.id, mt.code, mt.code_type, mt.description, te.embedding
            FROM terminology_embeddings te
            JOIN medical_terminologies mt ON mt.id = te.terminology_id
            WHERE te.terminology_id = ANY($ids) AND te.model = $model
            ORDER BY mt.id
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
            "model": _required_model(self.params),
            "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
            "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
        }

    def _get_or_create_clinical_dimension(self, terminology_ids: list[int]):
        model_name = _required_model(self.params)
        create_dimensions_table_if_not_exists()
        if dimension_exists("clinical", model_name, terminology_ids) is False:
            logger.info(
                f"Inserting clinical dimensions for model {model_name} "
                f"({len(terminology_ids)} terminology ids)..."
            )
            insert_clinical_dimensions(model_name, terminology_ids)
        return read_dimension("clinical", terminology_ids=terminology_ids, model_name=model_name)
