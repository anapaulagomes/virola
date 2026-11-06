import numpy as np
import polars as pl
import pytest

import vir.views.base as base_module
from vir.views.demographic_features import DemographicFeaturesView


@pytest.fixture(autouse=True)
def isolated_views_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(base_module, "VIEWS_CACHE_DIR", tmp_path / "views")


def _two_diagnoses() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "terminology_id": [1, 1, 2],
            "sex": ["Feminino", "Masculino", "Masculino"],
            "age_group": ["16 a 59 anos", "Maior que 60 anos", "< 1 ano"],
            "quantity": [3, 1, 4],
        }
    )


def test_features_are_ratios_of_each_diagnosis_total():
    view = DemographicFeaturesView(df=_two_diagnoses(), dataset_name="test", params={})

    features = view.get_features()

    female_ratio = features.filter(pl.col("terminology_id") == 1)["female_ratio"].item()
    child_ratio = features.filter(pl.col("terminology_id") == 2)["child_ratio"].item()
    assert female_ratio == pytest.approx(0.75)
    assert child_ratio == pytest.approx(1.0)


def test_compute_returns_a_symmetric_distance_matrix_with_zero_diagonal():
    view = DemographicFeaturesView(df=_two_diagnoses(), dataset_name="test", params={})

    matrix = view.compute()

    assert matrix.shape == (2, 2)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)
    assert matrix[0, 1] > 0
