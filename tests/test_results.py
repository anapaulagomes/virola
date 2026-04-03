import json
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from vir.results import (
    _build_labels_count,
    _city_metadata,
    build_cluster_profiles,
    explain_clusters,
)


class TestCityMetadata:
    @pytest.mark.parametrize(
        "dataset_name,expected",
        [
            (
                "base_aesop_raw_extracao_01042026_salvador",
                {"city": "Salvador", "region": "Nordeste", "population_range": "> 500 mil"},
            ),
            (
                "base_aesop_raw_extracao_01042026_belo_horizonte",
                {"city": "Belo Horizonte", "region": "Sudeste", "population_range": "> 500 mil"},
            ),
            (
                "base_aesop_raw_extracao_01042026_rio_branco",
                {"city": "Rio Branco", "region": "Norte", "population_range": "250 - 500 mil"},
            ),
            (
                "SALVADOR_uppercase",
                {"city": "Salvador", "region": "Nordeste", "population_range": "> 500 mil"},
            ),
            (
                "base_aesop_raw_extracao_01042026_all_cities",
                {"city": None, "region": None, "population_range": None},
            ),
            (
                "unknown_dataset",
                {"city": None, "region": None, "population_range": None},
            ),
        ],
    )
    def test_known_cities_and_fallback(self, dataset_name, expected):
        assert _city_metadata(dataset_name) == expected

    def test_match_is_case_insensitive(self):
        result = _city_metadata("PORTO_ALEGRE")
        assert result["city"] == "Porto Alegre"

    def test_unknown_returns_all_none(self):
        result = _city_metadata("xpto_city_does_not_exist")
        assert result == {"city": None, "region": None, "population_range": None}


class TestBuildLabelsCount:
    def test_uses_member_centralities_when_present(self):
        row = {
            "explanation": json.dumps(
                {
                    "member_centralities": [
                        {"description": "hypertension", "centrality": 0.9},
                        {"description": "diabetes", "centrality": 0.7},
                    ]
                }
            ),
            "top_codes": json.dumps([]),
        }
        result = _build_labels_count(row)
        assert result == {"hypertension": 0.9, "diabetes": 0.7}

    def test_falls_back_to_top_codes_when_no_centralities(self):
        row = {
            "explanation": json.dumps({}),
            "top_codes": json.dumps(
                [
                    {"code": "I10", "code_type": "cid10", "total": 500},
                    {"code": "K86", "code_type": "ciap2", "total": 300},
                ]
            ),
        }
        result = _build_labels_count(row)
        assert result == {"I10 (cid10)": 500, "K86 (ciap2)": 300}

    def test_falls_back_to_top_codes_when_explanation_none(self):
        row = {
            "explanation": None,
            "top_codes": json.dumps([{"code": "Z760", "code_type": "cid10", "total": 100}]),
        }
        result = _build_labels_count(row)
        assert result == {"Z760 (cid10)": 100}

    def test_skips_member_centralities_without_description(self):
        row = {
            "explanation": json.dumps(
                {
                    "member_centralities": [
                        {"description": "hypertension", "centrality": 0.9},
                        {"description": "", "centrality": 0.5},
                        {"centrality": 0.3},
                    ]
                }
            ),
            "top_codes": json.dumps([]),
        }
        result = _build_labels_count(row)
        assert "hypertension" in result
        assert len(result) == 1

    def test_top_codes_capped_at_ten(self):
        codes = [{"code": f"C{i:02d}", "code_type": "cid10", "total": 100 - i} for i in range(15)]
        row = {"explanation": None, "top_codes": json.dumps(codes)}
        result = _build_labels_count(row)
        assert len(result) == 10

    def test_member_centralities_capped_at_ten(self):
        centralities = [
            {"description": f"condition_{i}", "centrality": 0.9 - i * 0.05} for i in range(15)
        ]
        row = {
            "explanation": json.dumps({"member_centralities": centralities}),
            "top_codes": json.dumps([]),
        }
        result = _build_labels_count(row)
        assert len(result) == 10


@pytest.fixture(scope="class")
def cluster_df() -> pl.DataFrame:
    """Minimal prepared dataframe with two clusters (0 and 1)."""
    rows = []
    codes = [
        (1, "I10", "cid10"),
        (2, "K86", "ciap2"),
        (3, "Z760", "cid10"),
        (4, "ABP005", "ab"),
    ]
    for terminology_id, code, code_type in codes:
        for week in range(1, 4):
            rows.append(
                {
                    "year": 2022,
                    "week": week,
                    "sex": "M" if week % 2 == 0 else "F",
                    "age_group": "adult",
                    "code_type": code_type,
                    "code": code,
                    "quantity": 10,
                    "terminology_id": terminology_id,
                    "terminology_label": f"{code}:{code_type}",
                }
            )
    return pl.DataFrame(rows)


class TestBuildClusterProfiles:
    @pytest.fixture(scope="class")
    def captured_calls(self, cluster_df):
        """Run build_cluster_profiles once and capture all upsert kwargs, keyed by cluster_id."""
        common_ids = [1, 2, 3, 4]
        fused_labels = np.array([0, 0, 1, 1])
        captured = {}

        def capture_upsert(**kwargs):
            captured[kwargs["cluster_id"]] = kwargs

        with patch("vir.results.upsert_cluster_profile", side_effect=capture_upsert):
            build_cluster_profiles(
                run_id=1, df=cluster_df, common_ids=common_ids, fused_labels=fused_labels
            )
        return captured

    def test_calls_upsert_for_each_cluster(self, captured_calls):
        assert set(captured_calls.keys()) == {0, 1}

    def test_n_codes_per_cluster(self, captured_calls):
        assert captured_calls[0]["n_codes"] == 2
        assert captured_calls[1]["n_codes"] == 2

    def test_total_cases_per_cluster(self, captured_calls):
        # cluster 0: codes 1 and 2, 3 weeks each, quantity 10 → 60 total
        assert captured_calls[0]["total_cases"] == 60
        assert captured_calls[1]["total_cases"] == 60

    def test_top_codes_are_valid_json(self, captured_calls):
        top_codes = json.loads(captured_calls[0]["top_codes"])
        assert isinstance(top_codes, list)
        assert all("code" in entry and "total" in entry for entry in top_codes)

    def test_run_id_is_passed_to_upsert(self, cluster_df):
        """run_id must be forwarded to every upsert_cluster_profile call."""
        common_ids = [1, 2, 3, 4]
        fused_labels = np.array([0, 0, 1, 1])
        captured_run_ids = set()

        def capture_upsert(**kwargs):
            captured_run_ids.add(kwargs["run_id"])

        with patch("vir.results.upsert_cluster_profile", side_effect=capture_upsert):
            build_cluster_profiles(
                run_id=42, df=cluster_df, common_ids=common_ids, fused_labels=fused_labels
            )

        assert captured_run_ids == {42}


def _make_distance_matrix(n: int, base: float = 0.5) -> np.ndarray:
    """Symmetric distance matrix with zeros on diagonal."""
    rng = np.random.default_rng(42)
    matrix = rng.random((n, n)) * base
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0.0)
    return matrix.astype(np.float32)


class TestExplainClusters:
    @pytest.fixture
    def setup_explain(self):
        """Three codes in cluster 0, two in cluster 1."""
        common_ids = [1, 2, 3, 4, 5]
        fused_labels = np.array([0, 0, 0, 1, 1])

        terminology = pl.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "code": ["I10", "K86", "Z760", "ABP005", "J06"],
                "code_type": ["cid10", "ciap2", "cid10", "ab", "cid10"],
                "description": [
                    "hypertension",
                    "hypertension uncomplicated",
                    "repeat prescription",
                    "hypertension program",
                    "acute upper respiratory",
                ],
            }
        )

        profiles = pl.DataFrame(
            {"cluster_id": [0, 1], "run_id": [1, 1], "explanation": [None, None]}
        )

        return common_ids, fused_labels, terminology, profiles

    @staticmethod
    def _run_explain(
        common_ids, fused_labels, terminology, profiles, upsert_mock, **distance_overrides
    ):
        n = len(common_ids)
        clinical = distance_overrides.get("clinical", _make_distance_matrix(n, base=0.2))
        demographic = distance_overrides.get("demographic", _make_distance_matrix(n, base=0.8))
        temporal = distance_overrides.get("temporal", _make_distance_matrix(n, base=0.5))

        mock_con = MagicMock()
        mock_con.execute.return_value.pl.return_value = terminology

        with (
            patch("vir.results.fetch_cluster_profiles", return_value=profiles),
            patch("vir.results.get_connection", return_value=mock_con),
            patch("vir.results.upsert_cluster_profile", side_effect=upsert_mock),
        ):
            explain_clusters(
                run_id=1,
                common_ids=common_ids,
                fused_labels=fused_labels,
                clinical_distance=clinical,
                demographic_distance=demographic,
                temporal_distance=temporal,
            )

    @staticmethod
    def _capture_explanations(
        common_ids, fused_labels, terminology, profiles, **distance_overrides
    ):
        """Run explain_clusters and return decoded explanation dict per cluster_id."""
        captured = {}

        def capture(**kwargs):
            if "explanation" in kwargs:
                captured[kwargs["cluster_id"]] = json.loads(kwargs["explanation"])

        TestExplainClusters._run_explain(
            common_ids, fused_labels, terminology, profiles, capture, **distance_overrides
        )
        return captured

    def test_upserts_explanation_for_each_cluster(self, setup_explain):
        common_ids, fused_labels, terminology, profiles = setup_explain
        captured = self._capture_explanations(common_ids, fused_labels, terminology, profiles)
        assert set(captured.keys()) == {0, 1}

    def test_explanation_has_required_keys(self, setup_explain):
        common_ids, fused_labels, terminology, profiles = setup_explain
        captured = self._capture_explanations(common_ids, fused_labels, terminology, profiles)

        for explanation in captured.values():
            assert "view_similarity" in explanation
            assert "primary_driver" in explanation
            assert "central_code" in explanation
            assert "member_centralities" in explanation

    def test_primary_driver_is_valid_view_name(self, setup_explain):
        common_ids, fused_labels, terminology, profiles = setup_explain
        captured = self._capture_explanations(common_ids, fused_labels, terminology, profiles)

        for explanation in captured.values():
            assert explanation["primary_driver"] in {"clinical", "demographic", "temporal"}

    def test_primary_driver_matches_highest_similarity(self, setup_explain):
        """Cluster with low clinical distances must identify clinical as primary driver."""
        common_ids, fused_labels, terminology, profiles = setup_explain
        n = len(common_ids)

        # Make clinical very similar (low distances), others very dissimilar
        clinical = np.ones((n, n), dtype=np.float32) * 0.01
        np.fill_diagonal(clinical, 0.0)
        demographic = np.ones((n, n), dtype=np.float32) * 0.99
        np.fill_diagonal(demographic, 0.0)
        temporal = np.ones((n, n), dtype=np.float32) * 0.99
        np.fill_diagonal(temporal, 0.0)

        captured = self._capture_explanations(
            common_ids,
            fused_labels,
            terminology,
            profiles,
            clinical=clinical,
            demographic=demographic,
            temporal=temporal,
        )

        for explanation in captured.values():
            assert explanation["primary_driver"] == "clinical"

    def test_singletons_are_skipped(self):
        """Clusters with only one member must not produce an explanation."""
        common_ids = [1, 2]
        fused_labels = np.array([0, 1])  # each code alone in its cluster

        terminology = pl.DataFrame(
            {
                "id": [1, 2],
                "code": ["I10", "K86"],
                "code_type": ["cid10", "ciap2"],
                "description": ["hypertension", "hypertension uncomplicated"],
            }
        )
        profiles = pl.DataFrame(
            {"cluster_id": [0, 1], "run_id": [1, 1], "explanation": [None, None]}
        )

        captured = {}

        def capture(**kwargs):
            if "explanation" in kwargs:
                captured[kwargs["cluster_id"]] = kwargs

        n = len(common_ids)
        distance = _make_distance_matrix(n)
        mock_con = MagicMock()
        mock_con.execute.return_value.pl.return_value = terminology

        with (
            patch("vir.results.fetch_cluster_profiles", return_value=profiles),
            patch("vir.results.get_connection", return_value=mock_con),
            patch("vir.results.upsert_cluster_profile", side_effect=capture),
        ):
            explain_clusters(
                run_id=1,
                common_ids=common_ids,
                fused_labels=fused_labels,
                clinical_distance=distance,
                demographic_distance=distance,
                temporal_distance=distance,
            )

        assert len(captured) == 0

    def test_skips_when_no_profiles(self):
        """Must return early without calling upsert when profiles are empty."""
        empty_profiles = pl.DataFrame(
            {"cluster_id": [], "run_id": [], "explanation": []},
            schema={"cluster_id": pl.Int64, "run_id": pl.Int64, "explanation": pl.String},
        )
        with (
            patch("vir.results.fetch_cluster_profiles", return_value=empty_profiles),
            patch("vir.results.upsert_cluster_profile") as mock_upsert,
        ):
            explain_clusters(
                run_id=1,
                common_ids=[1, 2],
                fused_labels=np.array([0, 0]),
                clinical_distance=np.zeros((2, 2)),
                demographic_distance=np.zeros((2, 2)),
                temporal_distance=np.zeros((2, 2)),
            )

        mock_upsert.assert_not_called()
