import json
from unittest.mock import MagicMock, patch

import duckdb
import numpy as np
import polars as pl
import pytest

import vir.db as db_module
from vir import db
from vir.clusters import _build_labels_count, build_cluster_profiles, explain_clusters
from vir.db import _city_metadata


class TestGetConnection:
    def test_returns_connection(self, tmp_db):
        con = db_module.get_connection()
        assert isinstance(con, duckdb.DuckDBPyConnection)

    def test_singleton_same_object(self, tmp_db):
        con1 = db_module.get_connection()
        con2 = db_module.get_connection()
        assert con1 is con2

    def test_connection_is_functional(self, tmp_db):
        con = db_module.get_connection()
        result = con.execute("SELECT 42 AS answer").fetchone()
        assert result == (42,)

    def test_raises_after_retries_exhausted(self, tmp_path, monkeypatch):
        """Should raise ConnectionException when the db is unreachable after retries."""
        monkeypatch.setattr(db_module, "PROJECT_DB", str(tmp_path / "test.db"))

        def always_fail(path, **kwargs):
            raise duckdb.ConnectionException("locked")

        monkeypatch.setattr(duckdb, "connect", always_fail)

        with pytest.raises(duckdb.ConnectionException):
            db_module.get_connection(retries=2, delay=0)

    def test_close_connection_resets_singleton(self, tmp_db):
        con = db_module.get_connection()
        assert db_module._read_only_connection is con
        db._close_read_only_connection()
        assert db_module._read_only_connection is None

    def test_get_connection_after_close_opens_new(self, tmp_db):
        con1 = db_module.get_connection()
        db._close_read_only_connection()
        con2 = db_module.get_connection()
        assert con2 is not con1
        assert isinstance(con2, duckdb.DuckDBPyConnection)


class TestRunsAdapterMigration:
    def _columns(self, connection) -> set[str]:
        rows = connection.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'runs'"
        ).fetchall()
        return {row[0] for row in rows}

    def test_fresh_schema_has_adapter_column(self, tmp_db):
        db_module.create_embeddings_table_if_not_exists()
        db_module.create_results_schema()
        connection = db_module.get_connection()
        assert "adapter" in self._columns(connection)

    def test_existing_rows_are_backfilled_to_datasus(self, tmp_db):
        db_module.create_embeddings_table_if_not_exists()
        db_module.create_results_schema()
        run_id = db_module.insert_run(
            dataset_name="legacy",
            city="Bauru",
            region="Sudeste",
            population_range="250 - 500 mil",
            year_min=2022,
            year_max=2024,
            snf_k=10,
            n_clusters=3,
        )
        db_module.create_results_schema()
        adapter_value = (
            db_module.get_connection()
            .execute("SELECT adapter FROM runs WHERE run_id = ?", [run_id])
            .fetchone()[0]
        )
        assert adapter_value == "datasus"

    def test_backfill_does_not_overwrite_existing_adapter_values(self, tmp_db):
        db_module.create_embeddings_table_if_not_exists()
        db_module.create_results_schema()
        run_id = db_module.insert_run(
            dataset_name="aktin_germany",
            city="germany",
            region=None,
            population_range=None,
            year_min=2018,
            year_max=2023,
            snf_k=10,
            n_clusters=3,
        )
        db_module.get_connection(write=True).execute(
            "UPDATE runs SET adapter = 'aktin' WHERE run_id = ?", [run_id]
        )
        db_module.create_results_schema()
        adapter_value = (
            db_module.get_connection()
            .execute("SELECT adapter FROM runs WHERE run_id = ?", [run_id])
            .fetchone()[0]
        )
        assert adapter_value == "aktin"


class TestFetchClusterProfilesForRuns:
    @pytest.fixture
    def seeded_db(self, tmp_db):
        db_module.create_embeddings_table_if_not_exists()
        run_id_1 = db_module.insert_run(
            dataset_name="test_a",
            city="City A",
            region="North",
            population_range="> 500 mil",
            year_min=2022,
            year_max=2024,
            snf_k=10,
            n_clusters=3,
        )
        run_id_2 = db_module.insert_run(
            dataset_name="test_b",
            city="City B",
            region="South",
            population_range="250 - 500 mil",
            year_min=2022,
            year_max=2024,
            snf_k=10,
            n_clusters=2,
        )
        db_module.get_connection(write=True).executemany(
            "INSERT INTO cluster_profiles (run_id, cluster_id, n_codes) VALUES (?, ?, ?)",
            [
                (run_id_1, 0, 10),
                (run_id_1, 1, 20),
                (run_id_1, 2, 5),
                (run_id_2, 0, 15),
                (run_id_2, 1, 25),
            ],
        )
        return run_id_1, run_id_2

    def test_empty_input_returns_empty_dataframe(self, tmp_db):
        result = db_module.fetch_cluster_profiles_for_runs([])
        assert result.is_empty()

    def test_returns_profiles_for_single_run(self, seeded_db):
        run_id_1, _ = seeded_db
        result = db_module.fetch_cluster_profiles_for_runs([run_id_1])
        assert len(result) == 3
        assert result["run_id"].unique().to_list() == [run_id_1]

    def test_returns_profiles_for_multiple_runs(self, seeded_db):
        run_id_1, run_id_2 = seeded_db
        result = db_module.fetch_cluster_profiles_for_runs([run_id_1, run_id_2])
        assert len(result) == 5

    def test_ordered_by_run_id_then_cluster_id(self, seeded_db):
        run_id_1, run_id_2 = seeded_db
        result = db_module.fetch_cluster_profiles_for_runs([run_id_1, run_id_2])
        assert result["run_id"].to_list() == [run_id_1, run_id_1, run_id_1, run_id_2, run_id_2]
        assert result["cluster_id"].to_list() == [0, 1, 2, 0, 1]

    def test_unknown_run_id_returns_empty(self, seeded_db):
        result = db_module.fetch_cluster_profiles_for_runs([99999])
        assert result.is_empty()


class TestDeleteRun:
    @pytest.fixture
    def seeded_run(self, tmp_db):
        db_module.create_embeddings_table_if_not_exists()
        db_module.create_results_schema()
        run_id = db_module.insert_run(
            dataset_name="test_delete",
            city="Manaus",
            region="Norte",
            population_range="> 500 mil",
            year_min=2022,
            year_max=2024,
            snf_k=10,
            n_clusters=2,
        )
        return run_id

    @pytest.fixture
    def seeded_run_with_terminology(self, seeded_run):
        connection = db_module.get_connection(write=True)
        connection.execute(
            "INSERT INTO medical_terminologies (id, code, code_type, description, model, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [1001, "A00", "cid10", "Cólera", "test-model", [0.1, 0.2, 0.3]],
        )
        connection.execute(
            "INSERT INTO cluster_assignments (run_id, terminology_id, cluster_id) VALUES (?, ?, ?)",
            [seeded_run, 1001, 0],
        )
        return seeded_run, 1001

    def test_returns_false_when_run_does_not_exist(self, tmp_db):
        db_module.create_embeddings_table_if_not_exists()
        db_module.create_results_schema()
        assert db_module.delete_run(9999) is False

    def test_returns_true_and_removes_run(self, seeded_run):
        assert db_module.delete_run(seeded_run) is True
        assert db_module.fetch_runs().is_empty()

    def test_cascades_to_cluster_assignments(self, seeded_run_with_terminology):
        run_id, _ = seeded_run_with_terminology
        db_module.delete_run(run_id)
        remaining = (
            db_module.get_connection()
            .execute("SELECT COUNT(*) FROM cluster_assignments WHERE run_id = ?", [run_id])
            .fetchone()[0]
        )
        assert remaining == 0

    def test_cascades_to_cluster_profiles(self, seeded_run):
        db_module.get_connection(write=True).execute(
            "INSERT INTO cluster_profiles (run_id, cluster_id, n_codes) VALUES (?, ?, ?)",
            [seeded_run, 0, 7],
        )
        db_module.delete_run(seeded_run)
        remaining = db_module.fetch_cluster_profiles(seeded_run)
        assert remaining.is_empty()

    def test_cascades_to_run_multi_k(self, seeded_run):
        db_module.insert_run_multi_k(
            seeded_run,
            [{"n_clusters": 2, "silhouette": 0.5, "largest_cluster": 1, "median_size": 1.0}],
        )
        db_module.delete_run(seeded_run)
        assert db_module.fetch_run_multi_k(seeded_run).is_empty()

    def test_cascades_to_run_leiden(self, seeded_run):
        db_module.insert_run_leiden(seeded_run, [{"resolution": 1.0, "n_communities": 5}])
        db_module.delete_run(seeded_run)
        assert db_module.fetch_run_leiden(seeded_run).is_empty()

    def test_does_not_touch_other_runs(self, seeded_run):
        other_run = db_module.insert_run(
            dataset_name="other",
            city="Bauru",
            region="Sudeste",
            population_range="250 - 500 mil",
            year_min=2022,
            year_max=2024,
            snf_k=10,
            n_clusters=3,
        )
        db_module.get_connection(write=True).execute(
            "INSERT INTO cluster_profiles (run_id, cluster_id, n_codes) VALUES (?, ?, ?)",
            [other_run, 0, 11],
        )
        db_module.delete_run(seeded_run)

        remaining_runs = db_module.fetch_runs()
        assert remaining_runs["run_id"].to_list() == [other_run]
        assert not db_module.fetch_cluster_profiles(other_run).is_empty()

    def test_removes_parquet_sidecar_when_present(self, seeded_run, tmp_path, monkeypatch):
        monkeypatch.setattr(db_module, "DATA_DIR", tmp_path)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        sidecar = runs_dir / f"{seeded_run}.parquet"
        sidecar.write_bytes(b"placeholder")

        db_module.delete_run(seeded_run)

        assert not sidecar.exists()

    def test_idempotent_when_parquet_missing(self, seeded_run, tmp_path, monkeypatch):
        monkeypatch.setattr(db_module, "DATA_DIR", tmp_path)
        assert db_module.delete_run(seeded_run) is True


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

        with patch("vir.clusters.upsert_cluster_profile", side_effect=capture_upsert):
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

        with patch("vir.clusters.upsert_cluster_profile", side_effect=capture_upsert):
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
            patch("vir.clusters.fetch_cluster_profiles", return_value=profiles),
            patch("vir.clusters.get_connection", return_value=mock_con),
            patch("vir.clusters.upsert_cluster_profile", side_effect=upsert_mock),
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
            patch("vir.clusters.fetch_cluster_profiles", return_value=profiles),
            patch("vir.clusters.get_connection", return_value=mock_con),
            patch("vir.clusters.upsert_cluster_profile", side_effect=capture),
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
            patch("vir.clusters.fetch_cluster_profiles", return_value=empty_profiles),
            patch("vir.clusters.upsert_cluster_profile") as mock_upsert,
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
