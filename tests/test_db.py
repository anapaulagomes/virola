import duckdb
import pytest

import vir.db as db_module


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
        db_module._close_read_only_connection()
        assert db_module._read_only_connection is None

    def test_get_connection_after_close_opens_new(self, tmp_db):
        con1 = db_module.get_connection()
        db_module._close_read_only_connection()
        con2 = db_module.get_connection()
        assert con2 is not con1
        assert isinstance(con2, duckdb.DuckDBPyConnection)


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
