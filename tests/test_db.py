import duckdb
import pytest

import vir.db as db_module


@pytest.fixture(autouse=True)
def reset_connection():
    """Ensure module-level connection singletons are clean before and after each test."""
    db_module._read_only_connection = None
    db_module._write_connection = None
    yield
    if db_module._read_only_connection is not None:
        db_module._read_only_connection.close()
        db_module._read_only_connection = None
    if db_module._write_connection is not None:
        db_module._write_connection.close()
        db_module._write_connection = None


@pytest.fixture
def tmp_db(tmp_path):
    """A temporary DuckDB path, patched into the module.

    Creates the database file upfront with a write connection so that
    subsequent read-only connections (the default) find an existing file.
    """
    path = str(tmp_path / "test.db")
    bootstrap = duckdb.connect(path)
    bootstrap.close()
    original = db_module.PROJECT_DB
    db_module.PROJECT_DB = path
    yield path
    db_module.PROJECT_DB = original


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
