import duckdb
import pytest

import vir.db as db_module


@pytest.fixture(autouse=True)
def reset_connection():
    """Ensure the module-level connection singleton is clean before and after each test."""
    db_module._connection = None
    yield
    if db_module._connection is not None:
        db_module._connection.close()
        db_module._connection = None


@pytest.fixture
def tmp_db(tmp_path):
    """A temporary DuckDB path, patched into the module."""
    path = str(tmp_path / "test.db")
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
        assert db_module._connection is con
        db_module._close_connection()
        assert db_module._connection is None

    def test_get_connection_after_close_opens_new(self, tmp_db):
        con1 = db_module.get_connection()
        db_module._close_connection()
        con2 = db_module.get_connection()
        assert con2 is not con1
        assert isinstance(con2, duckdb.DuckDBPyConnection)
