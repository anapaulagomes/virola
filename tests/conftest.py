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
