import polars as pl
import pytest
from typer.testing import CliRunner

import vir.config as config_module
from vir.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def redirect_processed_dir(tmp_path, monkeypatch):
    """Make every test in this module write into a private tmp processed/ dir."""
    monkeypatch.setattr(config_module, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr("vir.cli.PROCESSED_DATA_DIR", tmp_path, raising=False)
    return tmp_path


class TestSetupTerminologiesAdapter:
    def test_datasus_writes_cid10_ciap2_ab_csv(self, redirect_processed_dir):
        result = runner.invoke(app, ["setup", "terminologies", "--adapter", "datasus"])

        assert result.exit_code == 0, result.output
        csv_path = redirect_processed_dir / "cid10_ciap2_ab.csv"
        assert csv_path.exists()
        labels = pl.read_csv(csv_path)
        assert set(labels["code_type"].unique().to_list()) == {"cid10", "ciap2", "ab"}

    def test_aktin_writes_icd10gm_csv(self, redirect_processed_dir):
        result = runner.invoke(app, ["setup", "terminologies", "--adapter", "aktin"])

        assert result.exit_code == 0, result.output
        csv_path = redirect_processed_dir / "icd10gm_aktin.csv"
        assert csv_path.exists()
        labels = pl.read_csv(csv_path)
        assert labels["code_type"].unique().to_list() == ["icd10gm"]
        assert labels.height > 10_000

    def test_errors_when_neither_flag_nor_env_var_is_provided(self):
        result = runner.invoke(app, ["setup", "terminologies"], env={"VIROLA_ADAPTER": ""})

        assert result.exit_code != 0
        assert "adapter" in result.output.lower()

    def test_env_var_used_when_flag_omitted(self, redirect_processed_dir):
        result = runner.invoke(app, ["setup", "terminologies"], env={"VIROLA_ADAPTER": "aktin"})

        assert result.exit_code == 0, result.output
        assert (redirect_processed_dir / "icd10gm_aktin.csv").exists()

    def test_errors_on_unknown_adapter(self):
        result = runner.invoke(app, ["setup", "terminologies", "--adapter", "mars"])

        assert result.exit_code != 0
        assert "mars" in result.output.lower() or "unknown" in result.output.lower()
