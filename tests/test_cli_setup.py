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


class TestPrepareIndicativeTemplate:
    def test_emits_top_coverage_codes_with_labels(self, redirect_processed_dir, tmp_path):
        terminology_csv = redirect_processed_dir / "cid10_ciap2_ab.csv"
        pl.DataFrame(
            {
                "code": ["A1", "B2", "C3"],
                "code_type": ["cid10", "cid10", "cid10"],
                "description": ["alpha", "bravo", "charlie"],
            }
        ).write_csv(terminology_csv)

        cleaned = pl.DataFrame(
            {
                "code": ["A1", "B2", "C3"],
                "code_type": ["cid10"] * 3,
                "quantity": [70, 25, 5],
            }
        )
        cleaned_path = tmp_path / "cleaned.parquet"
        cleaned.write_parquet(cleaned_path)

        result = runner.invoke(
            app,
            [
                "prepare",
                "indicative-template",
                "--dataset",
                str(cleaned_path),
                "--adapter",
                "datasus",
                "--coverage",
                "0.9",
            ],
        )

        assert result.exit_code == 0, result.output
        output_csv = redirect_processed_dir / "indicative_codes_template_datasus.csv"
        assert output_csv.exists()
        template = pl.read_csv(output_csv)
        assert set(template["code"].to_list()) == {"A1", "B2"}
        assert set(template.columns) >= {"code", "code_type", "description", "indicative"}

    def test_works_for_aktin_against_its_icd10gm_terminology(
        self, redirect_processed_dir, tmp_path
    ):
        """AKTIN goes through the same CLI; the adapter swap is the only difference."""
        terminology_csv = redirect_processed_dir / "icd10gm_aktin.csv"
        pl.DataFrame(
            {
                "code": ["M22.0", "S93.4", "A00.0"],
                "code_type": ["icd10gm"] * 3,
                "description": [
                    "Habituelle Luxation der Patella",
                    "Verstauchung und Zerrung des oberen Sprunggelenkes",
                    "Cholera durch Vibrio cholerae O:1, Biovar cholerae",
                ],
            }
        ).write_csv(terminology_csv)

        cleaned = pl.DataFrame(
            {
                "code": ["M22.0", "S93.4", "A00.0"],
                "code_type": ["icd10gm"] * 3,
                "quantity": [80, 15, 5],
            }
        )
        cleaned_path = tmp_path / "cleaned_aktin_germany.parquet"
        cleaned.write_parquet(cleaned_path)

        result = runner.invoke(
            app,
            [
                "prepare",
                "indicative-template",
                "--dataset",
                str(cleaned_path),
                "--adapter",
                "aktin",
                "--coverage",
                "0.9",
            ],
        )

        assert result.exit_code == 0, result.output
        output_csv = redirect_processed_dir / "indicative_codes_template_aktin.csv"
        assert output_csv.exists()
        template = pl.read_csv(output_csv)
        assert set(template["code"].to_list()) == {"M22.0", "S93.4"}
        assert template["code_type"].unique().to_list() == ["icd10gm"]
        descriptions = {row["code"]: row["description"] for row in template.iter_rows(named=True)}
        assert descriptions["M22.0"] == "Habituelle Luxation der Patella"

    def test_errors_when_terminology_csv_missing(self, redirect_processed_dir, tmp_path):
        cleaned = pl.DataFrame({"code": ["A1"], "code_type": ["cid10"], "quantity": [1]})
        cleaned_path = tmp_path / "cleaned.parquet"
        cleaned.write_parquet(cleaned_path)

        result = runner.invoke(
            app,
            [
                "prepare",
                "indicative-template",
                "--dataset",
                str(cleaned_path),
                "--adapter",
                "datasus",
            ],
        )

        assert result.exit_code != 0
        assert "setup terminologies" in result.output
