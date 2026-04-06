from pathlib import Path

import polars as pl
import pytest

from vir.notebook_widgets import (
    load_and_prepare_dataframe,
    load_indicative_codes,
)


@pytest.fixture
def indicative_csv(tmp_path) -> Path:
    csv = tmp_path / "indicative_codes.csv"
    csv.write_text(
        "code,code_type,quantity,cum_pct,indicative,description\n"
        "I10,cid10,1000,0.1,yes,hypertension\n"
        "Z760,cid10,900,0.2,no,repeat prescription\n"
        "ABP005,ab,800,0.3,yes,hypertension program\n"
        "Z000,cid10,700,0.4,no,general exam\n"
        "K86,ciap2,600,0.5,yes,hypertension uncomplicated\n"
    )
    return csv


@pytest.fixture
def prepared_parquet(tmp_path) -> Path:
    """Write a minimal prepared parquet file (already has terminology columns, no admin cols)."""
    rows = []
    codes = [
        ("I10", "cid10", "I10:cid10"),
        ("Z760", "cid10", "Z760:cid10"),
        ("ABP005", "ab", "ABP005:ab"),
        ("Z000", "cid10", "Z000:cid10"),
        ("K86", "ciap2", "K86:ciap2"),
    ]
    for terminology_id, (code, code_type, label) in enumerate(codes, start=1):
        for year in [2022, 2023]:
            for week in range(1, 14):
                rows.append(
                    {
                        "year": year,
                        "week": week,
                        "sex": "M",
                        "age_group": "adult",
                        "code_type": code_type,
                        "code": code,
                        "quantity": 10,
                        "terminology_id": terminology_id,
                        "terminology_label": label,
                    }
                )
    path = tmp_path / "prepared_base_aesop_raw_extracao_01042026_salvador.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


class TestLoadIndicativeCodes:
    def test_disabled_returns_empty_list_and_false(self):
        codes, enabled = load_indicative_codes(enabled=False)
        assert codes == []
        assert enabled is False

    def test_enabled_returns_no_codes_for_exclusion(self, indicative_csv, monkeypatch):
        monkeypatch.setattr("vir.notebook_widgets.INDICATIVE_CODES_FILE", indicative_csv)
        codes, enabled = load_indicative_codes(enabled=True)

        assert enabled is True
        assert "Z760:cid10" in codes
        assert "Z000:cid10" in codes

    def test_enabled_does_not_include_yes_codes(self, indicative_csv, monkeypatch):
        monkeypatch.setattr("vir.notebook_widgets.INDICATIVE_CODES_FILE", indicative_csv)
        codes, _ = load_indicative_codes(enabled=True)

        assert "I10:cid10" not in codes
        assert "ABP005:ab" not in codes
        assert "K86:ciap2" not in codes

    def test_enabled_count_matches_no_rows(self, indicative_csv, monkeypatch):
        monkeypatch.setattr("vir.notebook_widgets.INDICATIVE_CODES_FILE", indicative_csv)
        codes, _ = load_indicative_codes(enabled=True)

        assert len(codes) == 2  # Z760 and Z000


class TestLoadAndPrepareDataframe:
    def test_year_min_max_filter(self, prepared_parquet):
        df = load_and_prepare_dataframe(
            file_path=str(prepared_parquet),
            year_min=2023,
            year_max=2023,
        )
        assert df["year"].unique().to_list() == [2023]

    def test_year_min_only(self, prepared_parquet):
        df = load_and_prepare_dataframe(
            file_path=str(prepared_parquet),
            year_min=2023,
            year_max=None,
        )
        assert all(year >= 2023 for year in df["year"].to_list())

    def test_year_max_only(self, prepared_parquet):
        df = load_and_prepare_dataframe(
            file_path=str(prepared_parquet),
            year_min=None,
            year_max=2022,
        )
        assert all(year <= 2022 for year in df["year"].to_list())

    def test_no_year_filter_keeps_all_years(self, prepared_parquet):
        df = load_and_prepare_dataframe(
            file_path=str(prepared_parquet),
            year_min=None,
            year_max=None,
        )
        assert set(df["year"].unique().to_list()) == {2022, 2023}

    def test_temporal_coverage_filter_applied(self, tmp_path):
        """Codes with fewer than MIN_WEEKS observations must be dropped."""
        rows = [
            {
                "year": 2022,
                "week": week,
                "sex": "M",
                "age_group": "adult",
                "code_type": "cid10",
                "code": "I10",
                "quantity": 10,
                "terminology_id": 1,
                "terminology_label": "I10:cid10",
            }
            for week in range(1, 5)  # only 4 weeks — below MIN_WEEKS (12)
        ]
        path = tmp_path / "prepared_sparse.parquet"
        pl.DataFrame(rows).write_parquet(path)

        df = load_and_prepare_dataframe(file_path=str(path), year_min=None, year_max=None)
        assert df.is_empty()
