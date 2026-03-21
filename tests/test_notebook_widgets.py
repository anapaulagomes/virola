"""Tests for vir.notebook_widgets."""

from pathlib import Path

import polars as pl
import pytest

from vir.notebook_widgets import (
    extract_dataset_name,
    load_and_prepare_dataframe,
    load_indicative_codes,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
def terminology_labels() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "code": ["I10", "Z760", "ABP005", "Z000", "K86"],
            "code_type": ["cid10", "cid10", "ab", "cid10", "ciap2"],
            "description": [
                "hypertension",
                "repeat prescription",
                "hypertension program",
                "general exam",
                "hypertension uncomplicated",
            ],
            "label": ["I10:cid10", "Z760:cid10", "ABP005:ab", "Z000:cid10", "K86:ciap2"],
        }
    )


@pytest.fixture
def raw_parquet(tmp_path, terminology_labels) -> Path:
    """Write a minimal raw parquet file with all required columns."""
    rows = []
    for i, row in enumerate(terminology_labels.iter_rows(named=True)):
        for year in [2022, 2023]:
            for week in range(1, 14):  # 13 weeks to pass min_weeks filter
                rows.append(
                    {
                        "unique_id": f"uid_{i}",
                        "state": "SP",
                        "city": "Salvador",
                        "region": "NE",
                        "population_range": "large",
                        "year": year,
                        "week": week,
                        "sex": "M",
                        "age_group": "adult",
                        "code_type": row["code_type"],
                        "code": row["code"],
                        "quantity": 10,
                    }
                )
    df = pl.DataFrame(rows)
    path = tmp_path / "cleaned_20municipios_salvador.parquet"
    df.write_parquet(path)
    return path


# ---------------------------------------------------------------------------
# load_indicative_codes
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# load_and_prepare_dataframe
# ---------------------------------------------------------------------------


class TestLoadAndPrepareDataframe:
    def test_year_min_max_filter(self, raw_parquet, terminology_labels):
        df = load_and_prepare_dataframe(
            file_path=str(raw_parquet),
            terminology_labels=terminology_labels,
            year_min=2023,
            year_max=2023,
        )
        assert df["year"].unique().to_list() == [2023]

    def test_year_min_only(self, raw_parquet, terminology_labels):
        df = load_and_prepare_dataframe(
            file_path=str(raw_parquet),
            terminology_labels=terminology_labels,
            year_min=2023,
            year_max=None,
        )
        assert all(y >= 2023 for y in df["year"].to_list())

    def test_year_max_only(self, raw_parquet, terminology_labels):
        df = load_and_prepare_dataframe(
            file_path=str(raw_parquet),
            terminology_labels=terminology_labels,
            year_min=None,
            year_max=2022,
        )
        assert all(y <= 2022 for y in df["year"].to_list())

    def test_exclude_codes_removes_non_indicative(self, raw_parquet, terminology_labels):
        exclude = ["Z760:cid10", "Z000:cid10"]
        df = load_and_prepare_dataframe(
            file_path=str(raw_parquet),
            terminology_labels=terminology_labels,
            exclude_codes=exclude,
        )
        labels = df["terminology_label"].unique().to_list()
        assert "Z760:cid10" not in labels
        assert "Z000:cid10" not in labels

    def test_exclude_codes_keeps_yes_codes(self, raw_parquet, terminology_labels):
        """Codes labeled 'yes' (not in exclude list) must remain in the dataframe."""
        exclude = ["Z760:cid10", "Z000:cid10"]
        df = load_and_prepare_dataframe(
            file_path=str(raw_parquet),
            terminology_labels=terminology_labels,
            exclude_codes=exclude,
        )
        labels = df["terminology_label"].unique().to_list()
        assert "I10:cid10" in labels
        assert "ABP005:ab" in labels
        assert "K86:ciap2" in labels

    def test_no_exclude_codes_keeps_all(self, raw_parquet, terminology_labels):
        df = load_and_prepare_dataframe(
            file_path=str(raw_parquet),
            terminology_labels=terminology_labels,
            exclude_codes=None,
        )
        labels = set(df["terminology_label"].unique().to_list())
        expected = {"I10:cid10", "Z760:cid10", "ABP005:ab", "Z000:cid10", "K86:ciap2"}
        assert expected == labels

    def test_codes_not_in_csv_are_kept_when_filtering(self, tmp_path, terminology_labels):
        """Codes not present in the indicative CSV must NOT be removed when filtering."""
        # Build a parquet with an extra code not in the indicative CSV
        extra_labels = terminology_labels.vstack(
            pl.DataFrame(
                {
                    "id": [99],
                    "code": ["EXTRA"],
                    "code_type": ["ab"],
                    "description": ["some extra code"],
                    "label": ["EXTRA:ab"],
                }
            )
        )
        rows = []
        for row in extra_labels.iter_rows(named=True):
            for year in [2022, 2023]:
                for week in range(1, 14):
                    rows.append(
                        {
                            "unique_id": "uid",
                            "state": "SP",
                            "city": "Salvador",
                            "region": "NE",
                            "population_range": "large",
                            "year": year,
                            "week": week,
                            "sex": "M",
                            "age_group": "adult",
                            "code_type": row["code_type"],
                            "code": row["code"],
                            "quantity": 10,
                        }
                    )
        path = tmp_path / "cleaned_20municipios_extra.parquet"
        pl.DataFrame(rows).write_parquet(path)

        # Only exclude the "no" codes from the CSV — EXTRA is not in the CSV
        exclude = ["Z760:cid10", "Z000:cid10"]
        df = load_and_prepare_dataframe(
            file_path=str(path),
            terminology_labels=extra_labels,
            exclude_codes=exclude,
        )
        labels = df["terminology_label"].unique().to_list()
        assert "EXTRA:ab" in labels


# ---------------------------------------------------------------------------
# extract_dataset_name
# ---------------------------------------------------------------------------


class TestExtractDatasetName:
    @pytest.mark.parametrize(
        "file_path,expected",
        [
            (
                "/data/interim/cleaned_aesop_20municipios_salvador.parquet",
                "salvador",
            ),
            (
                "/data/interim/cleaned_20municipios_sao_paulo.parquet",
                "sao_paulo",
            ),
            (
                "/data/interim/cleaned_20municipios_porto_alegre.parquet",
                "porto_alegre",
            ),
        ],
    )
    def test_extracts_name_after_municipios(self, file_path, expected):
        assert extract_dataset_name(file_path) == expected

    def test_returns_unknown_when_no_municipios(self):
        assert extract_dataset_name("/data/raw/dados.parquet") == "unknown"

    def test_returns_unknown_when_no_parquet(self):
        assert extract_dataset_name("/data/interim/cleaned_20municipios_salvador") == "unknown"
