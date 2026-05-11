from pathlib import Path

import polars as pl
import pytest

from vir.adapters.datasus.helpers import load_all_cities


class TestLoadAllCities:
    @pytest.fixture
    def fake_data_layout(self, tmp_path, monkeypatch):
        """Point helpers at tmp directories with a controlled raw-stem and city list."""
        interim_dir = tmp_path / "interim"
        processed_dir = tmp_path / "processed"
        raw_dir = tmp_path / "raw"
        for directory in (interim_dir, processed_dir, raw_dir):
            directory.mkdir()
        raw_data_file = raw_dir / "fake_dataset.parquet" / "*.parquet"

        monkeypatch.setattr("vir.adapters.datasus.helpers.RAW_DATA_FILE", raw_data_file)
        monkeypatch.setattr("vir.adapters.datasus.helpers.INTERIM_DATA_DIR", interim_dir)
        monkeypatch.setattr("vir.adapters.datasus.helpers.PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr("vir.adapters.datasus.helpers.CITY_NAMES", ["Salvador", "Rio Branco"])

        return interim_dir, processed_dir

    @staticmethod
    def _write_parquet(directory: Path, filename: str, frame: pl.DataFrame) -> None:
        frame.write_parquet(directory / filename)

    def _seed_cleaned_two_cities(self, interim_dir: Path) -> None:
        self._write_parquet(
            interim_dir,
            "cleaned_fake_dataset_salvador.parquet",
            pl.DataFrame({"city": ["Salvador"], "value": [1]}),
        )
        self._write_parquet(
            interim_dir,
            "cleaned_fake_dataset_rio_branco.parquet",
            pl.DataFrame({"city": ["Rio Branco"], "value": [2]}),
        )

    def _seed_prepared_two_cities(self, processed_dir: Path) -> None:
        self._write_parquet(
            processed_dir,
            "prepared_fake_dataset_salvador.parquet",
            pl.DataFrame({"value": [10]}),
        )
        self._write_parquet(
            processed_dir,
            "prepared_fake_dataset_rio_branco.parquet",
            pl.DataFrame({"value": [20]}),
        )

    def test_concatenates_cleaned_per_city_files_from_interim_dir(self, fake_data_layout):
        interim_dir, _ = fake_data_layout
        self._seed_cleaned_two_cities(interim_dir)

        result = load_all_cities(from_="cleaned")

        assert result.height == 2
        assert sorted(result["city"].to_list()) == ["Rio Branco", "Salvador"]

    def test_concatenates_prepared_per_city_files_from_processed_dir(self, fake_data_layout):
        _, processed_dir = fake_data_layout
        self._seed_prepared_two_cities(processed_dir)

        result = load_all_cities(from_="prepared")

        assert result.height == 2
        assert sorted(result["city"].to_list()) == ["Rio Branco", "Salvador"]

    def test_injects_city_column_when_missing(self, fake_data_layout):
        _, processed_dir = fake_data_layout
        self._seed_prepared_two_cities(processed_dir)

        result = load_all_cities(from_="prepared")

        assert "city" in result.columns
        assert result.filter(pl.col("city") == "Salvador")["value"].to_list() == [10]
        assert result.filter(pl.col("city") == "Rio Branco")["value"].to_list() == [20]

    def test_does_not_override_existing_city_column(self, fake_data_layout):
        interim_dir, _ = fake_data_layout
        self._seed_cleaned_two_cities(interim_dir)

        result = load_all_cities(from_="cleaned")

        assert result.filter(pl.col("city") == "Salvador")["value"].to_list() == [1]
        assert result.filter(pl.col("city") == "Rio Branco")["value"].to_list() == [2]

    def test_unknown_source_raises_value_error(self):
        with pytest.raises(ValueError, match="cleaned.*prepared"):
            load_all_cities(from_="raw")

    def test_raises_when_a_per_city_file_is_missing(self, fake_data_layout):
        interim_dir, _ = fake_data_layout
        self._write_parquet(
            interim_dir,
            "cleaned_fake_dataset_salvador.parquet",
            pl.DataFrame({"city": ["Salvador"], "value": [1]}),
        )

        with pytest.raises(FileNotFoundError):
            load_all_cities(from_="cleaned")

    @pytest.mark.parametrize(
        "city,expected_slug",
        [
            ("Salvador", "salvador"),
            ("Rio Branco", "rio_branco"),
            ("Distrito Federal", "distrito_federal"),
            ("Anápolis", "anápolis"),
            ("Gravataí", "gravataí"),
        ],
    )
    def test_filename_pattern_lowercases_replaces_spaces_and_preserves_accents(
        self, fake_data_layout, monkeypatch, city, expected_slug
    ):
        interim_dir, _ = fake_data_layout
        monkeypatch.setattr("vir.adapters.datasus.helpers.CITY_NAMES", [city])
        self._write_parquet(
            interim_dir,
            f"cleaned_fake_dataset_{expected_slug}.parquet",
            pl.DataFrame({"city": [city], "value": [1]}),
        )

        result = load_all_cities(from_="cleaned")

        assert result["city"].to_list() == [city]
