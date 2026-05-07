from pathlib import Path

import polars as pl
import pytest

from vir.helpers import extract_dataset_name, load_all_cities, slugify, suffix_from


class TestSuffixFrom:
    @pytest.mark.parametrize(
        "filename,break_on,expected_suffix",
        [
            (
                "data/interim/cleaned_aesop_raw_dados_2017_20250223_20municipios_082025_salvador.parquet",
                "20municipios",
                "_082025_salvador.parquet",
            ),
            (
                "data/processed/processed_optimized__082025_ibitiara.parquet",
                "__",
                "082025_ibitiara.parquet",
            ),
            (
                "data/raw/dados.parquet",
                "i-do-not-exist-there",
                "dados.parquet",
            ),
            (
                "data/raw/dados.parquet",
                "",
                "dados.parquet",
            ),
        ],
    )
    def test_suffix_from(self, filename, break_on, expected_suffix):
        assert suffix_from(Path(filename), break_on=break_on) == expected_suffix

    def test_suffix_from_with_remove_extension(self):
        filepath = Path(
            "data/interim/cleaned_aesop_raw_dados_2017_20250223_20municipios_082025_salvador.parquet"
        )
        expected_suffix = "_082025_salvador"
        assert suffix_from(filepath, remove_extension=True) == expected_suffix


class TestExtractDatasetName:
    @pytest.mark.parametrize(
        "file_path,expected",
        [
            (
                "/data/processed/prepared_base_aesop_raw_extracao_01042026_all_cities.parquet",
                "base_aesop_raw_extracao_01042026_all_cities",
            ),
            (
                "/data/processed/prepared_base_aesop_raw_extracao_01042026_salvador.parquet",
                "base_aesop_raw_extracao_01042026_salvador",
            ),
            (
                "/data/interim/cleaned_aesop_raw_dados_2017_20250223_20municipios_salvador.parquet",
                "aesop_raw_dados_2017_20250223_20municipios_salvador",
            ),
            (
                "/data/interim/cleaned_base_aesop_raw_extracao_01042026_porto_alegre.parquet",
                "base_aesop_raw_extracao_01042026_porto_alegre",
            ),
        ],
    )
    def test_strips_known_prefix(self, file_path, expected):
        assert extract_dataset_name(file_path) == expected

    def test_returns_stem_when_no_known_prefix(self):
        assert extract_dataset_name("/data/raw/dados.parquet") == "dados"

    def test_works_on_plain_filename(self):
        assert extract_dataset_name("prepared_my_dataset.parquet") == "my_dataset"


class TestSlugify:
    @pytest.mark.parametrize(
        "input_text,expected_slug",
        [
            ("Salvador", "salvador"),
            ("Anápolis", "anapolis"),
            ("Rio Branco", "rio_branco"),
            ("Distrito Federal", "distrito_federal"),
            ("salvador", "salvador"),
        ],
    )
    def test_slugify(self, input_text, expected_slug):
        assert slugify(input_text) == expected_slug


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

        monkeypatch.setattr("vir.helpers.RAW_DATA_FILE", raw_data_file)
        monkeypatch.setattr("vir.helpers.INTERIM_DATA_DIR", interim_dir)
        monkeypatch.setattr("vir.helpers.PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr("vir.helpers.CITY_NAMES", ["Salvador", "Rio Branco"])

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
        monkeypatch.setattr("vir.helpers.CITY_NAMES", [city])
        self._write_parquet(
            interim_dir,
            f"cleaned_fake_dataset_{expected_slug}.parquet",
            pl.DataFrame({"city": [city], "value": [1]}),
        )

        result = load_all_cities(from_="cleaned")

        assert result["city"].to_list() == [city]
