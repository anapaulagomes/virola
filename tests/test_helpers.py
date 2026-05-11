from pathlib import Path

import pytest

from vir.helpers import extract_dataset_name, slugify, suffix_from


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
