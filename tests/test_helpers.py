import unicodedata
from pathlib import Path

import pytest

from vir.helpers import extract_dataset_name, find_dataset_file, slugify, suffix_from


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
            ("cities/salvador", "cities_salvador"),
        ],
    )
    def test_slugify(self, input_text, expected_slug):
        assert slugify(input_text) == expected_slug


class TestFindDatasetFile:
    def test_matches_plain_ascii_name(self):
        candidates = [
            "/data/processed/prepared_base_salvador.parquet",
            "/data/processed/prepared_base_manaus.parquet",
        ]
        assert (
            find_dataset_file("base_manaus", candidates)
            == "/data/processed/prepared_base_manaus.parquet"
        )

    def test_matches_nfc_name_against_nfd_filename(self):
        nfc_name = unicodedata.normalize("NFC", "base_anápolis")
        nfd_path = unicodedata.normalize("NFD", "/data/processed/prepared_base_anápolis.parquet")
        assert nfc_name != nfd_path[: len(nfc_name)]
        assert find_dataset_file(nfc_name, [nfd_path]) == nfd_path

    def test_returns_original_path_unchanged(self):
        nfd_path = unicodedata.normalize("NFD", "/data/processed/prepared_base_gravataí.parquet")
        result = find_dataset_file(unicodedata.normalize("NFC", "base_gravataí"), [nfd_path])
        assert result is nfd_path

    def test_returns_none_when_no_match(self):
        assert find_dataset_file("base_curitiba", ["/data/prepared_base_salvador.parquet"]) is None
