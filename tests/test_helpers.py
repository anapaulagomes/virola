from pathlib import Path

import pytest

from vir.helpers import suffix_from


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
