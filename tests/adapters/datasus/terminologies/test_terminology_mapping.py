import polars as pl

from vir.adapters.datasus.terminologies.terminology_mapping import load_mapping


def test_load_mapping_returns_long_format_with_expected_columns():
    mapping = load_mapping()

    assert isinstance(mapping, pl.DataFrame)
    assert set(mapping.columns) == {"ab_code", "ab_programme_code", "ciap2_code", "cid10_code"}
    assert mapping.height > 0
