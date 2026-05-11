import polars as pl

from vir.adapters.datasus.cleaning import clean_datasus_raw


def _minimal_raw() -> pl.DataFrame:
    """One AESOP-shaped row covering all required columns."""
    return pl.DataFrame(
        {
            "CIDCIAP": ["CID(I10)"],
            "ANO": [2023],
            "SEMANA": [10],
            "SG_UF": ["BA"],
            "SEXO": ["M"],
            "FX_ETARIA": ["30 a 39 anos"],
            "QT": [5],
            "CO_MUNICIPIO_IBGE": [292740],
            "CO_CNES": ["1234"],
            "Data_folder": ["x"],
        }
    )


def test_renames_aesop_columns_to_canonical_names():
    result = clean_datasus_raw(_minimal_raw())

    expected = {"year", "week", "sex", "age_group", "quantity"}
    assert expected.issubset(set(result.columns))
    assert "ANO" not in result.columns
    assert "SEMANA" not in result.columns
    assert "FX_ETARIA" not in result.columns
    assert "QT" not in result.columns


def test_parses_cidciap_into_code_and_code_type():
    raw = pl.DataFrame(
        {
            "CIDCIAP": ["CID(I10)", "CIAP(K86)", "AB(ABP005)"],
            "ANO": [2023, 2023, 2023],
            "SEMANA": [1, 1, 1],
            "SG_UF": ["BA", "BA", "BA"],
            "SEXO": ["M", "F", "M"],
            "FX_ETARIA": ["30 a 39 anos", "30 a 39 anos", "30 a 39 anos"],
            "QT": [1, 1, 1],
            "CO_MUNICIPIO_IBGE": [292740, 292740, 292740],
            "CO_CNES": ["1", "1", "1"],
            "Data_folder": ["x", "x", "x"],
        }
    )
    result = clean_datasus_raw(raw)

    codes = sorted(result["code"].to_list())
    code_types = sorted(result["code_type"].unique().to_list())
    assert codes == ["ABP005", "I10", "K86"]
    assert code_types == ["ab", "ciap2", "cid10"]


def test_reduces_original_age_groups_to_four_buckets():
    raw = _minimal_raw().with_columns(pl.lit("5 a 11 anos").alias("FX_ETARIA"))
    result = clean_datasus_raw(raw)

    assert result["age_group"].to_list() == ["1 a 15 anos"]


def test_resolves_ibge_municipality_to_city_region_and_population_range():
    result = clean_datasus_raw(_minimal_raw())

    assert result["city"].to_list() == ["Salvador"]
    assert result["region"].to_list() == ["Nordeste"]
    assert result["population_range"].to_list() == ["> 500 mil"]


def test_drops_internal_aesop_columns():
    """CO_CNES and CO_MUNICIPIO_IBGE are administrative — not needed downstream."""
    result = clean_datasus_raw(_minimal_raw())

    for column in ["CIDCIAP", "Data_folder", "CO_CNES", "CO_MUNICIPIO_IBGE"]:
        assert column not in result.columns


def test_drops_rows_with_null_year_week_or_age_group():
    raw = pl.DataFrame(
        {
            "CIDCIAP": ["CID(I10)", "CID(I10)", "CID(I10)", "CID(I10)"],
            "ANO": [2023, None, 2023, 2023],
            "SEMANA": [1, 1, None, 1],
            "SG_UF": ["BA", "BA", "BA", "BA"],
            "SEXO": ["M", "M", "M", "M"],
            "FX_ETARIA": ["30 a 39 anos", "30 a 39 anos", "30 a 39 anos", None],
            "QT": [1, 1, 1, 1],
            "CO_MUNICIPIO_IBGE": [292740, 292740, 292740, 292740],
            "CO_CNES": ["1", "1", "1", "1"],
            "Data_folder": ["x", "x", "x", "x"],
        }
    )
    result = clean_datasus_raw(raw)

    assert result.height == 1
    assert result["year"].to_list() == [2023]


def test_strips_disambiguation_suffix_from_ab_codes():
    raw = pl.DataFrame(
        {
            "CIDCIAP": ["AB(ABP008-1)", "AB(ABP008-2)"],
            "ANO": [2023, 2023],
            "SEMANA": [1, 1],
            "SG_UF": ["BA", "BA"],
            "SEXO": ["M", "F"],
            "FX_ETARIA": ["30 a 39 anos", "30 a 39 anos"],
            "QT": [1, 1],
            "CO_MUNICIPIO_IBGE": [292740, 292740],
            "CO_CNES": ["1", "1"],
            "Data_folder": ["x", "x"],
        }
    )
    result = clean_datasus_raw(raw)

    assert set(result["code"].to_list()) == {"ABP008"}
