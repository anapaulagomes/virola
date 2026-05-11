from vir.adapters.datasus.features import (
    AGE_GROUP,
    CITY_NAMES,
    CODE_TYPES_MAPPING,
    IBGE_CITY_MAP,
    ORIGINAL_AGE_GROUP,
    ORIGINAL_TO_REDUCED_AGE_GROUP,
    POPULATION_RANGE,
)


def test_ibge_city_map_is_keyed_by_integer_codes():
    assert all(isinstance(ibge_code, int) for ibge_code in IBGE_CITY_MAP)
    assert "Salvador" in {entry["city"] for entry in IBGE_CITY_MAP.values()}


def test_city_names_match_ibge_city_map():
    assert set(CITY_NAMES) == {entry["city"] for entry in IBGE_CITY_MAP.values()}


def test_age_group_reduction_covers_every_original_bucket():
    assert set(ORIGINAL_TO_REDUCED_AGE_GROUP.keys()) == set(ORIGINAL_AGE_GROUP)
    assert set(ORIGINAL_TO_REDUCED_AGE_GROUP.values()) == set(AGE_GROUP)


def test_code_types_mapping_normalises_to_canonical_keys():
    assert CODE_TYPES_MAPPING["CID"] == "cid10"
    assert CODE_TYPES_MAPPING["CIAP"] == "ciap2"
    assert CODE_TYPES_MAPPING["AB"] == "ab"


def test_population_range_buckets_are_ordered_ascending():
    assert POPULATION_RANGE[0] == "< 100 mil"
    assert POPULATION_RANGE[-1] == "> 500 mil"
