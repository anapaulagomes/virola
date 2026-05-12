import polars as pl
import pytest

from vir.adapters import ADAPTERS
from vir.adapters.base import Adapter


def _aktin():
    return ADAPTERS["aktin"]


def test_aktin_adapter_is_registered():
    assert "aktin" in ADAPTERS
    assert isinstance(ADAPTERS["aktin"], Adapter)


def test_aktin_adapter_name():
    assert _aktin().name == "aktin"


def test_strata_is_a_single_stratum_named_germany():
    assert _aktin().strata == ["germany"]


def test_code_types_lists_only_icd10gm():
    assert _aktin().code_types == ["icd10gm"]


def test_skip_patterns_drop_icd10gm_z_chapter():
    assert _aktin().skip_patterns == [("icd10gm", r"^Z")]


def test_indicative_codes_file_is_none_until_german_epi_labels_top_codes():
    assert _aktin().indicative_codes_file is None


def test_stratum_metadata_resolves_germany_with_no_region_or_population_range():
    metadata = _aktin().stratum_metadata("germany")
    assert metadata == {"city": "germany", "region": None, "population_range": None}


def test_stratum_metadata_is_case_insensitive_for_germany():
    metadata = _aktin().stratum_metadata("Germany")
    assert metadata["city"] == "germany"


def test_stratum_metadata_returns_none_values_when_stratum_is_unknown():
    metadata = _aktin().stratum_metadata("bavaria")
    assert metadata == {"city": None, "region": None, "population_range": None}


def test_build_terminology_labels_delegates_to_icd10gm_builder():
    labels = _aktin().build_terminology_labels()
    assert isinstance(labels, pl.DataFrame)
    assert labels.height > 10_000
    assert labels["code_type"].unique().to_list() == ["icd10gm"]


def _write_aktin_site(directory, encounter_num: int, code: str) -> None:
    """Drop a minimal AKTIN ``case_data.txt`` + ``diag_data.txt`` pair into a site dir."""
    case_columns = (
        "geschlecht\taufnahme_ts\tentlassung_ts\tage_group\tplz_kurz\t"
        "triage\tsaettigung\tverbleib\ta_encounter_num"
    )
    case_row = (
        f"M\t2023-06-15T10:00:00Z\t2023-06-15T12:00:00Z\t45-49\t123\t"
        f"3\t98.0\thome\t{encounter_num}"
    )
    (directory / "case_data.txt").write_text(case_columns + "\n" + case_row + "\n")

    diag_columns = "icd_code\tdiagnose_fuehrend\tdiagnose_zusatz\ta_encounter_num"
    diag_row = f"{code}\t\tG\t{encounter_num}"
    (directory / "diag_data.txt").write_text(diag_columns + "\n" + diag_row + "\n")


def test_clean_reads_aktin_tsvs_and_writes_canonical_parquet(tmp_path):
    """One site, one encounter, one confirmed diagnosis — verify the full
    read → join → transform → write path."""
    site = tmp_path / "1_result"
    site.mkdir()
    _write_aktin_site(site, encounter_num=42, code="M22.0")

    output_path = tmp_path / "cleaned.parquet"
    returned = _aktin().clean(tmp_path, output_path)

    assert returned == output_path
    cleaned = pl.read_parquet(output_path)
    assert cleaned.height == 1
    assert cleaned["code"].to_list() == ["M22.0"]
    assert cleaned["sex"].to_list() == ["M"]
    assert cleaned["city"].to_list() == ["germany"]
    assert cleaned["code_type"].to_list() == ["icd10gm"]
    assert cleaned["year"].to_list() == [2023]
    assert cleaned["week"].to_list() == [24]
    assert cleaned["age_group"].to_list() == ["40-59"]


def test_clean_pools_multiple_aktin_sites_into_one_interim_parquet(tmp_path):
    for site_id, encounter_num, code in [(1, 100, "M22.0"), (2, 200, "I10")]:
        site = tmp_path / f"{site_id}_result"
        site.mkdir()
        _write_aktin_site(site, encounter_num=encounter_num, code=code)

    output_path = tmp_path / "cleaned.parquet"
    _aktin().clean(tmp_path, output_path)

    cleaned = pl.read_parquet(output_path)
    assert cleaned.height == 2
    assert set(cleaned["code"].to_list()) == {"M22.0", "I10"}


def test_clean_raises_when_no_case_data_is_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="case_data.txt"):
        _aktin().clean(tmp_path, tmp_path / "out.parquet")
