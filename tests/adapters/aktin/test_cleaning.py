import polars as pl
import pytest

from vir.adapters.aktin.cleaning import clean_aktin_raw


@pytest.fixture
def cases() -> pl.DataFrame:
    """Five encounters covering the cases the transformation has to handle."""
    return pl.DataFrame(
        {
            "geschlecht": ["M", "F", "M", "F", ""],
            "aufnahme_ts": [
                "2023-06-15T10:00:00Z",
                "2023-12-31T23:30:00Z",
                "2023-06-15T10:00:00Z",
                "",
                "2023-06-15T10:00:00Z",
            ],
            "entlassung_ts": ["2023-06-15T12:00:00Z"] * 5,
            "age_group": ["45-49", "10-14", "80+", "30-34", "45-49"],
            "plz_kurz": ["123", "456", "789", "111", "222"],
            "triage": ["3", "2", "4", "5", "3"],
            "saettigung": ["98.0", "96.5", "94.0", "99.0", "97.5"],
            "verbleib": ["home"] * 5,
            "a_encounter_num": [1, 2, 3, 4, 5],
        }
    )


@pytest.fixture
def diagnoses() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "icd_code": ["M22.0", "S93.4", "I10", "K86", "R10.3", "M54.2", "A00.0"],
            "diagnose_fuehrend": ["", "", "", "", "", "", ""],
            "diagnose_zusatz": ["G", "G", "G", "V", "", "G", "G"],
            "a_encounter_num": [1, 1, 2, 2, 3, 3, 4],
        }
    )


def test_inner_joins_one_row_per_confirmed_diagnosis(cases, diagnoses):
    """Encounter 1 has 2 confirmed diagnoses → 2 rows.
    Encounter 2 has 1 G + 1 V → 1 row (V is dropped).
    Encounter 3 has 1 G + 1 blank → 1 row.
    Encounter 4 has 1 G but case row has empty aufnahme_ts → dropped.
    Encounter 5 case row has empty geschlecht → dropped, and no diagnosis anyway.
    Total: 2 + 1 + 1 = 4 rows.
    """
    result = clean_aktin_raw(cases, diagnoses)

    assert result.height == 4
    assert sorted(result["a_encounter_num"].to_list()) == [1, 1, 2, 3]


def test_drops_diagnoses_that_are_not_confirmed(cases, diagnoses):
    result = clean_aktin_raw(cases, diagnoses)

    codes = result["code"].to_list()
    assert "K86" not in codes
    assert "R10.3" not in codes


def test_drops_cases_missing_required_fields(cases, diagnoses):
    result = clean_aktin_raw(cases, diagnoses)

    encounter_nums = result["a_encounter_num"].to_list()
    assert 4 not in encounter_nums
    assert 5 not in encounter_nums


def test_iso_year_and_week_are_derived_from_admission_timestamp(cases, diagnoses):
    result = clean_aktin_raw(cases, diagnoses)

    encounter_1 = result.filter(pl.col("a_encounter_num").eq(1)).row(0, named=True)
    assert encounter_1["year"] == 2023
    assert encounter_1["week"] == 24


def test_iso_week_uses_utc_without_local_time_conversion(cases, diagnoses):
    """``2023-12-31T23:30:00Z`` is Sunday in UTC → ISO week 52 of 2023.
    If the adapter were converting to ``Europe/Berlin`` (UTC+1), it would
    land in week 1 of 2024 instead; this test pins the chosen behavior.
    """
    result = clean_aktin_raw(cases, diagnoses)

    encounter_2 = result.filter(pl.col("a_encounter_num").eq(2)).row(0, named=True)
    assert encounter_2["year"] == 2023
    assert encounter_2["week"] == 52


def test_native_age_bins_collapse_to_eight_aktin_buckets(cases, diagnoses):
    result = clean_aktin_raw(cases, diagnoses)

    by_encounter = {
        row["a_encounter_num"]: row["age_group"] for row in result.iter_rows(named=True)
    }
    assert by_encounter[1] == "40-59"
    assert by_encounter[2] == "10-14"
    assert by_encounter[3] == "80+"


def test_geschlecht_is_renamed_to_sex_with_values_preserved(cases, diagnoses):
    result = clean_aktin_raw(cases, diagnoses)

    assert "geschlecht" not in result.columns
    assert "sex" in result.columns
    sex_by_encounter = {row["a_encounter_num"]: row["sex"] for row in result.iter_rows(named=True)}
    assert sex_by_encounter[1] == "M"
    assert sex_by_encounter[2] == "F"


def test_icd_code_is_exposed_as_code(cases, diagnoses):
    result = clean_aktin_raw(cases, diagnoses)

    assert "icd_code" not in result.columns
    assert "code" in result.columns
    assert set(result["code"].to_list()) == {"M22.0", "S93.4", "I10", "M54.2"}


def test_canonical_static_fields_are_set(cases, diagnoses):
    result = clean_aktin_raw(cases, diagnoses)

    assert result["city"].unique().to_list() == ["germany"]
    assert result["code_type"].unique().to_list() == ["icd10gm"]
    assert result["quantity"].unique().to_list() == [1]


def test_plz_kurz_is_dropped(cases, diagnoses):
    result = clean_aktin_raw(cases, diagnoses)

    assert "plz_kurz" not in result.columns


def test_case_data_extras_are_retained_for_later_statistics(cases, diagnoses):
    """Triage / vitals / verbleib survive cleaning — they're dropped only at
    ``prepare_dataframe`` (canonical-row selection)."""
    result = clean_aktin_raw(cases, diagnoses)

    for column in ("triage", "saettigung", "verbleib", "a_encounter_num", "entlassung_ts"):
        assert column in result.columns
