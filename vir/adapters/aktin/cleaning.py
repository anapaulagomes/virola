"""AKTIN raw → canonical interim transformation.

Takes the per-encounter ``case_data`` and per-diagnosis ``diag_data`` tables
exported from AKTIN and produces one row per (encounter × confirmed diagnosis)
in the canonical interim shape that ``prepare_dataframe`` consumes.
"""

import polars as pl

AKTIN_AGE_BUCKETS: dict[str, str] = {
    "0-4": "0-4",
    "00-04": "0-4",
    "5-9": "5-9",
    "05-09": "5-9",
    "10-14": "10-14",
    "15-19": "15-19",
    "20-24": "20-39",
    "25-29": "20-39",
    "30-34": "20-39",
    "35-39": "20-39",
    "40-44": "40-59",
    "45-49": "40-59",
    "50-54": "40-59",
    "55-59": "40-59",
    "60-64": "60-79",
    "65-69": "60-79",
    "70-74": "60-79",
    "75-79": "60-79",
    "80+": "80+",
}

_REQUIRED_CASE_FIELDS: tuple[str, ...] = ("geschlecht", "aufnahme_ts", "age_group")


def _null_when_empty(column: str) -> pl.Expr:
    return pl.when(pl.col(column).eq("")).then(None).otherwise(pl.col(column)).alias(column)


def clean_aktin_raw(cases: pl.DataFrame, diagnoses: pl.DataFrame) -> pl.DataFrame:
    """Transform AKTIN ``case_data`` + ``diag_data`` into the canonical interim shape.

    Steps:

    1. Drop case rows with empty / null ``geschlecht``, ``aufnahme_ts`` or
       ``age_group``.
    2. Parse ``aufnahme_ts`` accepting both ``Z`` and ``+00:00`` offsets;
       unparseable timestamps drop the row rather than ride through as a null
       year/week.
    3. Collapse the native 5-year ``age_group`` to one of the 8 AKTIN buckets;
       any value outside the bucket map drops the row.
    4. Keep only diagnoses with ``diagnose_zusatz == 'G'`` (confirmed).
    5. Inner-join cases × confirmed diagnoses on ``a_encounter_num``.
    6. Add ``city = "germany"``, ``code_type = "icd10gm"``, ``quantity = 1``.
    7. Drop ``plz_kurz``. Rename ``geschlecht`` → ``sex``. Retain
       ``a_encounter_num`` and the case_data extras (``triage``, vitals,
       ``verbleib``, ...) for later statistics — they are dropped later by
       ``prepare_dataframe`` when projecting onto the canonical row.
    """
    cleaned_cases = (
        cases.with_columns([_null_when_empty(column) for column in _REQUIRED_CASE_FIELDS])
        .drop_nulls(subset=list(_REQUIRED_CASE_FIELDS))
        .with_columns(
            _admission=pl.col("aufnahme_ts")
            .str.replace(r"Z$", "+00:00")
            .str.to_datetime("%Y-%m-%dT%H:%M:%S%z", strict=False),
            age_group=pl.col("age_group").replace_strict(AKTIN_AGE_BUCKETS, default=None),
        )
        .drop_nulls(subset=["_admission", "age_group"])
        .with_columns(
            year=pl.col("_admission").dt.iso_year(),
            week=pl.col("_admission").dt.week(),
        )
        .drop(["_admission", "plz_kurz"])
        .rename({"geschlecht": "sex"})
    )

    confirmed_diagnoses = diagnoses.filter(pl.col("diagnose_zusatz").eq("G")).select(
        code=pl.col("icd_code"),
        a_encounter_num=pl.col("a_encounter_num"),
    )

    joined = cleaned_cases.join(confirmed_diagnoses, on="a_encounter_num", how="inner")
    return joined.with_columns(
        city=pl.lit("germany"),
        code_type=pl.lit("icd10gm"),
        quantity=pl.lit(1),
    )
