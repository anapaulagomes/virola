import polars as pl

from vir.adapters.aktin.terminology import build_icd10gm_labels


def test_labels_have_canonical_schema():
    labels = build_icd10gm_labels()

    assert set(labels.columns) == {"code", "code_type", "description"}


def test_code_type_is_always_icd10gm():
    labels = build_icd10gm_labels()

    assert labels["code_type"].unique().to_list() == ["icd10gm"]


def test_includes_a_known_four_character_diagnosis_with_its_german_label():
    """A00.0 (Cholera durch Vibrio cholerae O:1, Biovar cholerae) is a stable
    BfArM endpoint code; if the parser is wired correctly it must appear with
    the long German title."""
    labels = build_icd10gm_labels()

    row = labels.filter(pl.col("code") == "A00.0").row(0, named=True)
    assert row["description"] == "Cholera durch Vibrio cholerae O:1, Biovar cholerae"


def test_includes_codes_seen_in_aktin_data():
    """Spot-check a handful of codes observed in the AKTIN export sample
    (case_data / diag_data fixtures from the syndromes export)."""
    labels = build_icd10gm_labels()

    present = set(labels["code"].to_list())
    expected = {"M22.0", "S93.4", "S01.9", "R06.4", "M54.2"}
    assert expected.issubset(present)


def test_does_not_include_bfarm_placeholder_syntax():
    """BfArM marks non-leaf nodes with a trailing ``-`` (e.g. ``A00.-``,
    ``S93.4-``) in column 6. Column 7 — what we expose as ``code`` — never
    carries the placeholder, so it must never appear in the labels output."""
    labels = build_icd10gm_labels()

    placeholder_rows = labels.filter(pl.col("code").str.ends_with("-"))
    assert placeholder_rows.height == 0


def test_includes_three_character_root_codes_with_their_block_title():
    """AKTIN coders occasionally record only the 3-char root (``A00`` for
    cholera). The labels table must still resolve that to a description."""
    labels = build_icd10gm_labels()

    row = labels.filter(pl.col("code") == "A00").row(0, named=True)
    assert row["description"] == "Cholera"


def test_has_enough_codes_to_cover_the_classification():
    """ICD-10-GM 2025 has on the order of 10 000+ codable terminal codes."""
    labels = build_icd10gm_labels()

    assert labels.height > 10_000


def test_no_null_or_empty_codes_or_descriptions():
    labels = build_icd10gm_labels()

    assert labels.filter(pl.col("code").is_null() | (pl.col("code") == "")).height == 0
    assert (
        labels.filter(pl.col("description").is_null() | (pl.col("description") == "")).height == 0
    )
