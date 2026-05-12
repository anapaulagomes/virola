import polars as pl
import pytest

from vir.cleaning.indicative_template import build_indicative_template


@pytest.fixture
def cleaned() -> pl.DataFrame:
    """Six codes whose quantities form a long-tail distribution.

    Cumulative volumes (descending): 50 + 25 + 15 + 5 + 3 + 2 = 100.
    cum_pct: 0.50, 0.75, 0.90, 0.95, 0.98, 1.00.

    At ``coverage=0.90`` ``filter_most_representative_rows`` keeps rows whose
    *previous* cumulative percentage is < 0.90 — that means the first three
    codes (cum_pct 0.50, 0.75, 0.90) make the cut.
    """
    return pl.DataFrame(
        {
            "code": ["A1", "B2", "C3", "D4", "E5", "F6"],
            "code_type": ["cid10"] * 6,
            "quantity": [50, 25, 15, 5, 3, 2],
        }
    )


@pytest.fixture
def terminology_labels() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "code": ["A1", "B2", "C3", "D4", "E5", "F6"],
            "code_type": ["cid10"] * 6,
            "description": [
                "high-volume code A1",
                "high-volume code B2",
                "mid-volume code C3",
                "low-volume code D4",
                "low-volume code E5",
                "low-volume code F6",
            ],
        }
    )


@pytest.fixture
def existing_labels() -> pl.DataFrame:
    """Two of the top-three codes are already labeled."""
    return pl.DataFrame(
        {
            "code": ["A1", "C3"],
            "code_type": ["cid10", "cid10"],
            "indicative": ["yes", "no"],
        }
    )


def test_output_schema_matches_labeling_csv_convention(cleaned, terminology_labels):
    result = build_indicative_template(cleaned, terminology_labels, coverage=0.9)

    assert set(result.columns) >= {"code", "code_type", "description", "indicative"}


def test_top_codes_cover_at_least_the_requested_fraction(cleaned, terminology_labels):
    """Top-3 codes sum to 90% of total volume; the function must return all three."""
    result = build_indicative_template(cleaned, terminology_labels, coverage=0.9)

    kept = set(result["code"].to_list())
    assert kept == {"A1", "B2", "C3"}


def test_tail_codes_are_excluded(cleaned, terminology_labels):
    result = build_indicative_template(cleaned, terminology_labels, coverage=0.9)

    excluded = set(cleaned["code"].to_list()) - set(result["code"].to_list())
    assert excluded == {"D4", "E5", "F6"}


def test_lower_coverage_returns_fewer_codes(cleaned, terminology_labels):
    """At coverage=0.6, only A1 (50%) covers under the threshold; the cumulative
    crossing row B2 (cum_pct 0.75) is also included by the filter, so output
    is {A1, B2}."""
    result = build_indicative_template(cleaned, terminology_labels, coverage=0.6)

    assert set(result["code"].to_list()) == {"A1", "B2"}


def test_descriptions_are_joined_from_terminology_labels(cleaned, terminology_labels):
    result = build_indicative_template(cleaned, terminology_labels, coverage=0.9)

    by_code = {row["code"]: row["description"] for row in result.iter_rows(named=True)}
    assert by_code["A1"] == "high-volume code A1"
    assert by_code["C3"] == "mid-volume code C3"


def test_existing_indicative_labels_are_preserved(cleaned, terminology_labels, existing_labels):
    result = build_indicative_template(
        cleaned, terminology_labels, existing_labels=existing_labels, coverage=0.9
    )

    by_code = {row["code"]: row["indicative"] for row in result.iter_rows(named=True)}
    assert by_code["A1"] == "yes"
    assert by_code["C3"] == "no"
    assert by_code["B2"] is None


def test_indicative_is_null_for_every_row_when_no_existing_labels(cleaned, terminology_labels):
    result = build_indicative_template(cleaned, terminology_labels, coverage=0.9)

    assert result["indicative"].null_count() == result.height


def test_rows_are_sorted_by_volume_descending(cleaned, terminology_labels):
    result = build_indicative_template(cleaned, terminology_labels, coverage=0.9)

    assert result["code"].to_list() == ["A1", "B2", "C3"]
