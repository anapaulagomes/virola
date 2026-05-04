import polars as pl
import pytest

from vir.data_filters import (
    filter_min_temporal_coverage,
    filter_skip_patterns,
    filter_temporal_quantiles,
)


@pytest.fixture
def sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "terminology_id": [1, 2, 3, 4, 5, 6],
            "code": ["Z000", "Z760", "I10", "A77", "ABP005", "Z96"],
            "code_type": ["cid10", "cid10", "cid10", "ciap2", "ab", "ciap2"],
            "week": [1, 2, 3, 4, 5, 6],
            "quantity": [10, 20, 30, 40, 50, 60],
        }
    )


class TestFilterSkipPatterns:
    def test_empty_patterns_returns_df_unchanged(self, sample_df):
        result = filter_skip_patterns(sample_df, [])
        assert result.shape == sample_df.shape

    def test_skip_z_chapter_cid10_removes_matching_rows(self, sample_df):
        result = filter_skip_patterns(sample_df, [("cid10", r"^Z")])
        codes = result["code"].to_list()
        assert "Z000" not in codes
        assert "Z760" not in codes

    def test_skip_z_chapter_cid10_keeps_non_z_cid10(self, sample_df):
        result = filter_skip_patterns(sample_df, [("cid10", r"^Z")])
        assert "I10" in result["code"].to_list()

    def test_pattern_is_scoped_to_code_type(self, sample_df):
        """Z96 is ciap2 — a cid10-only Z pattern must not remove it."""
        result = filter_skip_patterns(sample_df, [("cid10", r"^Z")])
        assert "Z96" in result["code"].to_list()

    def test_none_code_type_applies_to_all_types(self, sample_df):
        """None code_type means the pattern applies regardless of code_type."""
        result = filter_skip_patterns(sample_df, [(None, r"^Z")])
        codes = result["code"].to_list()
        assert "Z000" not in codes
        assert "Z760" not in codes
        assert "Z96" not in codes

    def test_none_code_type_keeps_non_matching_codes(self, sample_df):
        result = filter_skip_patterns(sample_df, [(None, r"^Z")])
        codes = result["code"].to_list()
        assert "I10" in codes
        assert "A77" in codes
        assert "ABP005" in codes

    def test_multiple_patterns_are_all_applied(self, sample_df):
        result = filter_skip_patterns(sample_df, [("cid10", r"^Z"), ("ab", r"^ABP")])
        codes = result["code"].to_list()
        assert "Z000" not in codes
        assert "Z760" not in codes
        assert "ABP005" not in codes

    def test_multiple_patterns_keep_unmatched_codes(self, sample_df):
        result = filter_skip_patterns(sample_df, [("cid10", r"^Z"), ("ab", r"^ABP")])
        codes = result["code"].to_list()
        assert "I10" in codes
        assert "A77" in codes
        assert "Z96" in codes

    def test_pattern_matching_nothing_keeps_all_rows(self, sample_df):
        result = filter_skip_patterns(sample_df, [("cid10", r"^X")])
        assert result.shape == sample_df.shape

    def test_pattern_matching_entire_type_removes_all_of_that_type(self, sample_df):
        result = filter_skip_patterns(sample_df, [("cid10", r".*")])
        assert result.filter(pl.col("code_type") == "cid10").is_empty()

    def test_pattern_matching_entire_type_keeps_other_types(self, sample_df):
        result = filter_skip_patterns(sample_df, [("cid10", r".*")])
        assert not result.filter(pl.col("code_type") == "ciap2").is_empty()
        assert not result.filter(pl.col("code_type") == "ab").is_empty()

    def test_pattern_is_anchored_correctly(self, sample_df):
        """Pattern ^I should only remove codes starting with I, not containing I."""
        result = filter_skip_patterns(sample_df, [("cid10", r"^I")])
        assert "I10" not in result["code"].to_list()
        assert "Z000" in result["code"].to_list()

    def test_unknown_code_type_in_pattern_removes_nothing(self, sample_df):
        """A code_type not present in the data should be a no-op."""
        result = filter_skip_patterns(sample_df, [("unknown_type", r".*")])
        assert result.shape == sample_df.shape

    def test_row_count_decreases_by_expected_amount(self, sample_df):
        result = filter_skip_patterns(sample_df, [("cid10", r"^Z")])
        assert len(result) == len(sample_df) - 2  # Z000 and Z760 removed

    def test_other_columns_are_preserved(self, sample_df):
        result = filter_skip_patterns(sample_df, [("cid10", r"^Z")])
        assert set(result.columns) == set(sample_df.columns)


def _build_temporal_rows(
    terminology_id: int, weeks: list[tuple[int, int]], quantities: list[int]
) -> dict:
    """Expand a code's per-week observations into a row-dict for a DataFrame."""
    return {
        "terminology_id": [terminology_id] * len(weeks),
        "year": [year for year, _ in weeks],
        "week": [week for _, week in weeks],
        "quantity": quantities,
    }


@pytest.fixture
def temporal_df() -> pl.DataFrame:
    """Five codes with carefully chosen profiles so that P20 is deterministic.

    nonzero_weeks distribution: [1, 2, 5, 8, 10]   -> P20 = 2
    weekly_mean   distribution: [1, 2, 5, 10, 20]  -> P20 = 2
    Only code A (1 week, mean 1) fails both thresholds; B/C/D/E pass.
    """
    rows: list[dict] = []
    rows.append(
        {
            "terminology_id": [1],
            "year": [2024],
            "week": [1],
            "quantity": [1],
        }
    )
    rows.append(
        {
            "terminology_id": [2, 2],
            "year": [2024, 2024],
            "week": [1, 2],
            "quantity": [2, 2],
        }
    )
    rows.append(
        {
            "terminology_id": [3] * 5,
            "year": [2024] * 5,
            "week": [1, 2, 3, 4, 5],
            "quantity": [5, 5, 5, 5, 5],
        }
    )
    rows.append(
        {
            "terminology_id": [4] * 8,
            "year": [2024] * 8,
            "week": list(range(1, 9)),
            "quantity": [10] * 8,
        }
    )
    rows.append(
        {
            "terminology_id": [5] * 10,
            "year": [2024] * 10,
            "week": list(range(1, 11)),
            "quantity": [20] * 10,
        }
    )
    combined = {key: [] for key in rows[0]}
    for chunk in rows:
        for key, values in chunk.items():
            combined[key].extend(values)
    return pl.DataFrame(combined)


class TestFilterTemporalQuantiles:
    def test_drops_codes_below_p20_on_either_metric(self, temporal_df):
        result = filter_temporal_quantiles(temporal_df, quantile=0.20)
        kept_ids = sorted(result["terminology_id"].unique().to_list())
        assert kept_ids == [2, 3, 4, 5]

    def test_dropped_code_rows_are_fully_removed(self, temporal_df):
        result = filter_temporal_quantiles(temporal_df, quantile=0.20)
        assert result.filter(pl.col("terminology_id") == 1).is_empty()

    def test_kept_code_rows_are_intact(self, temporal_df):
        result = filter_temporal_quantiles(temporal_df, quantile=0.20)
        for terminology_id in (2, 3, 4, 5):
            original_rows = temporal_df.filter(pl.col("terminology_id") == terminology_id)
            kept_rows = result.filter(pl.col("terminology_id") == terminology_id)
            assert kept_rows.shape == original_rows.shape

    def test_columns_are_preserved(self, temporal_df):
        result = filter_temporal_quantiles(temporal_df, quantile=0.20)
        assert set(result.columns) == set(temporal_df.columns)

    def test_quantile_zero_keeps_every_code(self, temporal_df):
        result = filter_temporal_quantiles(temporal_df, quantile=0.0)
        assert sorted(result["terminology_id"].unique().to_list()) == [1, 2, 3, 4, 5]

    def test_higher_quantile_drops_more_codes(self, temporal_df):
        """At P50 the median is 5 weeks / weekly_mean 5, so codes 1 and 2 fall out."""
        result = filter_temporal_quantiles(temporal_df, quantile=0.50)
        kept_ids = sorted(result["terminology_id"].unique().to_list())
        assert kept_ids == [3, 4, 5]

    def test_both_thresholds_must_pass(self):
        """A code with many weeks but tiny weekly mean must still be dropped."""
        df = pl.DataFrame(
            {
                "terminology_id": [1] * 10 + [2] * 2 + [3] * 2 + [4] * 2 + [5] * 2,
                "year": [2024] * 18,
                "week": list(range(1, 11)) + [1, 2] * 4,
                "quantity": [1] * 10 + [50, 50, 50, 50, 50, 50, 50, 50],
            }
        )
        result = filter_temporal_quantiles(df, quantile=0.20)
        assert result.filter(pl.col("terminology_id") == 1).is_empty()

    def test_distinct_year_week_pairs_are_counted(self):
        """Multiple rows in the same (year, week) must count as one nonzero_week.

        Code 1 has five rows but they all sit in the same week — its
        ``nonzero_weeks`` is 1, below the P20 threshold of 2 set by codes 2-5.
        If the implementation counted rows instead of distinct weeks, code 1
        would incorrectly survive.
        """
        df = pl.DataFrame(
            {
                "terminology_id": [1] * 5 + [2, 2, 3, 3, 4, 4, 5, 5],
                "year": [2024] * 13,
                "week": [1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2],
                "quantity": [100] * 5 + [1] * 8,
            }
        )
        result = filter_temporal_quantiles(df, quantile=0.20)
        assert result.filter(pl.col("terminology_id") == 1).is_empty()
        assert sorted(result["terminology_id"].unique().to_list()) == [2, 3, 4, 5]


class TestFilterMinTemporalCoverage:
    @pytest.fixture
    def multi_year_df(self) -> pl.DataFrame:
        """Two years where code coverage differs across years.

        Each (terminology_id, year, week) is split into three demographic rows
        (sex × age_group), mirroring the prepared parquet aggregation level.
        This lets the tests catch implementations that count rows instead of
        distinct weeks.

        Distinct-week profile (per (terminology_id, year)):

        Year 2024:
          code 1: 2 weeks  (below min_weeks=3)
          code 2: 5 weeks
          code 3: 4 weeks

        Year 2025:
          code 1: 4 weeks
          code 2: 1 week   (below min_weeks=3)
          code 3: 3 weeks

        With min_weeks=3 (default by_year=True):
          2024 keeps codes 2, 3
          2025 keeps codes 1, 3
        """
        profiles = [
            (1, 2024, [1, 2]),
            (2, 2024, [1, 2, 3, 4, 5]),
            (3, 2024, [1, 2, 3, 4]),
            (1, 2025, [1, 2, 3, 4]),
            (2, 2025, [1]),
            (3, 2025, [1, 2, 3]),
        ]
        demographics = [("M", "0-19"), ("F", "0-19"), ("M", "20-59")]
        records = [
            {
                "terminology_id": terminology_id,
                "year": year,
                "week": week,
                "sex": sex,
                "age_group": age_group,
                "quantity": 10,
            }
            for terminology_id, year, weeks in profiles
            for week in weeks
            for sex, age_group in demographics
        ]
        return pl.DataFrame(records)

    def test_drops_code_below_threshold_in_its_year(self, multi_year_df):
        result = filter_min_temporal_coverage(multi_year_df, min_weeks=3)
        year_2024 = result.filter(pl.col("year") == 2024)
        assert 1 not in year_2024["terminology_id"].unique().to_list()

    def test_keeps_codes_above_threshold(self, multi_year_df):
        result = filter_min_temporal_coverage(multi_year_df, min_weeks=3)
        year_2024 = result.filter(pl.col("year") == 2024)
        assert sorted(year_2024["terminology_id"].unique().to_list()) == [2, 3]

    def test_second_year_evaluated_independently(self, multi_year_df):
        result = filter_min_temporal_coverage(multi_year_df, min_weeks=3)
        year_2025 = result.filter(pl.col("year") == 2025)
        assert sorted(year_2025["terminology_id"].unique().to_list()) == [1, 3]

    def test_preserves_all_columns(self, multi_year_df):
        result = filter_min_temporal_coverage(multi_year_df, min_weeks=3)
        assert set(result.columns) == set(multi_year_df.columns)

    def test_by_year_false_evaluates_across_full_period(self, multi_year_df):
        """With by_year=False, codes are evaluated once across the entire period.

        Distinct-week totals: code 1 has 6, code 2 has 6, code 3 has 7 — all
        ≥ 3, so every row is kept. By contrast the default per-year mode drops
        code 1's 2024 rows and code 2's 2025 row.
        """
        result = filter_min_temporal_coverage(multi_year_df, min_weeks=3, by_year=False)
        assert sorted(result["terminology_id"].unique().to_list()) == [1, 2, 3]
        assert result.shape == multi_year_df.shape

    def test_by_year_false_counts_distinct_weeks_not_rows(self):
        """A code with one week of demographic-split rows must not satisfy
        min_weeks. Without distinct-week counting the four sex/age_group rows
        for week 1 would be counted as four "weeks"."""
        df = pl.DataFrame(
            {
                "terminology_id": [1] * 4 + [2] * 3,
                "year": [2024] * 7,
                "week": [1, 1, 1, 1, 1, 2, 3],
                "sex": ["M", "F", "M", "F", "M", "M", "M"],
                "age_group": ["0-19", "0-19", "20-59", "20-59", "0-19", "0-19", "0-19"],
                "quantity": [10] * 7,
            }
        )
        result = filter_min_temporal_coverage(df, min_weeks=3, by_year=False)
        assert result.filter(pl.col("terminology_id") == 1).is_empty()
        assert sorted(result["terminology_id"].unique().to_list()) == [2]

    def test_min_weeks_zero_keeps_all_rows(self, multi_year_df):
        """min_weeks=0 means no temporal cutoff — every code should survive.
        A falsy check on min_weeks would silently fall back to MIN_WEEKS."""
        result = filter_min_temporal_coverage(multi_year_df, min_weeks=0)
        assert result.shape == multi_year_df.shape
        assert sorted(result["terminology_id"].unique().to_list()) == [1, 2, 3]
