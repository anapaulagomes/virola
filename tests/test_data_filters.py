import polars as pl
import pytest

from vir.data_filters import filter_skip_patterns


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
