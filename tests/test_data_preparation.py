import polars as pl
import pytest

from vir.cleaning.prepare import prepare_dataframe


@pytest.fixture
def terminology_labels() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "code": ["I10", "Z760", "ABP005", "Z000", "K86"],
            "code_type": ["cid10", "cid10", "ab", "cid10", "ciap2"],
            "description": [
                "hypertension",
                "repeat prescription",
                "hypertension program",
                "general exam",
                "hypertension uncomplicated",
            ],
            "label": ["I10:cid10", "Z760:cid10", "ABP005:ab", "Z000:cid10", "K86:ciap2"],
        }
    )


def _build_interim_dataframe(terminology_labels: pl.DataFrame) -> pl.DataFrame:
    """Build a minimal interim dataframe with admin columns and all required fields."""
    rows = []
    for row in terminology_labels.iter_rows(named=True):
        for year in [2022, 2023]:
            for week in range(1, 14):
                rows.append(
                    {
                        "unique_id": "uid",
                        "state": "SP",
                        "city": "Salvador",
                        "region": "NE",
                        "population_range": "large",
                        "year": year,
                        "week": week,
                        "sex": "M",
                        "age_group": "adult",
                        "code_type": row["code_type"],
                        "code": row["code"],
                        "quantity": 10,
                    }
                )
    return pl.DataFrame(rows)


class TestPrepareDataframe:
    def test_drops_administrative_columns(self, terminology_labels):
        interim = _build_interim_dataframe(terminology_labels)
        result = prepare_dataframe(interim, terminology_labels, skip_patterns=[])

        for column in ["unique_id", "state", "city", "region", "population_range"]:
            assert column not in result.columns

    def test_adapter_specific_extra_columns_are_excluded(self, terminology_labels):
        """Extras carried in the interim (e.g. AKTIN's triage/vitals) must not leak through."""
        interim = _build_interim_dataframe(terminology_labels).with_columns(
            pl.lit(3).alias("triage"),
            pl.lit("home").alias("verbleib"),
            pl.lit(98.5).alias("saettigung"),
        )
        result = prepare_dataframe(interim, terminology_labels, skip_patterns=[])

        canonical = {
            "year",
            "week",
            "sex",
            "age_group",
            "code_type",
            "code",
            "quantity",
            "terminology_id",
            "terminology_label",
        }
        assert set(result.columns) == canonical

    def test_missing_canonical_column_raises(self, terminology_labels):
        """Interim missing a canonical column (e.g. quantity) must fail explicitly."""
        interim = _build_interim_dataframe(terminology_labels).drop("quantity")
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            prepare_dataframe(interim, terminology_labels, skip_patterns=[])

    def test_output_has_expected_columns(self, terminology_labels):
        interim = _build_interim_dataframe(terminology_labels)
        result = prepare_dataframe(interim, terminology_labels, skip_patterns=[])

        expected = {
            "year",
            "week",
            "sex",
            "age_group",
            "code_type",
            "code",
            "quantity",
            "terminology_id",
            "terminology_label",
        }
        assert set(result.columns) >= expected

    def test_quantities_are_aggregated(self, terminology_labels):
        """Two rows with the same group key must have their quantities summed."""
        duplicate_rows = pl.DataFrame(
            [
                {
                    "year": 2022,
                    "week": 1,
                    "sex": "M",
                    "age_group": "adult",
                    "code_type": "cid10",
                    "code": "I10",
                    "quantity": 5,
                },
                {
                    "year": 2022,
                    "week": 1,
                    "sex": "M",
                    "age_group": "adult",
                    "code_type": "cid10",
                    "code": "I10",
                    "quantity": 7,
                },
            ]
        )
        result = prepare_dataframe(duplicate_rows, terminology_labels, skip_patterns=[])
        row = result.filter(
            (pl.col("code") == "I10") & (pl.col("year") == 2022) & (pl.col("week") == 1)
        )
        assert row["quantity"].sum() == 12

    def test_legacy_code_types_are_normalized(self, terminology_labels):
        """Interim data with old code_type values (cid, ciap) must be normalised and join correctly."""
        rows = []
        for row in terminology_labels.iter_rows(named=True):
            legacy_type = {"cid10": "cid", "ciap2": "ciap"}.get(row["code_type"], row["code_type"])
            for year in [2022, 2023]:
                for week in range(1, 14):
                    rows.append(
                        {
                            "year": year,
                            "week": week,
                            "sex": "M",
                            "age_group": "adult",
                            "code_type": legacy_type,
                            "code": row["code"],
                            "quantity": 10,
                        }
                    )
        interim = pl.DataFrame(rows)
        result = prepare_dataframe(interim, terminology_labels, skip_patterns=[])

        assert result["terminology_id"].null_count() == 0
        assert result["terminology_label"].n_unique() == len(terminology_labels)

    def test_exclude_codes_removes_non_indicative(self, terminology_labels):
        interim = _build_interim_dataframe(terminology_labels)
        exclude = ["Z760:cid10", "Z000:cid10"]
        result = prepare_dataframe(
            interim, terminology_labels, exclude_codes=exclude, skip_patterns=[]
        )

        labels = result["terminology_label"].unique().to_list()
        assert "Z760:cid10" not in labels
        assert "Z000:cid10" not in labels

    def test_exclude_codes_keeps_yes_codes(self, terminology_labels):
        interim = _build_interim_dataframe(terminology_labels)
        exclude = ["Z760:cid10", "Z000:cid10"]
        result = prepare_dataframe(
            interim, terminology_labels, exclude_codes=exclude, skip_patterns=[]
        )

        labels = result["terminology_label"].unique().to_list()
        assert "I10:cid10" in labels
        assert "ABP005:ab" in labels
        assert "K86:ciap2" in labels

    def test_no_exclude_codes_keeps_all(self, terminology_labels):
        interim = _build_interim_dataframe(terminology_labels)
        result = prepare_dataframe(
            interim, terminology_labels, exclude_codes=None, skip_patterns=[]
        )

        labels = set(result["terminology_label"].unique().to_list())
        expected = {"I10:cid10", "Z760:cid10", "ABP005:ab", "Z000:cid10", "K86:ciap2"}
        assert expected == labels

    def test_codes_not_in_exclude_list_are_kept(self, terminology_labels):
        """Codes not in the exclude list must remain even if they appear non-indicative-like."""
        extra_labels = terminology_labels.vstack(
            pl.DataFrame(
                {
                    "id": [99],
                    "code": ["EXTRA"],
                    "code_type": ["ab"],
                    "description": ["some extra code"],
                    "label": ["EXTRA:ab"],
                }
            )
        )
        rows = []
        for row in extra_labels.iter_rows(named=True):
            rows.append(
                {
                    "year": 2022,
                    "week": 1,
                    "sex": "M",
                    "age_group": "adult",
                    "code_type": row["code_type"],
                    "code": row["code"],
                    "quantity": 10,
                }
            )
        interim = pl.DataFrame(rows)
        exclude = ["Z760:cid10", "Z000:cid10"]
        result = prepare_dataframe(interim, extra_labels, exclude_codes=exclude, skip_patterns=[])
        labels = result["terminology_label"].unique().to_list()
        assert "EXTRA:ab" in labels

    def test_skip_patterns_removes_z_chapter_cid10(self, terminology_labels):
        interim = _build_interim_dataframe(terminology_labels)
        result = prepare_dataframe(interim, terminology_labels, skip_patterns=[("cid10", r"^Z")])
        labels = result["terminology_label"].unique().to_list()
        assert "Z760:cid10" not in labels
        assert "Z000:cid10" not in labels
        assert "I10:cid10" in labels
        assert "ABP005:ab" in labels
        assert "K86:ciap2" in labels

    def test_skip_patterns_respects_code_type_scope(self, terminology_labels):
        """A cid10 Z-pattern must not remove a ciap2 code that starts with Z."""
        extra_labels = terminology_labels.vstack(
            pl.DataFrame(
                {
                    "id": [99],
                    "code": ["Z96"],
                    "code_type": ["ciap2"],
                    "description": ["some ciap2 Z code"],
                    "label": ["Z96:ciap2"],
                }
            )
        )
        rows = []
        for row in extra_labels.iter_rows(named=True):
            rows.append(
                {
                    "year": 2022,
                    "week": 1,
                    "sex": "M",
                    "age_group": "adult",
                    "code_type": row["code_type"],
                    "code": row["code"],
                    "quantity": 10,
                }
            )
        interim = pl.DataFrame(rows)
        result = prepare_dataframe(interim, extra_labels, skip_patterns=[("cid10", r"^Z")])
        labels = result["terminology_label"].unique().to_list()
        assert "Z96:ciap2" in labels
        assert "Z760:cid10" not in labels

    def test_default_skip_patterns_use_union_of_registered_adapters(self, terminology_labels):
        """``skip_patterns=None`` should drop every adapter's Z-chapter (DATASUS cid10 + AKTIN icd10gm)
        without touching unrelated code_types."""
        extra_labels = terminology_labels.vstack(
            pl.DataFrame(
                {
                    "id": [99, 100],
                    "code": ["Z96", "Z00"],
                    "code_type": ["ciap2", "icd10gm"],
                    "description": ["some ciap2 Z code", "ICD-10-GM Z code"],
                    "label": ["Z96:ciap2", "Z00:icd10gm"],
                }
            )
        )
        rows = []
        for row in extra_labels.iter_rows(named=True):
            rows.append(
                {
                    "year": 2022,
                    "week": 1,
                    "sex": "M",
                    "age_group": "adult",
                    "code_type": row["code_type"],
                    "code": row["code"],
                    "quantity": 10,
                }
            )
        interim = pl.DataFrame(rows)
        result = prepare_dataframe(interim, extra_labels, skip_patterns=None)
        labels = set(result["terminology_label"].unique().to_list())

        assert "Z760:cid10" not in labels
        assert "Z00:icd10gm" not in labels
        assert "Z96:ciap2" in labels
        assert "I10:cid10" in labels

    def test_no_skip_patterns_keeps_all_codes(self, terminology_labels):
        interim = _build_interim_dataframe(terminology_labels)
        result = prepare_dataframe(interim, terminology_labels, skip_patterns=[])

        labels = set(result["terminology_label"].unique().to_list())
        expected = {"I10:cid10", "Z760:cid10", "ABP005:ab", "Z000:cid10", "K86:ciap2"}
        assert expected == labels
