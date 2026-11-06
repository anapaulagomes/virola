import polars as pl

from vir.adapters import ADAPTERS
from vir.adapters.base import Adapter
from vir.adapters.datasus.features import CITY_NAMES
from vir.config import RAW_DATA_FILE


def _datasus():
    return ADAPTERS["datasus"]


def test_datasus_adapter_is_registered():
    assert "datasus" in ADAPTERS
    assert isinstance(ADAPTERS["datasus"], Adapter)


def test_datasus_adapter_name():
    assert _datasus().name == "datasus"


def test_default_model_is_sapbert_pubmedbert():
    """DATASUS keeps the original English biomedical SapBERT so that existing
    embeddings and clinical-view caches stay valid."""
    assert _datasus().default_model == "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


def test_terminology_csv_filename_is_the_legacy_brazilian_name():
    """Keep the existing filename so existing DATASUS embeddings runs don't break."""
    assert _datasus().terminology_csv_filename == "cid10_ciap2_ab.csv"


def test_cleaned_filename_preserves_existing_raw_stem_and_city_slug():
    """Format: ``cleaned_<raw_stem>_<city_slug>.parquet`` matches files written by
    the legacy DATASUS notebook so existing interim parquets keep being recognised."""
    expected_stem = RAW_DATA_FILE.parent.stem

    assert _datasus().cleaned_filename("Salvador") == f"cleaned_{expected_stem}_salvador.parquet"
    assert (
        _datasus().cleaned_filename("Belo Horizonte")
        == f"cleaned_{expected_stem}_belo_horizonte.parquet"
    )


def test_cleaned_filename_uses_all_cities_when_no_stratum_is_given():
    expected_stem = RAW_DATA_FILE.parent.stem
    assert _datasus().cleaned_filename(None) == f"cleaned_{expected_stem}_all_cities.parquet"


def test_strata_returns_the_ibge_city_names():
    assert set(_datasus().strata) == set(CITY_NAMES)


def test_code_types_lists_cid10_ciap2_and_ab():
    assert _datasus().code_types == ["cid10", "ciap2", "ab"]


def test_skip_patterns_drop_cid10_z_chapter():
    assert _datasus().skip_patterns == [("cid10", r"^Z")]


def test_stratum_metadata_resolves_known_city():
    metadata = _datasus().stratum_metadata("Salvador")
    assert metadata == {"city": "Salvador", "region": "Nordeste", "population_range": "> 500 mil"}


def test_stratum_metadata_matches_dataset_name_form():
    """``belo_horizonte`` is the slug form used in dataset names; both must resolve."""
    metadata = _datasus().stratum_metadata("belo_horizonte")
    assert metadata["city"] == "Belo Horizonte"


def test_stratum_metadata_returns_none_values_when_unknown():
    metadata = _datasus().stratum_metadata("atlantis")
    assert metadata == {"city": None, "region": None, "population_range": None}


def test_indicative_codes_file_points_at_datasus_references():
    path = _datasus().indicative_codes_file
    assert path is not None
    assert path.name == "indicative_codes_yes_no__2022_2025__90perc.csv"
    assert path.exists()


def test_build_terminology_labels_returns_non_empty_dataframe_with_code_columns():
    labels = _datasus().build_terminology_labels()
    assert isinstance(labels, pl.DataFrame)
    assert labels.height > 0
    assert {"code", "code_type"}.issubset(set(labels.columns))
    assert set(labels["code_type"].unique().to_list()) == {"cid10", "ciap2", "ab"}


def _build_minimal_raw_aesop(tmp_path):
    """Write a tiny AESOP-shaped parquet covering two cities."""
    raw = pl.DataFrame(
        {
            "CIDCIAP": ["CID(I10)", "CIAP(K86)"],
            "ANO": [2023, 2023],
            "SEMANA": [10, 12],
            "SG_UF": ["BA", "AM"],
            "SEXO": ["M", "F"],
            "FX_ETARIA": ["30 a 39 anos", "5 a 11 anos"],
            "QT": [5, 3],
            "CO_MUNICIPIO_IBGE": [292740, 130260],
            "CO_CNES": ["1", "2"],
            "Data_folder": ["x", "x"],
        }
    )
    raw_path = tmp_path / "raw.parquet"
    raw.write_parquet(raw_path)
    return raw_path


def test_clean_produces_canonical_interim_parquet(tmp_path):
    raw_path = _build_minimal_raw_aesop(tmp_path)
    output_path = tmp_path / "cleaned.parquet"

    returned = _datasus().clean(raw_path, output_path)

    assert returned == output_path
    cleaned = pl.read_parquet(output_path)
    expected = {"year", "week", "sex", "age_group", "code_type", "code", "quantity", "city"}
    assert expected.issubset(set(cleaned.columns))
    assert cleaned.height == 2


def test_clean_with_stratum_filters_to_a_single_city(tmp_path):
    raw_path = _build_minimal_raw_aesop(tmp_path)
    output_path = tmp_path / "cleaned.parquet"

    _datasus().clean(raw_path, output_path, stratum="Salvador")

    cleaned = pl.read_parquet(output_path)
    assert cleaned["city"].unique().to_list() == ["Salvador"]
    assert cleaned.height == 1


def test_clean_finds_the_aesop_export_inside_the_raw_data_dir(tmp_path):
    """`vir pipeline run` passes RAW_DATA_DIR (e.g. data/raw), which holds the export folder."""
    single_file = _build_minimal_raw_aesop(tmp_path)
    raw_data_dir = tmp_path / "raw"
    export_dir = raw_data_dir / RAW_DATA_FILE.parent.name
    export_dir.mkdir(parents=True)
    single_file.rename(export_dir / "part-00000.parquet")
    output_path = tmp_path / "cleaned.parquet"

    _datasus().clean(raw_data_dir, output_path)

    assert pl.read_parquet(output_path).height == 2


def test_clean_reads_an_export_dir_given_directly(tmp_path):
    single_file = _build_minimal_raw_aesop(tmp_path)
    export_dir = tmp_path / "export.parquet"
    export_dir.mkdir()
    single_file.rename(export_dir / "part-00000.parquet")
    output_path = tmp_path / "cleaned.parquet"

    _datasus().clean(export_dir, output_path)

    assert pl.read_parquet(output_path).height == 2
