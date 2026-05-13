from pathlib import Path

import polars as pl

from vir.adapters.base import Adapter
from vir.adapters.datasus.cleaning import clean_datasus_raw
from vir.adapters.datasus.features import CITY_NAMES, IBGE_CITY_MAP
from vir.adapters.datasus.terminology import create_terminologies_df
from vir.adapters.datasus.viz import AGE_GROUP_COLOR, CODE_TYPE_COLOR, SEX_COLOR
from vir.config import RAW_DATA_FILE

_REFERENCES_DIR = Path(__file__).parent / "references"
INDICATIVE_CODES_FILE = _REFERENCES_DIR / "indicative_codes_yes_no__2022_2025__90perc.csv"


class DatasusAdapter(Adapter):
    """Adapter for the Brazilian DATASUS / AESOP dataset.

    Wraps the existing DATASUS-specific cleaning, terminology, and stratum
    semantics into the framework's :class:`vir.adapters.base.Adapter` contract.
    """

    name = "datasus"
    default_model = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    terminology_csv_filename = "cid10_ciap2_ab.csv"

    def clean(
        self,
        raw_path: Path,
        output_path: Path,
        stratum: str | None = None,
        **options,
    ) -> Path:
        raw_dataframe = (
            pl.read_parquet(raw_path / "*.parquet")
            if raw_path.is_dir()
            else pl.read_parquet(raw_path)
        )
        raw_dataframe = raw_dataframe.with_row_index("unique_id", offset=0)
        cleaned = clean_datasus_raw(raw_dataframe)
        if stratum:
            cleaned = cleaned.filter(pl.col("city").eq(stratum))
        cleaned.write_parquet(output_path)
        return output_path

    def build_terminology_labels(self) -> pl.DataFrame:
        return create_terminologies_df()

    def stratum_metadata(self, stratum: str) -> dict:
        normalised_stratum = stratum.lower()
        for entry in IBGE_CITY_MAP.values():
            normalised_city = entry["city"].lower().replace(" ", "_")
            if normalised_city in normalised_stratum:
                return {
                    "city": entry["city"],
                    "region": entry["region"],
                    "population_range": entry["population_range"],
                }
        return {"city": None, "region": None, "population_range": None}

    @property
    def strata(self) -> list[str]:
        return list(CITY_NAMES)

    @property
    def skip_patterns(self) -> list[tuple[str, str]]:
        return [("cid10", r"^Z")]

    @property
    def code_types(self) -> list[str]:
        return ["cid10", "ciap2", "ab"]

    @property
    def indicative_codes_file(self) -> Path | None:
        return INDICATIVE_CODES_FILE

    def cleaned_filename(self, stratum: str | None) -> str:
        raw_stem = RAW_DATA_FILE.parent.stem
        slug = stratum.lower().replace(" ", "_") if stratum else "all_cities"
        return f"cleaned_{raw_stem}_{slug}.parquet"

    @property
    def viz_palettes(self) -> dict[str, dict[str, str]]:
        return {
            "sex": SEX_COLOR,
            "age_group": AGE_GROUP_COLOR,
            "code_type": {key.lower(): value for key, value in CODE_TYPE_COLOR.items()},
        }
