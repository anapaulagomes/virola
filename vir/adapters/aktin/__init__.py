from pathlib import Path

import polars as pl

from vir.adapters.aktin.cleaning import clean_aktin_raw
from vir.adapters.aktin.terminology import build_icd10gm_labels
from vir.adapters.base import Adapter


class AktinAdapter(Adapter):
    """Adapter for the German AKTIN emergency-department dataset.

    Single stratum (``"germany"`` — the country as a whole), per-encounter
    diagnoses (one row per encounter × confirmed diagnosis after cleaning),
    ICD-10-GM labels from the vendored BfArM 2025 release.
    """

    name = "aktin"
    default_model = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
    terminology_csv_filename = "icd10gm_aktin.csv"

    def clean(
        self,
        raw_path: Path,
        output_path: Path,
        stratum: str | None = None,
        **options,
    ) -> Path:
        """Pool every ``case_data.txt`` / ``diag_data.txt`` pair under ``raw_path``.

        Layout expected:

        - ``raw_path/`` directly containing a single site's ``case_data.txt``
          and ``diag_data.txt`` (test fixtures, ad-hoc runs).
        - ``raw_path/<site>_result/`` for one or more sites — the AKTIN export
          shape, pooled into a single interim parquet.
        """
        case_files = sorted(raw_path.glob("**/case_data.txt"))
        diag_files = sorted(raw_path.glob("**/diag_data.txt"))
        if not case_files:
            raise FileNotFoundError(f"No case_data.txt found under {raw_path}")

        cases = pl.concat(
            [pl.read_csv(path, separator="\t", infer_schema_length=0) for path in case_files],
            how="diagonal_relaxed",
        )
        diagnoses = pl.concat(
            [pl.read_csv(path, separator="\t", infer_schema_length=0) for path in diag_files],
            how="diagonal_relaxed",
        )

        cleaned = clean_aktin_raw(cases, diagnoses)
        cleaned.write_parquet(output_path)
        return output_path

    def build_terminology_labels(self) -> pl.DataFrame:
        return build_icd10gm_labels()

    def stratum_metadata(self, stratum: str) -> dict:
        if stratum.lower() == "germany":
            return {"city": "germany", "region": None, "population_range": None}
        return {"city": None, "region": None, "population_range": None}

    @property
    def strata(self) -> list[str]:
        return ["germany"]

    @property
    def skip_patterns(self) -> list[tuple[str, str]]:
        return [("icd10gm", r"^Z")]

    @property
    def code_types(self) -> list[str]:
        return ["icd10gm"]
