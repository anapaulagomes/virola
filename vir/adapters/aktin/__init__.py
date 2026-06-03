import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import polars as pl
from loguru import logger

from vir.adapters.aktin.cleaning import clean_aktin_raw
from vir.adapters.aktin.terminology import build_icd10gm_labels
from vir.adapters.aktin.viz import AGE_GROUP_COLOR, CODE_TYPE_COLOR, SEX_COLOR
from vir.adapters.base import Adapter

_FileSource = Union[Path, tuple[Path, str]]


@dataclass(frozen=True)
class _Site:
    """One AKTIN site's pair of source streams (either filesystem paths or zip members)."""

    name: str
    case_source: _FileSource
    diag_source: _FileSource


def _read_aktin_tsv(source: _FileSource) -> pl.DataFrame:
    """Read a tab-separated AKTIN file either directly or from inside a zip.

    Uses ``truncate_ragged_lines=True``: AKTIN free-text columns (e.g.
    ``isolation_reason``, ``zuweisung``) occasionally contain embedded tabs in
    real exports, producing rows with more fields than the header declares.
    Truncating the trailing fields lets the row survive — the dropped data is
    in narrative columns that we don't consume downstream.
    """
    read_kwargs = dict(separator="\t", infer_schema_length=0, truncate_ragged_lines=True)
    if isinstance(source, tuple):
        zip_path, member = source
        with zipfile.ZipFile(zip_path) as archive, archive.open(member) as stream:
            payload = stream.read()
        return pl.read_csv(io.BytesIO(payload), **read_kwargs)
    return pl.read_csv(source, **read_kwargs)


def _discover_sites(raw_path: Path) -> list[_Site]:
    """Find every AKTIN site under ``raw_path``.

    Recognises two layouts: an unpacked ``<site>/`` containing both
    ``case_data.txt`` and ``diag_data.txt``, and a ``<site>_result.zip``
    whose members include both.

    Raises ``FileNotFoundError`` for a site that has one of the two files but
    not the other — silent orphan-site loss is the worst failure mode (every
    encounter from that site would disappear at the inner-join).
    """
    sites: list[_Site] = []

    for case_path in sorted(raw_path.glob("**/case_data.txt")):
        diag_path = case_path.parent / "diag_data.txt"
        if not diag_path.exists():
            raise FileNotFoundError(
                f"Site {case_path.parent.name!r} has case_data.txt but no diag_data.txt"
            )
        sites.append(
            _Site(
                name=case_path.parent.name or "root", case_source=case_path, diag_source=diag_path
            )
        )

    for zip_path in sorted(raw_path.glob("**/*.zip")):
        with zipfile.ZipFile(zip_path) as archive:
            members = set(archive.namelist())
        has_case = "case_data.txt" in members
        has_diag = "diag_data.txt" in members
        if not has_case and not has_diag:
            continue
        if has_case != has_diag:
            raise FileNotFoundError(
                f"Zip {zip_path.name!r} has only one of case_data.txt / diag_data.txt"
            )
        sites.append(
            _Site(
                name=zip_path.stem,
                case_source=(zip_path, "case_data.txt"),
                diag_source=(zip_path, "diag_data.txt"),
            )
        )

    return sites


def _namespace_encounter_num(frame: pl.DataFrame, site_name: str) -> pl.DataFrame:
    """Prefix ``a_encounter_num`` with the site name so cross-site encounter-id collisions don't merge."""
    return frame.with_columns(
        a_encounter_num=pl.concat_str(
            [pl.lit(f"{site_name}:"), pl.col("a_encounter_num").cast(pl.Utf8)]
        )
    )


class AktinAdapter(Adapter):
    """Adapter for the German AKTIN emergency-department dataset.

    Single stratum (``"germany"`` — the country as a whole), per-encounter
    diagnoses (one row per encounter × confirmed diagnosis after cleaning),
    ICD-10-GM labels from the vendored BfArM 2025 release.
    """

    name = "aktin"
    default_model = os.getenv(
        "VIROLA_AKTIN_MODEL", "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
    )
    terminology_csv_filename = "icd10gm_aktin.csv"

    def clean(
        self,
        raw_path: Path,
        output_path: Path,
        stratum: str | None = None,
        **options,
    ) -> Path:
        """Pool every AKTIN site under ``raw_path`` into a single canonical interim parquet.

        Each site (unpacked directory or ``<site>_result.zip``) must contribute
        both ``case_data.txt`` and ``diag_data.txt`` — orphan sites raise.
        ``a_encounter_num`` is namespaced per site so id collisions don't merge.
        """
        sites = _discover_sites(raw_path)
        if not sites:
            raise FileNotFoundError(
                f"No AKTIN sites found under {raw_path}. Expected ``<site>/case_data.txt`` + "
                "``diag_data.txt`` pairs or ``<site>_result.zip`` archives."
            )

        logger.info(f"AKTIN clean: pooling {len(sites)} site(s) under {raw_path}")

        case_frames = [
            _namespace_encounter_num(_read_aktin_tsv(site.case_source), site.name)
            for site in sites
        ]
        diag_frames = [
            _namespace_encounter_num(_read_aktin_tsv(site.diag_source), site.name)
            for site in sites
        ]

        cases = pl.concat(case_frames, how="diagonal_relaxed")
        diagnoses = pl.concat(diag_frames, how="diagonal_relaxed")

        cleaned = clean_aktin_raw(cases, diagnoses)
        cleaned.write_parquet(output_path)
        return output_path

    def build_terminology_labels(self) -> pl.DataFrame:
        return build_icd10gm_labels()

    def stratum_metadata(self, stratum: str) -> dict:
        if "germany" in stratum.lower():
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

    def cleaned_filename(self, stratum: str | None) -> str:
        slug = (stratum or "all").lower().replace(" ", "_")
        return f"cleaned_aktin_{slug}.parquet"

    @property
    def viz_palettes(self) -> dict[str, dict[str, str]]:
        return {
            "sex": SEX_COLOR,
            "age_group": AGE_GROUP_COLOR,
            "code_type": CODE_TYPE_COLOR,
        }
