"""AKTIN / BfArM ICD-10-GM terminology builder.

Parses the vendored ``icd10gm2025syst_kodes.txt`` (BfArM ICD-10-GM 2025) into
the canonical ``(code, code_type, description)`` labels shape consumed by
``vir setup terminologies``.
"""

from pathlib import Path

import polars as pl

ICD10GM_KODES_FILE = Path(__file__).parent / "references" / "icd10gm2025syst_kodes.txt"


def build_icd10gm_labels() -> pl.DataFrame:
    """Return rows of ``(code, code_type, description)`` for every ICD-10-GM code.

    Source format (BfArM ICD-10-GM, semicolon-separated, no header, 28 columns):

    - column 7 — Kode mit Punkt: the dot-form clinicians record (``M22.0``,
      ``S93.40``) or, for 3-character roots / 4-character hierarchical-only
      codes, the clean root (``A00``, ``S93.4``). Never carries BfArM's
      ``-`` placeholder.
    - column 9 — Klassentitel: long German title used as ``description``.

    Both terminal (``T``) and hierarchical-only (``N``) rows are kept so that
    AKTIN records using truncated codes (e.g. ``S93.4`` without a 5th char)
    still get a meaningful parent label rather than a NULL join.
    """
    raw = pl.read_csv(
        ICD10GM_KODES_FILE,
        separator=";",
        has_header=False,
        encoding="utf8",
        infer_schema_length=0,
    )
    return (
        raw.select(
            code=pl.col("column_7"),
            description=pl.col("column_9"),
        )
        .with_columns(code_type=pl.lit("icd10gm"))
        .select("code", "code_type", "description")
    )
