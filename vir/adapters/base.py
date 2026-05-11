from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl


class Adapter(ABC):
    """Contract for dataset-specific adapters that produce the canonical aggregated row.

    Concrete adapters live under ``vir/adapters/<name>/`` and are registered in
    ``vir.adapters.ADAPTERS``. Everything downstream of ``prepare_dataframe``
    (views, SNF, clusters, DuckDB writes) consumes the canonical row and does
    not need to know which adapter produced it.
    """

    name: str

    @abstractmethod
    def clean(
        self,
        raw_path: Path,
        output_path: Path,
        stratum: str | None = None,
        **options,
    ) -> Path:
        """Read the adapter's native raw format; write the cleaned interim parquet."""

    @abstractmethod
    def build_terminology_labels(self) -> pl.DataFrame:
        """Return rows of ``(code, code_type, label)``; IDs are assigned by the DB on insert."""

    @abstractmethod
    def stratum_metadata(self, stratum: str) -> dict:
        """Return ``{'city': ..., 'region': ..., 'population_range': ...}``."""

    @property
    @abstractmethod
    def strata(self) -> list[str]:
        """Strata this adapter owns (e.g. ``['germany']`` for AKTIN)."""

    @property
    @abstractmethod
    def skip_patterns(self) -> list[tuple[str, str]]:
        """``(code_type, regex)`` pairs used to drop codes during ``prepare_dataframe``."""

    @property
    @abstractmethod
    def code_types(self) -> list[str]:
        """Terminology code-type identifiers this adapter emits (e.g. ``['icd10gm']``)."""

    @property
    def indicative_codes_file(self) -> Path | None:
        """Optional path to the indicative-codes labeling CSV. ``None`` when not provided."""
        return None
