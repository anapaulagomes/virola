from pathlib import Path

import polars as pl
import pytest

from vir.adapters import adapter_for_stratum, known_strata
from vir.adapters.base import Adapter


class _StubAdapter(Adapter):
    """Minimal concrete adapter implementing the full abstract surface for tests."""

    name = "stub"

    def __init__(self, strata: list[str] | None = None):
        self._strata = strata or []

    def clean(self, raw_path, output_path, stratum=None, **options):
        return output_path

    def build_terminology_labels(self) -> pl.DataFrame:
        return pl.DataFrame(schema={"code": pl.Utf8, "code_type": pl.Utf8, "label": pl.Utf8})

    def stratum_metadata(self, stratum: str) -> dict:
        return {"city": stratum, "region": None, "population_range": None}

    @property
    def strata(self) -> list[str]:
        return self._strata

    @property
    def skip_patterns(self) -> list[tuple[str, str]]:
        return []

    @property
    def code_types(self) -> list[str]:
        return []


def test_adapter_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Adapter()


def test_concrete_adapter_can_be_instantiated():
    adapter = _StubAdapter()
    assert adapter.clean(Path("/raw"), Path("/out")) == Path("/out")


def test_indicative_codes_file_defaults_to_none():
    assert _StubAdapter().indicative_codes_file is None


def test_known_strata_is_empty_when_registry_is_empty(monkeypatch):
    monkeypatch.setattr("vir.adapters.ADAPTERS", {})
    assert known_strata() == set()


def test_known_strata_unions_strata_across_registered_adapters(monkeypatch):
    first = _StubAdapter(strata=["alpha", "beta"])
    second = _StubAdapter(strata=["gamma"])
    monkeypatch.setattr("vir.adapters.ADAPTERS", {"first": first, "second": second})
    assert known_strata() == {"alpha", "beta", "gamma"}


def test_adapter_for_stratum_returns_the_unique_owner(monkeypatch):
    adapter = _StubAdapter(strata=["only-here"])
    monkeypatch.setattr("vir.adapters.ADAPTERS", {"one": adapter})
    assert adapter_for_stratum("only-here") is adapter


def test_adapter_for_stratum_raises_on_unknown_stratum(monkeypatch):
    adapter = _StubAdapter(strata=["alpha"])
    monkeypatch.setattr("vir.adapters.ADAPTERS", {"only": adapter})
    with pytest.raises(LookupError, match="unknown stratum"):
        adapter_for_stratum("not-registered")


def test_adapter_for_stratum_raises_when_two_adapters_claim_the_same_stratum(monkeypatch):
    first = _StubAdapter(strata=["shared"])
    second = _StubAdapter(strata=["shared"])
    monkeypatch.setattr("vir.adapters.ADAPTERS", {"first": first, "second": second})
    with pytest.raises(LookupError, match="ambiguous stratum"):
        adapter_for_stratum("shared")
