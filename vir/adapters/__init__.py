from vir.adapters.aktin import AktinAdapter
from vir.adapters.base import Adapter
from vir.adapters.datasus import DatasusAdapter

ADAPTERS: dict[str, Adapter] = {
    "datasus": DatasusAdapter(),
    "aktin": AktinAdapter(),
}


def known_strata() -> set[str]:
    """Return the union of strata declared by every registered adapter."""
    return {stratum for adapter in ADAPTERS.values() for stratum in adapter.strata}


def adapter_for_stratum(stratum: str) -> Adapter:
    """Return the unique adapter that declares this stratum.

    Raises:
        LookupError: if no adapter claims the stratum, or if more than one does.
    """
    owners = [adapter for adapter in ADAPTERS.values() if stratum in adapter.strata]
    if not owners:
        raise LookupError(f"unknown stratum: {stratum!r}")
    if len(owners) > 1:
        raise LookupError(f"ambiguous stratum: {stratum!r} is claimed by {len(owners)} adapters")
    return owners[0]


def stratum_metadata(stratum: str) -> dict:
    """Resolve ``{city, region, population_range}`` for a stratum or dataset slug.

    Iterates the registered adapters and asks each to interpret the value.
    The first adapter that recognises it (returns a non-null ``city``) wins.
    Falls back to all-None when no adapter claims the value — matching the
    historical behaviour of the framework's ``_city_metadata`` helper.

    This accepts both display-form stratum names (e.g. ``"Belo Horizonte"``)
    and dataset-name slugs (e.g. ``"belo_horizonte"``); each adapter is
    responsible for normalising what it owns.
    """
    for adapter in ADAPTERS.values():
        metadata = adapter.stratum_metadata(stratum)
        if metadata.get("city") is not None:
            return metadata
    return {"city": None, "region": None, "population_range": None}
