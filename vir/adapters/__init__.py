from vir.adapters.base import Adapter
from vir.adapters.datasus import DatasusAdapter

ADAPTERS: dict[str, Adapter] = {
    "datasus": DatasusAdapter(),
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
