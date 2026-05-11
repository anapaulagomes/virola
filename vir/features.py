"""Temporary re-export shim.

The contents of this module moved to ``vir.adapters.datasus.features`` as part
of the AKTIN adapter migration. This shim keeps existing imports working while
callers are updated. Remove once no consumer imports from ``vir.features``.

Deprecated: import from ``vir.adapters.datasus.features`` instead.
"""

from vir.adapters.datasus.features import (
    AGE_GROUP,
    CITY_NAMES,
    CODE_TYPES_MAPPING,
    IBGE_CITY_MAP,
    ORIGINAL_AGE_GROUP,
    ORIGINAL_TO_REDUCED_AGE_GROUP,
    POPULATION_RANGE,
)

__all__ = [
    "AGE_GROUP",
    "CITY_NAMES",
    "CODE_TYPES_MAPPING",
    "IBGE_CITY_MAP",
    "ORIGINAL_AGE_GROUP",
    "ORIGINAL_TO_REDUCED_AGE_GROUP",
    "POPULATION_RANGE",
]
