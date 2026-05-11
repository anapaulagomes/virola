"""DATASUS-specific helpers.

Lives alongside the DATASUS adapter because the per-city parquet sharding
convention (one cleaned/prepared file per IBGE city) is DATASUS-specific.
"""

from typing import Literal

import polars as pl

from vir.adapters.datasus.features import CITY_NAMES
from vir.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_FILE


def load_all_cities(from_: Literal["cleaned", "prepared"]) -> pl.DataFrame:
    """Concatenate per-city parquet files from a pipeline stage into one DataFrame.

    Reads ``<from_>_<raw_stem>_<city_slug>.parquet`` for each city in
    :data:`vir.adapters.datasus.features.CITY_NAMES` and concatenates them
    vertically.

    Stage layout:

    - ``from_="cleaned"`` reads from :data:`vir.config.INTERIM_DATA_DIR`
    - ``from_="prepared"`` reads from :data:`vir.config.PROCESSED_DATA_DIR`

    ``<raw_stem>`` is ``RAW_DATA_FILE.parent.stem`` (e.g.
    ``base_aesop_raw_extracao_01042026``); ``<city_slug>`` is
    ``city.lower().replace(' ', '_')`` — accents are preserved (so
    ``Anápolis`` becomes ``anápolis``), matching the convention used by the
    cleaning and preparation notebooks when they write per-city outputs.

    The ``city`` column is injected when missing (the per-city ``prepared``
    files drop it during aggregation), so the result is equivalent to the
    materialized ``..._all_cities.parquet`` (cleaned) and
    ``..._all_cities_multiple_cities.parquet`` (prepared) cache files.
    """
    match from_:
        case "cleaned":
            directory = INTERIM_DATA_DIR
        case "prepared":
            directory = PROCESSED_DATA_DIR
        case _:
            raise ValueError(f"Unknown source {from_!r}; expected 'cleaned' or 'prepared'.")

    raw_stem = RAW_DATA_FILE.parent.stem
    frames = []
    for city in CITY_NAMES:
        city_slug = city.lower().replace(" ", "_")
        path = directory / f"{from_}_{raw_stem}_{city_slug}.parquet"
        frame = pl.read_parquet(path)
        if "city" not in frame.columns:
            frame = frame.with_columns(city=pl.lit(city))
        frames.append(frame)
    return pl.concat(frames)
