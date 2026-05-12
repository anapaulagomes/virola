"""DATASUS / AESOP raw data cleaner.

Extracts the per-row transformations originally defined inline in
``vir/cleaning/data_cleaning.py`` (the marimo notebook). The notebook becomes a
thin UI shell around this function.
"""

import polars as pl

from vir.adapters.datasus.features import (
    CODE_TYPES_MAPPING,
    IBGE_CITY_MAP,
    ORIGINAL_TO_REDUCED_AGE_GROUP,
)


def _ibge_lookup_frame() -> pl.DataFrame:
    """Build the IBGE municipality → (city, region, population_range, lat, lon) lookup."""
    rows = [
        {"CO_MUNICIPIO_IBGE": ibge_code, **{k: v for k, v in entry.items() if k != "state"}}
        for ibge_code, entry in IBGE_CITY_MAP.items()
    ]
    return pl.DataFrame(rows)


AESOP_COLUMN_RENAMES: dict[str, str] = {
    "ANO": "year",
    "SEMANA": "week",
    "SG_UF": "state",
    "SEXO": "sex",
    "FX_ETARIA": "age_group",
    "QT": "quantity",
}


def clean_datasus_raw(raw_dataframe: pl.DataFrame) -> pl.DataFrame:
    """Apply the AESOP-specific transformations to a raw AESOP dataframe.

    The result is the cleaned interim shape consumed downstream by
    ``prepare_dataframe``.
    """
    dataframe = (
        raw_dataframe.with_columns(
            pl.col("CIDCIAP").str.extract(r"^([A-Z]+)\(", 1).alias("code_type"),
            pl.col("CIDCIAP").str.extract(r"\(([^)]+)\)", 1).alias("code"),
        )
        .drop(["CIDCIAP", "Data_folder"])
        .drop_nulls(subset=["ANO", "SEMANA", "FX_ETARIA"])
    )

    dataframe = dataframe.with_columns(
        code=pl.when(pl.col("code_type").eq("AB"))
        .then(pl.col("code").str.replace(r"-\d+$", ""))
        .otherwise(pl.col("code"))
    )

    dataframe = dataframe.with_columns(
        code_type=pl.col("code_type").replace_strict(CODE_TYPES_MAPPING, default=None),
        FX_ETARIA=pl.col("FX_ETARIA").replace_strict(ORIGINAL_TO_REDUCED_AGE_GROUP, default=None),
    )

    dataframe = dataframe.join(_ibge_lookup_frame(), on="CO_MUNICIPIO_IBGE", how="left")
    dataframe = dataframe.drop(["CO_CNES", "CO_MUNICIPIO_IBGE"])

    return dataframe.rename(AESOP_COLUMN_RENAMES)
