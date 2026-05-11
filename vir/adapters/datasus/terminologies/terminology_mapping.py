"""Cross-terminology mapping: AB → CIAP2 → CID10.

Sources
-------
references/abp_ciap2.csv   — AB programme codes mapped to CIAP2
references/ciap2-cid10.csv — CIAP2 codes mapped to CID10 (one-to-many)
"""

import polars as pl

from vir.config import REFERENCES_DIR, logger

_ABP_CIAP2 = REFERENCES_DIR / "abp_ciap2.csv"
_CIAP2_CID10 = REFERENCES_DIR / "ciap2-cid10.csv"


def load_mapping() -> pl.DataFrame:
    """Return a long-format DataFrame of all cross-terminology links.

    Columns: ab_code, ab_programme_code, ciap2_code, cid10_code
    """
    abp = (
        pl.read_csv(_ABP_CIAP2)
        .rename(
            {
                "CIAP2 relacionada": "ciap2_code",
                "Descrição AB": "ab_code",
                "Código AB": "ab_programme_code",
            }
        )
        .with_columns(
            pl.when(pl.col("ciap2_code").str.contains(r"^[A-Z]\d{2}$"))
            .then(pl.col("ciap2_code"))
            .otherwise(None)
            .alias("ciap2_code")
        )
    )

    ciap2_cid10 = (
        pl.read_csv(_CIAP2_CID10)
        .rename(
            {
                "Código da CIAP2": "ciap2_code",
                "Código da CID10": "cid10_code",
            }
        )
        .select("ciap2_code", "cid10_code")
    )

    return abp.join(ciap2_cid10, on="ciap2_code", how="left").select(
        "ab_code", "ab_programme_code", "ciap2_code", "cid10_code"
    )


def save_mapping(path=None) -> pl.DataFrame:
    """Save the full mapping to CSV and return the DataFrame."""
    out = path or (REFERENCES_DIR / "terminology_mapping.csv")
    df = load_mapping()
    df.write_csv(out)
    logger.info(f"Saved {len(df)} mapping rows to {out}")
    return df


if __name__ == "__main__":
    _df = save_mapping()
    logger.info(
        f"AB codes: {_df['ab_code'].n_unique()} | "
        f"CIAP2: {_df['ciap2_code'].drop_nulls().n_unique()} | "
        f"CID10: {_df['cid10_code'].drop_nulls().n_unique()}"
    )
