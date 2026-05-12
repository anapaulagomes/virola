import logging

import polars as pl

from vir.config import PROCESSED_DATA_DIR, REFERENCES_DIR

logger = logging.getLogger(__file__)


def create_cid10_df():
    _exclude_columns = [
        "ST_AGRAVO_NOTIFICACAO",
        "CO_SEXO",
        "ST_ESTADIO",
        "CO_IRRADIADOS",
        "DT_COMPETENCIA",
        "CO_VERSAO",
        "ST_ACAO",
    ]
    cid10 = pl.read_csv(
        REFERENCES_DIR / "RepositorioTerminologia_202506" / "tb_cid.csv",
        encoding="iso-8859-1",
        separator=";",
        has_header=False,
        new_columns=[
            "code",
            "description",
            "ST_AGRAVO_NOTIFICACAO",
            "CO_SEXO",
            "ST_ESTADIO",
            "CO_IRRADIADOS",
            "DT_COMPETENCIA",
            "CO_VERSAO",
            "ST_ACAO",
        ],
    )
    cid10 = cid10.drop(_exclude_columns)
    cid10 = cid10.with_columns(
        code_type=pl.lit("cid10"), description=pl.col("description").str.to_lowercase()
    )
    return cid10


def create_ciap2_df():
    ciap2_cid10 = pl.read_csv(REFERENCES_DIR / "ciap2-cid10.csv")
    ciap2_cid10 = ciap2_cid10.rename(
        {
            "Código da CIAP2": "code",
            "Código da CID10": "related_cid10",
            "Descrição da CIAP2": "description",
            "Descrição da CID10": "related_cid10_description",
        }
    )
    ciap2_cid10 = ciap2_cid10.with_columns(
        code_type=pl.lit("ciap2"),
        description=pl.col("description").str.to_lowercase(),
        related_cid10_description=pl.col("related_cid10_description").str.to_lowercase(),
    )
    return ciap2_cid10


def create_ab_df():
    abp_ciap2 = pl.read_csv(REFERENCES_DIR / "abp_ciap2.csv")
    abp_ciap2 = abp_ciap2.rename(
        {
            "CIAP2 relacionada": "related_ciap2",
            "Código AB": "code",
            "Descrição AB": "description",
        }
    )
    abp_ciap2 = abp_ciap2.with_columns(
        code_type=pl.lit("ab"), description=pl.col("description").str.to_lowercase()
    )
    return abp_ciap2


def create_terminologies_df():
    cid10 = create_cid10_df()
    ciap2_cid10 = create_ciap2_df()
    abp_ciap2 = create_ab_df()
    codes = pl.concat(
        [
            cid10,
            ciap2_cid10[["code", "description", "code_type"]],
            abp_ciap2[["code", "description", "code_type"]],
        ]
    )
    return codes


def main():
    """Create terminology CSV file from source data."""
    output_file = PROCESSED_DATA_DIR / "cid10_ciap2_ab.csv"
    # TODO if there are duplicates, add 1, 2... to it
    create_terminologies_df().write_csv(str(output_file))
    logger.info(f"Terminologies file created: {output_file}")


if __name__ == "__main__":
    main()
