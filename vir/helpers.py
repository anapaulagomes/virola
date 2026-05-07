import os
import unicodedata
from pathlib import Path
from typing import Literal

import numba
import polars as pl
from huggingface_hub import snapshot_download

from vir.config import (
    HF_DEFAULT_HOME,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_FILE,
    logger,
)
from vir.features import CITY_NAMES


def get_or_download_model(
    repo_id: str,
    revision: str = "main",
) -> Path:
    """
    Parse model name to locally stored weights. It will download the model from the library only when in Limited environment.
    Args:
        model_ref (str) : Model reference containing org_name/model_name such as 'meta-llama/Llama-2-7b-chat-hf'.
        revision (str): Model revision branch. Defaults to 'main'.

    Returns:
        str: path to model weights within model directory
    """
    if os.getenv("HF_FROM_LIMITED_ENV") is None:
        logger.info("Regular environment detected.")
        return repo_id

    logger.warning("Limited environment detected. Downloading model weights to cache folder...")

    models_dir = HF_DEFAULT_HOME / "hub"
    cached_name = f"models--{repo_id.replace('/', '--')}"
    cached_dir = models_dir / cached_name
    if not cached_dir.exists():
        snapshot_download(repo_id=repo_id, revision=revision, cache_dir=models_dir)

    snapshot_hash = (cached_dir / "refs" / revision).read_text()
    return cached_dir / "snapshots" / snapshot_hash


def suffix_from(filename: Path, break_on="20municipios", remove_extension=False):
    if break_on:
        suffix = filename.name.rsplit(break_on)[-1]
    else:
        suffix = filename.name
    if remove_extension:
        return suffix.replace(".parquet", "")
    return suffix


def extract_dataset_name(file_path: str) -> str:
    """
    Extract dataset name from file path.

    Strips the ``cleaned_`` or ``prepared_`` prefix from the filename stem,
    returning the suffix that identifies the dataset and city.
    """
    stem = Path(file_path).stem
    for prefix in ("cleaned_", "prepared_"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def get_n_jobs():
    if slurm_cpus := os.getenv("SLURM_CPUS_PER_TASK"):
        return int(slurm_cpus)
    return numba.get_num_threads()


def slugify(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower().replace(" ", "_")


def load_all_cities(from_: Literal["cleaned", "prepared"]) -> pl.DataFrame:
    """Concatenate per-city parquet files from a pipeline stage into one DataFrame.

    Reads ``<from_>_<raw_stem>_<city_slug>.parquet`` for each city in
    :data:`vir.features.CITY_NAMES` and concatenates them vertically.

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
