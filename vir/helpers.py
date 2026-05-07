import os
import unicodedata
from pathlib import Path

import numba
import polars as pl
from huggingface_hub import snapshot_download

from vir.config import HF_DEFAULT_HOME, INTERIM_DATA_DIR, RAW_DATA_FILE, logger
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


def load_cleaned_all_cities() -> pl.DataFrame:
    """Concatenate per-city cleaned parquet files into a single DataFrame.

    Reads ``cleaned_<raw_stem>_<city_slug>.parquet`` from
    :data:`vir.config.INTERIM_DATA_DIR` for each city in
    :data:`vir.features.CITY_NAMES` and concatenates them vertically.

    ``<raw_stem>`` is the directory name of :data:`vir.config.RAW_DATA_FILE`
    without the ``.parquet`` suffix; ``<city_slug>`` is
    ``city.lower().replace(' ', '_')`` (matching the convention used when the
    cleaning notebook writes per-city outputs).

    Replaces the materialized
    ``cleaned_<raw_stem>_all_cities.parquet`` cache file.
    """
    raw_stem = RAW_DATA_FILE.parent.stem
    paths = [
        INTERIM_DATA_DIR / f"cleaned_{raw_stem}_{city.lower().replace(' ', '_')}.parquet"
        for city in CITY_NAMES
    ]
    return pl.concat(pl.read_parquet(path) for path in paths)
