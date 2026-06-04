import os
import unicodedata
from pathlib import Path

import numba
from huggingface_hub import snapshot_download

from vir.config import HF_DEFAULT_HOME, logger


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
    if not os.getenv("HF_FROM_LIMITED_ENV"):
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


def find_dataset_file(dataset_name: str, candidate_paths: list[str]) -> str | None:
    """Return the candidate path whose name contains ``dataset_name``.

    Matching is robust to Unicode normalization differences: macOS stores
    filenames decomposed (NFD), while the database stores dataset names
    composed (NFC), so a naive substring test fails for accented cities such
    as ``anápolis`` or ``gravataí``. Both sides are normalized to NFC before
    comparison; the original path string is returned unchanged.
    """
    needle = unicodedata.normalize("NFC", dataset_name)
    for path in candidate_paths:
        if needle in unicodedata.normalize("NFC", path):
            return path
    return None


def get_n_jobs():
    if slurm_cpus := os.getenv("SLURM_CPUS_PER_TASK"):
        return int(slurm_cpus)
    return numba.get_num_threads()


def slugify(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower().replace(" ", "_").replace("/", "_")
