import os
from pathlib import Path

from huggingface_hub import snapshot_download

from vir.config import logger

HF_DEFAULT_HOME = Path(os.getenv("HF_HOME", "~/.cache/huggingface"))


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


def get_n_jobs():
    return int(os.getenv("NUMBER_OF_THREADS", -1))
