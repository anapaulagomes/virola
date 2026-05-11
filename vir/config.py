import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

logger.remove(0)
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
if os.getenv("DATA_DIR"):
    DATA_DIR = Path(os.getenv("DATA_DIR"))
RAW_DATA_DIR = DATA_DIR / "raw"
if os.getenv("RAW_DATA_DIR"):
    RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR"))

INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CLUSTERING_DIR = DATA_DIR / "clustering"
REFERENCES_DIR = PROJ_ROOT / "vir" / "adapters" / "datasus" / "references"
VIEWS_CACHE_DIR = DATA_DIR / "views"
VALIDATION_DIR = DATA_DIR / "validation"

RAW_DATA_FILE = RAW_DATA_DIR / "base_aesop_raw_extracao_01042026.parquet/*.parquet"

dirs = (
    DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    CLUSTERING_DIR,
    VIEWS_CACHE_DIR,
    VALIDATION_DIR,
)
for _directory in dirs:
    _directory.mkdir(parents=True, exist_ok=True)

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:1b")
HF_DEFAULT_HOME = Path(os.getenv("HF_HOME", "~/.cache/huggingface"))

# views
TEMPORAL_MODE = "chronological"  # "chronological" (full year_week series) | "annual" (52-point week_of_year profile)
TEMPORAL_NORMALIZATION = "minmax"  # "minmax" (row min→0, max→1) | "zscore" (mean=0, std=1)
DEFAULT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
MODEL_DIMENSIONS = 768

# filtering
INDICATIVE_CODES_FILE = REFERENCES_DIR / "indicative_codes_yes_no__2022_2025__90perc.csv"
INDICATIVE_ENABLED = True
MIN_WEEKS = 12
DEFAULT_SKIP_PATTERNS = [("cid10", r"^Z")]


def _parse_skip_patterns(raw: str) -> list[tuple]:
    """Parse VIROLA_SKIP_PATTERNS env var.

    Format: semicolon-separated entries, each as ``code_type|regex``.
    Use ``*`` as code_type to match all types (maps to None).

    Example: ``cid10|^Z;*|^ABP``
    """
    result = []
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        code_type, _, regex = entry.partition("|")
        result.append((None if code_type.strip() == "*" else code_type.strip(), regex))
    return result


_raw_skip = os.getenv("VIROLA_SKIP_PATTERNS")
SKIP_PATTERNS: list[tuple] = (
    _parse_skip_patterns(_raw_skip) if _raw_skip else DEFAULT_SKIP_PATTERNS
)


def get_indicative_label(enabled: bool = INDICATIVE_ENABLED) -> str:
    """Returns a short label indicating if indicative filtering is enabled."""
    return "indic" if enabled else "not-indic"


# snf config
AFFINITY_ZSCORE_PERMS = 1000
AFFINITY_ZSCORE_SEED = 42
