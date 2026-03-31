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
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
CLUSTERING_DIR = DATA_DIR / "clustering"
MODELS_DIR = PROJ_ROOT / "models"
REFERENCES_DIR = PROJ_ROOT / "references"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_FILE = RAW_DATA_DIR / "aesop_raw_dados_2017_20250223_20municipios.parquet"

VIEWS_CACHE_DIR = DATA_DIR / "views"
VALIDATION_DIR = DATA_DIR / "validation"

# FIXME create dirs if not exist

DEFAULT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
MODEL_DIMENSIONS = 768

EMBEDDING_COMPONENTS = 64
CHUNK_SIZE = 100_000  # Records per chunk - adjust based on available RAM
INDICATIVE_CODES_FILE = REFERENCES_DIR / "indicative_codes_yes_no_75perc.csv"
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
TEMPORAL_MODE = "chronological"  # "chronological" (full year_week series) | "annual" (52-point week_of_year profile)
TEMPORAL_NORMALIZATION = "minmax"  # "minmax" (row min→0, max→1) | "zscore" (mean=0, std=1)


def get_indicative_label(enabled: bool = INDICATIVE_ENABLED) -> str:
    """Returns a short label indicating if indicative filtering is enabled."""
    return "indic" if enabled else "not-indic"


# Memory recommendations by RAM:
# 8GB RAM: CHUNK_SIZE = 50_000
# 16GB RAM: CHUNK_SIZE = 100_000
# 32GB RAM: CHUNK_SIZE = 200_000
# 64GB RAM: CHUNK_SIZE = 500_000
