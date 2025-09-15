from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

logger.remove(0)
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.getenv("DATA_DIR")) or PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
CLUSTERING_DIR = DATA_DIR / "clustering"
MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_FILE = RAW_DATA_DIR / "aesop_raw_dados_2017_20250223_20municipios.parquet"

# FIXME create dirs if not exist

EMBEDDING_COMPONENTS = 64
CHUNK_SIZE = 100_000  # Records per chunk - adjust based on available RAM

# Memory recommendations by RAM:
# 8GB RAM: CHUNK_SIZE = 50_000
# 16GB RAM: CHUNK_SIZE = 100_000
# 32GB RAM: CHUNK_SIZE = 200_000
# 64GB RAM: CHUNK_SIZE = 500_000
