from pathlib import Path

# корень проекта
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
# LLM cache
CACHE_DIR = DATA_DIR / "cache" / "llm"
