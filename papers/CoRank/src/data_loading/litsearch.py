from datasets import load_dataset
from pathlib import Path

from src.config import RAW_DATA_DIR
from src.data_loading.common import ensure_dir


def load_litsearch_queries():
    """
    Загружает queries LitSearch
    """
    cache_dir = ensure_dir(RAW_DATA_DIR / "litsearch")

    return load_dataset(
        "princeton-nlp/LitSearch",
        name="query",
        cache_dir=str(cache_dir),
    )


def load_litsearch_corpus(clean: bool = True):
    """
    Загружает корпус LitSearch

    clean=True  -> corpus_clean
    clean=False -> corpus_s2orc
    """
    cache_dir = ensure_dir(RAW_DATA_DIR / "litsearch")

    config_name = "corpus_clean" if clean else "corpus_s2orc"

    return load_dataset(
        "princeton-nlp/LitSearch",
        name=config_name,
        cache_dir=str(cache_dir),
    )
