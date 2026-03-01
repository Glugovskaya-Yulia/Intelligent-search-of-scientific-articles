import json
from pathlib import Path
from typing import List, Dict

from src.config import PROCESSED_DATA_DIR
from src.data_loading.common import ensure_dir


def save_jsonl(
    data: List[Dict],
    path: Path,
) -> Path:
    """
    Сохраняет список словарей в JSONL.
    """
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return path


def save_litsearch_queries_jsonl(queries_norm: List[Dict]) -> Path:
    """
    Сохраняет нормализованные queries LitSearch в JSONL.
    """
    out_dir = ensure_dir(PROCESSED_DATA_DIR / "litsearch")
    out_path = out_dir / "queries.jsonl"
    return save_jsonl(queries_norm, out_path)


def save_litsearch_corpus_jsonl(corpus_norm: List[Dict]) -> Path:
    """
    Сохраняет нормализованные corpus LitSearch в JSONL.
    """
    out_dir = ensure_dir(PROCESSED_DATA_DIR / "litsearch")
    out_path = out_dir / "corpus.jsonl"
    return save_jsonl(corpus_norm, out_path)
