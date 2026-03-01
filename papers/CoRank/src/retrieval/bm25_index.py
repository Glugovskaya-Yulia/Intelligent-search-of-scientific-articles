import re
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from joblib import dump, load
from rank_bm25 import BM25Okapi
from src.data_loading.common import ensure_dir


TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)

def simple_tokenize(text: str) -> List[str]:
    if not text:
        return []
    return TOKEN_PATTERN.findall(text.lower())

def build_bm25_index_from_corpus(
    corpus_jsonl_path: Path,
    index_out_path: Path,
    use_full_text: bool = False,
    text_fields: Tuple[str, ...] = ("title", "abstract"),
):
    """
    Считывает corpus.jsonl и строит BM25 индекс.
    По умолчанию индексируется title + abstract.
    Если use_full_text=True, индексируем full_text (дольше и тяжелее).
    Сохраняет индекс и метаданные в index_out_path (joblib dump).
    """
    index_out_path = ensure_dir(index_out_path.parent) / index_out_path.name

    doc_ids = []
    docs_raw = []         # для хранения исходного текста (title+abstract или full_text)
    tokenized_docs = []

    with open(corpus_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            doc_id = rec.get("doc_id") or rec.get("corpusid")
            doc_ids.append(doc_id)

            if use_full_text:
                text = rec.get("full_text") or rec.get("full_paper") or ""
            else:
                parts = []
                for fld in text_fields:
                    v = rec.get(fld)
                    if v:
                        parts.append(v)
                text = " ".join(parts)

            docs_raw.append({"doc_id": doc_id, "text": text, "title": rec.get("title"), "abstract": rec.get("abstract")})
            tokenized = simple_tokenize(text)
            tokenized_docs.append(tokenized)

    bm25 = BM25Okapi(tokenized_docs)
    payload = {
        "bm25": bm25,
        "doc_ids": doc_ids,
        "docs_raw": docs_raw,
        "tokenized_docs": tokenized_docs,
        "use_full_text": use_full_text,
        "text_fields": text_fields,
    }
    
    return payload



def save_bm25_index(payload: Dict[str, Any], out_dir: Path, base_name: str = "bm25_index"):
    """
    payload должна содержать, как минимум:
      - 'bm25' : объект BM25Okapi
      - 'tokenized_docs' : List[List[str]]  (опционально)
      - 'doc_ids' : List[doc_id]
      - 'text_fields' : tuple или list description
    Сохраняет:
      - out_dir/base_name + ".joblib"
      - out_dir/base_name + "_meta.json"
    """
    out_dir = ensure_dir(out_dir)
    joblib_path = out_dir / f"{base_name}.joblib"
    meta_path = out_dir / f"{base_name}_meta.json"

    # Сохраняем тяжелый объект (joblib)
    dump(payload.get("bm25_payload", payload), joblib_path)  # payload или payload['bm25_payload']

    # Формируем лёгкую мета-инфу
    meta = {
        "doc_ids": payload.get("doc_ids"),
        "num_docs": len(payload.get("doc_ids", [])),
        "use_full_text": payload.get("use_full_text", False),
        "text_fields": payload.get("text_fields", []),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return joblib_path, meta_path

def load_bm25_index(index_path: Path, meta_path: Path = None):
    """
    Возвращает (payload, meta_dict)
    payload — то, что сохранили (например словарь с 'bm25','doc_ids',...)
    meta_dict — содержимое meta_path (если есть)
    """
    payload = load(index_path)
    meta = None
    if meta_path and meta_path.exists():
        import json
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return payload, meta
