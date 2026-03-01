import json
from pathlib import Path
from typing import List, Dict, Union
from joblib import load
from tqdm import tqdm
import numpy as np

# Импортируем простую токенизацию из bm25_index (или определите свою)
from src.retrieval.bm25_index import simple_tokenize


def retrieve_topk_for_query_from_payload(
    bm25,
    doc_ids: List[Union[int, str]],
    docs_raw: List[Dict],
    query: str,
    top_k: int = 200,
) -> List[Dict]:
    """
    Быстрый извлечь top_k кандидатов, используя уже загруженный bm25 объект и метаданные.

    Возвращает список словарей: {"doc_id", "score", "title", "abstract"}.
    """
    q_tokens = simple_tokenize(query)
    if not q_tokens:
        return []

    # Получаем скор для всех документов (numpy array)
    scores = bm25.get_scores(q_tokens)  # shape = (N,)

    N = scores.shape[0]
    if top_k <= 0:
        return []

    # Быстрый top-k: argpartition -> частичная сортировка
    if top_k >= N:
        # мало документов — делаем полную сортировку
        topk_idx = np.argsort(scores)[::-1]
    else:
        # argpartition возвращает неотсортированный набор индексов top_k
        part_idx = np.argpartition(scores, -top_k)[-top_k:]
        # затем сортируем эти top_k по убыванию скорa
        topk_idx = part_idx[np.argsort(scores[part_idx])[::-1]]

    results = []
    for idx in topk_idx:
        idx = int(idx)
        results.append({
            "doc_id": doc_ids[idx],
            "score": float(scores[idx]),
            "title": docs_raw[idx].get("title"),
            "abstract": docs_raw[idx].get("abstract"),
        })
    return results


def generate_candidates_for_all_queries(
    bm25_payload_path: Union[str, Path],
    queries_jsonl_path: Union[str, Path],
    out_jsonl_path: Union[str, Path],
    top_k: int = 200,
):
    """
    Генерирует candidates.jsonl для всех запросов.

    Улучшение по сравнению с naive-версией:
    - Загружает bm25 payload один раз (joblib.load).
    - Внутри цикла по запросам использует уже загруженный объект.
    - Для выбора top-k использует numpy.argpartition.

    Формат выходных строк:
    {"query_id": ..., "query": "...", "candidates": [{"doc_id": ..., "score": ...}, ...]}
    """
    bm25_payload_path = Path(bm25_payload_path)
    queries_jsonl_path = Path(queries_jsonl_path)
    out_jsonl_path = Path(out_jsonl_path)
    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Load payload ONE TIME
    print(f"Loading BM25 payload from {bm25_payload_path} ...")
    payload = load(bm25_payload_path)
    print("BM25 payload loaded.")

    # Expected keys in payload: 'bm25', 'doc_ids', 'docs_raw'
    if "bm25" not in payload or "doc_ids" not in payload or "docs_raw" not in payload:
        raise ValueError("Payload must contain keys: 'bm25', 'doc_ids', 'docs_raw'")

    bm25 = payload["bm25"]
    doc_ids = payload["doc_ids"]
    docs_raw = payload["docs_raw"]

    # Iterate over queries and retrieve candidates
    with open(queries_jsonl_path, "r", encoding="utf-8") as qf, \
         open(out_jsonl_path, "w", encoding="utf-8") as outf:
        for line in tqdm(qf, desc="queries"):
            qrec = json.loads(line)
            query_id = qrec.get("qid")
            # поддерживаем оба ключа: 'text' или 'query'
            text = qrec.get("text") or qrec.get("query") or ""
            candidates = retrieve_topk_for_query_from_payload(bm25, doc_ids, docs_raw, text, top_k=top_k)

            out_line = {
                "query_id": query_id,
                "query": text,
                "candidates": [{"doc_id": c["doc_id"], "score": c["score"]} for c in candidates]
            }
            outf.write(json.dumps(out_line, ensure_ascii=False) + "\n")

    print(f"Saved candidates to {out_jsonl_path}")
    return out_jsonl_path
