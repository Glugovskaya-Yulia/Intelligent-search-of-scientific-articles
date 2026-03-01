import json
from pathlib import Path
from typing import Callable, Optional
from src.ie.extractor import extract_ie_for_document


def _first_present(d: dict, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def run_ie_on_candidates(
    candidates_path: Path,
    corpus_path: Path,
    output_path: Path,
    llm_call_fn: Callable[[str], str],
    max_queries: Optional[int] = None,   # ← НОВОЕ
):
    # Загружаем корпус в словарь
    corpus = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "doc_id" in rec:
                key = rec["doc_id"]
            elif "corpusid" in rec:
                key = rec["corpusid"]
            else:
                continue
            corpus[key] = rec

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    processed_queries = 0  # ← счётчик запросов

    with open(candidates_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            # ограничение по числу запросов
            if max_queries is not None and processed_queries >= max_queries:
                break

            item = json.loads(line)

            qid = _first_present(item, ("query_id", "qid", "id"))
            query = _first_present(item, ("query", "text")) or ""

            for cand in item.get("candidates", []):
                doc_id = _first_present(cand, ("doc_id", "corpusid", "id"))
                if doc_id is None or doc_id not in corpus:
                    continue

                doc = corpus[doc_id]
                doc_title = doc.get("title", "")
                doc_abstract = (
                    doc.get("abstract", "")
                    or doc.get("full_text", "")
                    or doc.get("full_paper", "")
                )

                ie_result = extract_ie_for_document(
                    query=query,
                    doc={
                        "doc_id": doc_id,
                        "title": doc_title,
                        "abstract": doc_abstract,
                    },
                    llm_call_fn=llm_call_fn,
                )

                ie_result["query_id"] = qid
                fout.write(json.dumps(ie_result, ensure_ascii=False) + "\n")

            processed_queries += 1  # ← увеличиваем ТОЛЬКО на уровне query
