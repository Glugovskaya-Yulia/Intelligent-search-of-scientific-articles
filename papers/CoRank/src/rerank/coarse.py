import json
from pathlib import Path
from typing import Callable

from src.rerank.prompts import build_coarse_rerank_prompt
from src.rerank.utils import group_by_query_id, parse_ranked_doc_ids


def run_coarse_reranking(
    ie_path: Path,
    queries_path: Path,
    output_path: Path,
    llm_call_fn: Callable[[str], str],
    top_k: int = 20,
    max_queries=None
):
    

    # загрузка queries
    queries = {}
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["qid"]] = q["text"]

    # загрузка IE
    ie_records = []
    with open(ie_path, "r", encoding="utf-8") as f:
        for line in f:
            ie_records.append(json.loads(line))

    grouped = group_by_query_id(ie_records)

    for i, (qid, docs) in enumerate(grouped.items()):
        if max_queries is not None and i >= max_queries:
            break

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fout:
        for qid, docs in grouped.items():
            query = queries.get(qid, "")

            prompt = build_coarse_rerank_prompt(query, docs)
            response = llm_call_fn(prompt)

            ranked_doc_ids = parse_ranked_doc_ids(response)
            ranked_doc_ids = ranked_doc_ids[:top_k]

            fout.write(
                json.dumps(
                    {
                        "query_id": qid,
                        "ranked_doc_ids": ranked_doc_ids,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
