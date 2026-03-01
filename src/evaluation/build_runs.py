import json
from pathlib import Path
from ranx import Run

def build_run_from_candidates(candidates_path: Path) -> Run:
    run = {}

    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = str(item["query_id"])
            run[qid] = {}

            for rank, cand in enumerate(item["candidates"], start=1):
                run[qid][str(cand["doc_id"])] = 1.0 / rank

    return Run(run)


def build_run_from_rerank(path: Path, key: str) -> Run:
    run = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = str(item["query_id"])
            doc_ids = item.get(key, [])
            
            if not doc_ids:
                continue  # ← КРИТИЧЕСКИ ВАЖНО
            
            run[qid] = {}
            for rank, doc_id in enumerate(doc_ids, start=1):
                run[qid][str(doc_id)] = 1.0 / rank


    return Run(run)
