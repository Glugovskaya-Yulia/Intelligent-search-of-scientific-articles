import json
from pathlib import Path
from ranx import Qrels

def build_qrels_from_queries(queries_path: Path) -> Qrels:
    qrels = {}

    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)

            qid = str(q["qid"])
            rel_docs = q["relevant_doc_ids"]

            qrels[qid] = {str(doc_id): 1 for doc_id in rel_docs}

    return Qrels(qrels)
