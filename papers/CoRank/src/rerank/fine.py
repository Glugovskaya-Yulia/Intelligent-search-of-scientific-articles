import json
from pathlib import Path
from typing import Callable

from src.rerank.prompts import build_fine_rerank_prompt
from src.rerank.utils import parse_ranked_doc_ids


def run_fine_reranking(
    coarse_path: Path,
    queries_path: Path,
    corpus_path: Path,
    output_path: Path,
    llm_call_fn: Callable[[str], str],
    max_docs: int = 20,
):
    # queries
    queries = {}
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["qid"]] = q["text"]

    # corpus
    corpus = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc.get("doc_id") or doc.get("corpusid")
            corpus[doc_id] = doc

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(coarse_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            item = json.loads(line)
            qid = item["query_id"]
            query = queries.get(qid, "")

            doc_ids = item["ranked_doc_ids"][:max_docs]

            documents = []
            for doc_id in doc_ids:
                if doc_id not in corpus:
                    continue
                doc = corpus[doc_id]
                text = (
                    doc.get("full_paper")
                    or doc.get("abstract")
                    or ""
                )
                documents.append(
                    {
                        "doc_id": doc_id,
                        "title": doc.get("title", ""),
                        "text": text,
                    }
                )

            prompt = build_fine_rerank_prompt(query, documents)
            response = llm_call_fn(prompt)

            ranked_doc_ids = parse_ranked_doc_ids(response)

            fout.write(
                json.dumps(
                    {
                        "query_id": qid,
                        "final_ranked_doc_ids": ranked_doc_ids,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
