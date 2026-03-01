import json
from pathlib import Path
from collections import defaultdict

def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def check_non_empty(path: Path):
    count = sum(1 for _ in load_jsonl(path))
    return {
        "check": "non_empty_file",
        "num_lines": count,
        "ok": count > 0,
    }


def check_all_queries_present(coarse_path: Path, queries_path: Path):
    coarse_qids = set()
    for rec in load_jsonl(coarse_path):
        coarse_qids.add(rec["query_id"])

    query_qids = set()
    for rec in load_jsonl(queries_path):
        query_qids.add(rec["qid"])

    missing = sorted(query_qids - coarse_qids)

    return {
        "check": "all_queries_present",
        "num_queries": len(query_qids),
        "num_found": len(coarse_qids),
        "missing_query_ids": missing,
        "ok": len(missing) == 0,
    }


def check_non_empty_rankings(coarse_path: Path):
    empty = []
    for rec in load_jsonl(coarse_path):
        if not rec.get("ranked_doc_ids"):
            empty.append(rec["query_id"])

    return {
        "check": "non_empty_rankings",
        "num_empty": len(empty),
        "empty_query_ids": empty[:10],  # не раздуваем вывод
        "ok": len(empty) == 0,
    }


def check_doc_ids_exist(coarse_path: Path, corpus_path: Path):
    corpus_ids = set()
    for rec in load_jsonl(corpus_path):
        if "doc_id" in rec:
            corpus_ids.add(rec["doc_id"])
        elif "corpusid" in rec:
            corpus_ids.add(rec["corpusid"])

    missing = defaultdict(list)

    for rec in load_jsonl(coarse_path):
        qid = rec["query_id"]
        for doc_id in rec.get("ranked_doc_ids", []):
            if doc_id not in corpus_ids:
                missing[qid].append(doc_id)

    return {
        "check": "doc_ids_exist_in_corpus",
        "num_queries_with_missing": len(missing),
        "examples": dict(list(missing.items())[:3]),
        "ok": len(missing) == 0,
    }


def check_ranking_lengths(coarse_path: Path, max_k: int):
    too_long = []
    for rec in load_jsonl(coarse_path):
        if len(rec.get("ranked_doc_ids", [])) > max_k:
            too_long.append(rec["query_id"])

    return {
        "check": "ranking_length",
        "num_too_long": len(too_long),
        "too_long_query_ids": too_long[:10],
        "ok": len(too_long) == 0,
    }
