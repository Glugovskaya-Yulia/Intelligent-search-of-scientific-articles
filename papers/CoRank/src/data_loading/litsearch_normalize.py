from typing import List, Dict
from datasets import Dataset


def normalize_litsearch_queries(dataset: Dataset) -> List[Dict]:
    """
    Приводит LitSearch queries к unified-формату
    """
    normalized = []

    for idx, row in enumerate(dataset):
        normalized.append({
            "qid": idx,
            "text": row["query"],
            "relevant_doc_ids": row["corpusids"],
        })

    return normalized


def normalize_litsearch_corpus(dataset: Dataset) -> List[Dict]:
    """
    Приводит LitSearch corpus к unified-формату
    """
    normalized = []

    for row in dataset:
        normalized.append({
            "doc_id": row["corpusid"],
            "title": row["title"],
            "abstract": row["abstract"],
            "full_text": row["full_paper"],
        })

    return normalized
