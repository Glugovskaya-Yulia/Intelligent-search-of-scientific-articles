import math

def recall_at_k(ranked_doc_ids, relevant_doc_ids, k):
    ranked_k = ranked_doc_ids[:k]
    relevant = set(relevant_doc_ids)
    if not relevant:
        return 0.0
    return len(set(ranked_k) & relevant) / len(relevant)


def dcg_at_k(ranked_doc_ids, relevant_doc_ids, k):
    relevant = set(relevant_doc_ids)
    dcg = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k]):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 2)
    return dcg


def ndcg_at_k(ranked_doc_ids, relevant_doc_ids, k):
    dcg = dcg_at_k(ranked_doc_ids, relevant_doc_ids, k)
    ideal = min(len(relevant_doc_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    if idcg == 0:
        return 0.0
    return dcg / idcg
