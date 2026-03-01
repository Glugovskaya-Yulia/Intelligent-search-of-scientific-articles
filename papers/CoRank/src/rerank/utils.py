import json
from collections import defaultdict

def group_by_query_id(ie_records: list):
    grouped = defaultdict(list)
    for rec in ie_records:
        grouped[rec["query_id"]].append(rec)
    return grouped

def parse_ranked_doc_ids(text: str):
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return []
