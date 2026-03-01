from typing import Dict
from src.ie.prompts import build_ie_prompt
from src.ie.utils import parse_json_safe


# def extract_ie_for_document(
#     query: str,
#     doc: Dict,
#     llm_call_fn,
# ) -> Dict:
#     """
#     Извлекает компактное описание одного документа.

#     llm_call_fn(prompt: str) -> str
#     """
#     prompt = build_ie_prompt(
#         query=query,
#         title=doc["title"],
#         abstract=doc["abstract"],
#     )

#     response_text = llm_call_fn(prompt)
#     parsed = parse_json_safe(response_text)

#     return {
#         "doc_id": doc["doc_id"],
#         "title": doc["title"],
#         "keyphrases": parsed.get("keyphrases", []),
#         "summary": parsed.get("summary", ""),
#     }

def extract_ie_for_document(query: str, doc: Dict, llm_call_fn):
    prompt = build_ie_prompt(query=query, title=doc.get("title",""), abstract=doc.get("abstract",""))
    response_text = llm_call_fn(prompt)
    parsed = parse_json_safe(response_text)
    # fallback: если parse failed, можно запустить simple extractor (optional)
    return {
        "doc_id": doc["doc_id"],
        "title": doc.get("title",""),
        "keyphrases": parsed.get("keyphrases", []),
        "summary": parsed.get("summary", ""),
    }
