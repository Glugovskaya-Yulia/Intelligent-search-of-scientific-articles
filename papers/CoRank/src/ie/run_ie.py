import json
from pathlib import Path

def run_ie(
    candidates_path: Path,
    corpus_lookup: dict,
    output_path: Path,
    llm_call_fn
):
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            doc_id = item["doc_id"]

            document_text = corpus_lookup[doc_id]

            extracted = extract_document_representation(
                document_text=document_text,
                llm_call_fn=llm_call_fn
            )

            out = {
                "query_id": item["query_id"],
                "doc_id": doc_id,
                "extracted": extracted
            }

            with open(output_path, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
