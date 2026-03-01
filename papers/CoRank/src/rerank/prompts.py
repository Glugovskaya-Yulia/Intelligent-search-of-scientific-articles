def build_coarse_rerank_prompt(query: str, documents: list) -> str:
    """
    documents: list of dicts with keys:
      - doc_id
      - title
      - keyphrases
      - summary
    """
    docs_text = []
    for i, doc in enumerate(documents, 1):
        block = f"""
Document {i}:
Title: {doc['title']}
Keyphrases: {', '.join(doc['keyphrases'])}
Summary: {doc['summary']}
"""
        docs_text.append(block)

    docs_text = "\n".join(docs_text)

    return f"""
You are given a search query and a list of candidate papers.

Query:
{query}

Candidate papers:
{docs_text}

Task:
Rank the papers by relevance to the query from most relevant to least relevant.

Return ONLY a JSON list of doc_id values in ranked order.
"""

def build_fine_rerank_prompt(query: str, documents: list) -> str:
    """
    documents: list of dicts with keys:
      - doc_id
      - title
      - text
    """
    docs_text = []
    for i, doc in enumerate(documents, 1):
        block = f"""
Document {i}:
Title: {doc['title']}
Full text:
{doc['text']}
"""
        docs_text.append(block)

    docs_text = "\n".join(docs_text)

    return f"""
You are given a search query and a small set of candidate research papers.

Query:
{query}

Candidate papers:
{docs_text}

Task:
Carefully read the documents and rank them by relevance to the query.
Focus on actual content, not just keywords.

Return ONLY a JSON list of doc_id values in ranked order.
"""
