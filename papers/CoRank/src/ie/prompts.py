def build_ie_prompt(query: str, title: str, abstract: str) -> str:
    """
    Промпт для извлечения компактной информации о документе.
    """
    return f"""
You are given a scientific search query and a candidate paper.

Query:
{query}

Paper title:
{title}

Paper abstract:
{abstract}

Task:
1. Extract 3–5 key phrases that describe the main topics of the paper.
2. Write a very short summary (2–3 sentences) focusing on what problem the paper solves.

Return the result in JSON with keys:
- keyphrases (list of strings)
- summary (string)
"""
