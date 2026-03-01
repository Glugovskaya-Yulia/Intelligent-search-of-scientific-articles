import json

def parse_json_safe(text: str) -> dict:
    """
    Пытаемся аккуратно распарсить JSON из ответа LLM.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}
