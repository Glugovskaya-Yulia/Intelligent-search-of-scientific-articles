from src.llm.deepseek import DeepSeekClient

def build_llm_client(
    provider: str,
    **kwargs,
):
    if provider == "deepseek":
        return DeepSeekClient(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
