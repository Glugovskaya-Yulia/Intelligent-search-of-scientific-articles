import requests
import time
from src.llm.base import LLMClient

class DeepSeekClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: int = 60,
        retries: int = 3,
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.url = "https://api.deepseek.com/v1/chat/completions"

    def generate(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        for attempt in range(self.retries):
            try:
                response = requests.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt == self.retries - 1:
                    raise e
                time.sleep(2)
