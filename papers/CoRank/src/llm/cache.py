import json
import hashlib
import time
from pathlib import Path
from typing import Any, Callable, Optional
import tempfile
from src.config import CACHE_DIR

# default cache TTL (seconds). None = infinite
DEFAULT_TTL = None

def _make_key(prompt: str, meta: dict) -> str:
    """
    Формируем ключ из prompt + meta (model, temp, max_tokens, task и т.п.).
    Всегда используем json.dumps(sort_keys=True) для детерминизма.
    """
    meta_json = json.dumps(meta or {}, ensure_ascii=False, sort_keys=True)
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8"))
    h.update(b"\n")
    h.update(meta_json.encode("utf-8"))
    return h.hexdigest()


class LLMCache:
    def __init__(self, cache_dir: Optional[Path] = None, ttl: Optional[int] = DEFAULT_TTL):
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl

    def _path_for_key(self, key: str) -> Path:
        # Разбиваем папки по префиксу для избежания большого числа файлов в одной папке
        return self.cache_dir / key[0:2] / f"{key}.json"

    def _read_file(self, path: Path) -> Optional[dict]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def get(self, prompt: str, meta: dict = None) -> Optional[str]:
        key = _make_key(prompt, meta or {})
        p = self._path_for_key(key)
        if not p.exists():
            return None
        rec = self._read_file(p)
        if not rec:
            return None
        # TTL check
        created = rec.get("_created_at", 0)
        if self.ttl is not None and (time.time() - created) > self.ttl:
            try:
                p.unlink()
            except Exception:
                pass
            return None
        return rec.get("response")

    def set(self, prompt: str, response: str, meta: dict = None):
        key = _make_key(prompt, meta or {})
        p = self._path_for_key(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tempf = None
        payload = {
            "_created_at": time.time(),
            "meta": meta or {},
            "response": response,
        }
        # atomic write: write to tmp then rename
        p.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            encoding="utf-8",
            dir=str(p.parent),  # ← КРИТИЧЕСКИ ВАЖНО
        ) as tmp:
            tempf = tmp.name
            json.dump(payload, tmp, ensure_ascii=False)
        
        Path(tempf).replace(p)


    def get_or_call(self, prompt: str, call_fn: Callable[[str], str], meta: dict = None) -> str:
        """
        Возвращает кэшированный ответ, либо вызывает call_fn(prompt),
        сохраняет и возвращает ответ.
        """
        cached = self.get(prompt, meta=meta)
        if cached is not None:
            return cached

        # not cached -> call
        resp = call_fn(prompt)
        # guard: ensure resp is str
        if resp is None:
            resp = ""
        self.set(prompt, resp, meta=meta)
        return resp

    def cleanup(self):
        """Проход по cached files и удаление устаревших (TTL)."""
        for sub in self.cache_dir.iterdir():
            if not sub.is_dir():
                continue
            for f in sub.iterdir():
                try:
                    rec = self._read_file(f)
                    if not rec:
                        f.unlink()
                        continue
                    created = rec.get("_created_at", 0)
                    if self.ttl is not None and (time.time() - created) > self.ttl:
                        f.unlink()
                except Exception:
                    continue
