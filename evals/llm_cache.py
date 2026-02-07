"""Disk-backed LLM response cache for eval suite.

Caches LLM responses keyed by (model, temperature, max_tokens, system_prompt, user_content).
Cache is loaded into memory at startup and new entries are appended to disk on flush().
"""

import hashlib
import json
from pathlib import Path

from llm_client import LLMClient


class CachedLLMClient:
    """Wraps an LLMClient with disk-backed response caching.

    Drop-in replacement for LLMClient — the EmailClassifier doesn't need
    to know whether it's talking to a real LLM or a cached one.
    """

    def __init__(self, inner: LLMClient, cache_path: Path):
        self.inner = inner
        self.cache_path = cache_path
        self._cache: dict[str, str] = {}  # key -> response
        self._pending: list[dict] = []  # new entries to flush to disk
        self.hits = 0
        self.misses = 0
        self._load()

    def _load(self) -> None:
        """Load existing cache entries from JSONL into memory."""
        if not self.cache_path.exists():
            return
        with open(self.cache_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    self._cache[entry["key"]] = entry["response"]
                except (json.JSONDecodeError, KeyError):
                    continue  # skip corrupt entries

    def _cache_key(self, system_prompt: str, user_content: str) -> str:
        """Compute cache key from LLM parameters and prompt content."""
        raw = (
            f"{self.inner.model}|{self.inner.temperature}|{self.inner.max_tokens}"
            f"|{system_prompt}|{user_content}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    async def complete(self, system_prompt: str, user_content: str) -> str:
        """Return cached response on hit, otherwise call inner LLM and cache the result."""
        key = self._cache_key(system_prompt, user_content)

        if key in self._cache:
            self.hits += 1
            return self._cache[key]

        self.misses += 1
        response = await self.inner.complete(system_prompt, user_content)

        self._cache[key] = response
        self._pending.append({
            "key": key,
            "model": self.inner.model,
            "response": response,
        })
        return response

    async def is_available(self) -> bool:
        """Delegate to inner client."""
        return await self.inner.is_available()

    def flush(self) -> None:
        """Append pending cache entries to disk."""
        if not self._pending:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a") as f:
            for entry in self._pending:
                f.write(json.dumps(entry) + "\n")
        self._pending.clear()
