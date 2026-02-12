"""Disk-backed LLM response cache for eval suite.

Caches LLM responses keyed by (model, temperature, max_tokens, extra_body, system_prompt, user_content).
Cache is loaded into memory at startup and new entries are appended to disk on flush().
"""

import asyncio
import hashlib
import json
import time
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
        self._cache: dict[str, tuple[str, str]] = {}  # key -> (response, thinking)
        self._pending: list[dict] = []  # new entries to flush to disk
        self.hits = 0
        self.misses = 0
        self._llm_seconds = 0.0  # accumulated LLM call time (misses only)
        self._thinking_buffers: dict[int, list[str]] = {}  # per-task thinking buffers
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
                    response = entry["response"]
                    thinking = entry.get("thinking", "")
                    # Normalize old entries that may contain unstripped think tags
                    extra_thinking = LLMClient._extract_thinking(response)
                    response = LLMClient._strip_thinking(response)
                    if extra_thinking and not thinking:
                        thinking = extra_thinking
                    self._cache[entry["key"]] = (response, thinking)
                except (json.JSONDecodeError, KeyError):
                    continue  # skip corrupt entries

    def _cache_key(self, system_prompt: str, user_content: str) -> str:
        """Compute cache key from LLM parameters and prompt content."""
        raw = json.dumps(
            [self.inner.model, self.inner.temperature, self.inner.max_tokens,
             self.inner.extra_body, system_prompt, user_content],
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def _task_key(self) -> int:
        """Return a key for the current asyncio task (0 when not inside a task)."""
        task = asyncio.current_task()
        return id(task) if task else 0

    async def complete(
        self, system_prompt: str, user_content: str, include_thinking: bool = False,
    ) -> str | tuple[str, str]:
        """Return cached response on hit, otherwise call inner LLM and cache the result."""
        key = self._cache_key(system_prompt, user_content)
        tk = self._task_key()

        if key in self._cache:
            self.hits += 1
            response, thinking = self._cache[key]
            if thinking:
                self._thinking_buffers.setdefault(tk, []).append(thinking)
            if include_thinking:
                return response, thinking
            return response

        self.misses += 1
        start = time.monotonic()
        response, thinking = await self.inner.complete(
            system_prompt, user_content, include_thinking=True,
        )
        self._llm_seconds += time.monotonic() - start

        if thinking:
            self._thinking_buffers.setdefault(tk, []).append(thinking)

        self._cache[key] = (response, thinking)
        self._pending.append({
            "key": key,
            "model": self.inner.model,
            "response": response,
            "thinking": thinking,
        })
        if include_thinking:
            return response, thinking
        return response

    def take_thinking(self) -> str:
        """Return accumulated thinking content for the current task and reset its buffer.

        Uses asyncio.current_task() to isolate thinking per concurrent task,
        preventing cross-contamination when parallelism > 1.
        """
        buf = self._thinking_buffers.pop(self._task_key(), [])
        return "\n\n".join(t for t in buf if t)

    def take_llm_seconds(self) -> float:
        """Return accumulated LLM call time and reset to zero."""
        elapsed = self._llm_seconds
        self._llm_seconds = 0.0
        return elapsed

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
