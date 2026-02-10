"""Tests for evals.llm_cache — CachedLLMClient wrapper."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from evals.llm_cache import CachedLLMClient
from llm_client import LLMClient


def _make_inner(**overrides) -> LLMClient:
    """Create a mock LLMClient with typical attributes."""
    inner = AsyncMock(spec=LLMClient)
    inner.model = overrides.get("model", "test-model")
    inner.temperature = overrides.get("temperature", 0.2)
    inner.max_tokens = overrides.get("max_tokens", 1000)
    inner.extra_body = overrides.get("extra_body", {})
    return inner


def _set_return(inner, response, thinking=""):
    """Configure mock to return (response, thinking) tuple for include_thinking=True."""
    inner.complete.return_value = (response, thinking)


def _set_returns(inner, pairs):
    """Configure mock to return sequence of (response, thinking) tuples."""
    inner.complete.side_effect = pairs


class TestCacheHitMiss:
    async def test_miss_calls_inner(self, tmp_path: Path):
        inner = _make_inner()
        _set_return(inner, "PERSON")
        cached = CachedLLMClient(inner, tmp_path / "cache.jsonl")

        result = await cached.complete("system", "user")

        assert result == "PERSON"
        inner.complete.assert_awaited_once_with("system", "user", include_thinking=True)
        assert cached.hits == 0
        assert cached.misses == 1

    async def test_hit_skips_inner(self, tmp_path: Path):
        inner = _make_inner()
        _set_return(inner, "PERSON")
        cached = CachedLLMClient(inner, tmp_path / "cache.jsonl")

        await cached.complete("system", "user")  # miss
        result = await cached.complete("system", "user")  # hit

        assert result == "PERSON"
        assert inner.complete.await_count == 1
        assert cached.hits == 1
        assert cached.misses == 1

    async def test_different_prompts_are_different_keys(self, tmp_path: Path):
        inner = _make_inner()
        _set_returns(inner, [("PERSON", ""), ("SERVICE", "")])
        cached = CachedLLMClient(inner, tmp_path / "cache.jsonl")

        r1 = await cached.complete("system", "user-A")
        r2 = await cached.complete("system", "user-B")

        assert r1 == "PERSON"
        assert r2 == "SERVICE"
        assert inner.complete.await_count == 2
        assert cached.misses == 2

    async def test_pipe_in_prompt_does_not_collide(self, tmp_path: Path):
        """Prompts that differ only by field boundary are distinct cache keys."""
        inner = _make_inner()
        _set_returns(inner, [("PERSON", ""), ("SERVICE", "")])
        cached = CachedLLMClient(inner, tmp_path / "cache.jsonl")

        r1 = await cached.complete("sys|A", "B")
        r2 = await cached.complete("sys", "A|B")

        assert r1 == "PERSON"
        assert r2 == "SERVICE"
        assert cached.misses == 2

    async def test_different_extra_body_are_different_keys(self, tmp_path: Path):
        """Toggling extra_body (e.g. enable_thinking) produces a cache miss."""
        cache_path = tmp_path / "cache.jsonl"

        thinking_on = _make_inner(extra_body={})
        _set_return(thinking_on, "NEEDS_RESPONSE")
        cached_on = CachedLLMClient(thinking_on, cache_path)
        await cached_on.complete("sys", "usr")
        cached_on.flush()

        thinking_off = _make_inner(extra_body={"enable_thinking": False})
        _set_return(thinking_off, "FYI")
        cached_off = CachedLLMClient(thinking_off, cache_path)
        result = await cached_off.complete("sys", "usr")

        assert result == "FYI"
        assert cached_off.misses == 1  # cache miss, not a hit from thinking_on


class TestFlushAndReload:
    async def test_flush_writes_jsonl(self, tmp_path: Path):
        inner = _make_inner()
        _set_return(inner, "FYI")
        cache_path = tmp_path / "cache.jsonl"
        cached = CachedLLMClient(inner, cache_path)

        await cached.complete("sys", "usr")
        cached.flush()

        lines = cache_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["model"] == "test-model"
        assert entry["response"] == "FYI"
        assert "key" in entry

    async def test_flush_writes_thinking_field(self, tmp_path: Path):
        inner = _make_inner()
        _set_return(inner, "PERSON", "The sender is a real human")
        cache_path = tmp_path / "cache.jsonl"
        cached = CachedLLMClient(inner, cache_path)

        await cached.complete("sys", "usr")
        cached.flush()

        entry = json.loads(cache_path.read_text().strip())
        assert entry["thinking"] == "The sender is a real human"

    async def test_reload_serves_from_disk(self, tmp_path: Path):
        cache_path = tmp_path / "cache.jsonl"

        # First instance: populate cache
        inner1 = _make_inner()
        _set_return(inner1, "LOW_PRIORITY")
        cached1 = CachedLLMClient(inner1, cache_path)
        await cached1.complete("sys", "usr")
        cached1.flush()

        # Second instance: should load from disk
        inner2 = _make_inner()
        cached2 = CachedLLMClient(inner2, cache_path)
        result = await cached2.complete("sys", "usr")

        assert result == "LOW_PRIORITY"
        inner2.complete.assert_not_awaited()
        assert cached2.hits == 1

    async def test_no_flush_without_pending(self, tmp_path: Path):
        cache_path = tmp_path / "cache.jsonl"
        cached = CachedLLMClient(_make_inner(), cache_path)
        cached.flush()
        assert not cache_path.exists()


class TestSharedCacheFile:
    async def test_two_clients_share_cache_file(self, tmp_path: Path):
        """Two CachedLLMClients with different models sharing one file."""
        cache_path = tmp_path / "cache.jsonl"

        inner_cloud = _make_inner()
        inner_cloud.model = "cloud-model"
        _set_return(inner_cloud, "SERVICE")

        inner_local = _make_inner()
        inner_local.model = "local-model"
        _set_return(inner_local, "NEEDS_RESPONSE")

        cloud = CachedLLMClient(inner_cloud, cache_path)
        local = CachedLLMClient(inner_local, cache_path)

        # Same prompt, different models → different keys
        r1 = await cloud.complete("sys", "usr")
        r2 = await local.complete("sys", "usr")

        assert r1 == "SERVICE"
        assert r2 == "NEEDS_RESPONSE"
        assert cloud.misses == 1
        assert local.misses == 1

        cloud.flush()
        local.flush()

        lines = cache_path.read_text().strip().split("\n")
        assert len(lines) == 2


class TestCorruptCacheFile:
    async def test_skips_bad_lines(self, tmp_path: Path):
        cache_path = tmp_path / "cache.jsonl"
        # Write a corrupt line followed by a valid one
        inner = _make_inner()
        valid_key = CachedLLMClient(inner, cache_path)._cache_key("sys", "usr")
        cache_path.write_text(
            "not valid json\n"
            + json.dumps({"key": valid_key, "response": "CACHED"}) + "\n"
        )

        cached = CachedLLMClient(inner, cache_path)
        result = await cached.complete("sys", "usr")

        assert result == "CACHED"
        assert cached.hits == 1
        inner.complete.assert_not_awaited()

    async def test_old_cache_without_thinking_field(self, tmp_path: Path):
        """Cache entries from before thinking support still work."""
        cache_path = tmp_path / "cache.jsonl"
        inner = _make_inner()
        valid_key = CachedLLMClient(inner, cache_path)._cache_key("sys", "usr")
        # Old format: no "thinking" field
        cache_path.write_text(
            json.dumps({"key": valid_key, "model": "test-model", "response": "SERVICE"}) + "\n"
        )

        cached = CachedLLMClient(inner, cache_path)
        result = await cached.complete("sys", "usr")

        assert result == "SERVICE"
        assert cached.hits == 1
        assert cached.take_thinking() == ""  # No thinking in old entries


class TestErrorNotCached:
    async def test_llm_error_is_not_cached(self, tmp_path: Path):
        inner = _make_inner()
        inner.complete.side_effect = RuntimeError("LLM failed")
        cached = CachedLLMClient(inner, tmp_path / "cache.jsonl")

        with pytest.raises(RuntimeError):
            await cached.complete("sys", "usr")

        assert cached.misses == 1
        assert len(cached._pending) == 0
        cached.flush()
        assert not (tmp_path / "cache.jsonl").exists()


class TestLlmSeconds:
    async def test_miss_accumulates_time(self, tmp_path: Path):
        inner = _make_inner()
        _set_return(inner, "PERSON")
        cached = CachedLLMClient(inner, tmp_path / "cache.jsonl")

        await cached.complete("sys", "usr")

        assert cached._llm_seconds > 0
        elapsed = cached.take_llm_seconds()
        assert elapsed > 0
        assert cached._llm_seconds == 0.0  # reset after take

    async def test_hit_does_not_accumulate_time(self, tmp_path: Path):
        inner = _make_inner()
        _set_return(inner, "PERSON")
        cached = CachedLLMClient(inner, tmp_path / "cache.jsonl")

        await cached.complete("sys", "usr")
        cached.take_llm_seconds()  # drain

        await cached.complete("sys", "usr")  # cache hit

        assert cached.take_llm_seconds() == 0.0


class TestThinking:
    async def test_thinking_captured_on_miss(self, tmp_path: Path):
        inner = _make_inner()
        _set_return(inner, "PERSON", "The sender looks like a real person")
        cached = CachedLLMClient(inner, tmp_path / "cache.jsonl")

        await cached.complete("sys", "usr")
        thinking = cached.take_thinking()

        assert thinking == "The sender looks like a real person"

    async def test_thinking_captured_on_hit(self, tmp_path: Path):
        inner = _make_inner()
        _set_return(inner, "PERSON", "The sender looks like a real person")
        cached = CachedLLMClient(inner, tmp_path / "cache.jsonl")

        await cached.complete("sys", "usr")  # miss
        cached.take_thinking()  # drain

        await cached.complete("sys", "usr")  # hit
        thinking = cached.take_thinking()

        assert thinking == "The sender looks like a real person"

    async def test_take_thinking_resets(self, tmp_path: Path):
        inner = _make_inner()
        _set_return(inner, "SERVICE", "Automated notification")
        cached = CachedLLMClient(inner, tmp_path / "cache.jsonl")

        await cached.complete("sys", "usr")
        cached.take_thinking()
        assert cached.take_thinking() == ""

    async def test_multiple_calls_accumulate_thinking(self, tmp_path: Path):
        inner = _make_inner()
        _set_returns(inner, [
            ("PERSON", "thought about sender"),
            ("NEEDS_RESPONSE", "thought about label"),
        ])
        cached = CachedLLMClient(inner, tmp_path / "cache.jsonl")

        await cached.complete("sys", "usr-A")
        await cached.complete("sys", "usr-B")
        thinking = cached.take_thinking()

        assert "thought about sender" in thinking
        assert "thought about label" in thinking

    async def test_empty_thinking_not_buffered(self, tmp_path: Path):
        inner = _make_inner()
        _set_return(inner, "SERVICE", "")
        cached = CachedLLMClient(inner, tmp_path / "cache.jsonl")

        await cached.complete("sys", "usr")
        assert cached.take_thinking() == ""

    async def test_thinking_survives_reload(self, tmp_path: Path):
        """Thinking persisted to disk and available after cache reload."""
        cache_path = tmp_path / "cache.jsonl"

        inner1 = _make_inner()
        _set_return(inner1, "PERSON", "deep reasoning")
        cached1 = CachedLLMClient(inner1, cache_path)
        await cached1.complete("sys", "usr")
        cached1.flush()

        inner2 = _make_inner()
        cached2 = CachedLLMClient(inner2, cache_path)
        await cached2.complete("sys", "usr")  # hit from disk
        thinking = cached2.take_thinking()

        assert thinking == "deep reasoning"


class TestIsAvailable:
    async def test_delegates_to_inner(self, tmp_path: Path):
        inner = _make_inner()
        inner.is_available.return_value = True
        cached = CachedLLMClient(inner, tmp_path / "cache.jsonl")

        assert await cached.is_available() is True
        inner.is_available.assert_awaited_once()
