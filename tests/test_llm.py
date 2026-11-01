"""Tests for llm.py. A fake client replaces Groq, so they use no network and no quota."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from qualite.llm import build_cache_key, chat, read_cache, write_cache

HELLO = [{"role": "user", "content": "Hello"}]


class FakeMessage:
    """Copy the one method of a real message that chat() uses."""

    def model_dump(self, exclude_none: bool) -> dict[str, str]:
        return {"role": "assistant", "content": "Hi!"}


class FakeClient:
    """Count calls, and optionally fail like a real API error."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls = 0
        self.error = error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request: Any) -> SimpleNamespace:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=FakeMessage())],
            usage=SimpleNamespace(total_tokens=12),
        )


def test_same_request_gives_same_key() -> None:
    first = {"model": "m", "temperature": 0}
    second = {"temperature": 0, "model": "m"}
    assert build_cache_key(first) == build_cache_key(second)


def test_changed_request_gives_new_key() -> None:
    assert build_cache_key({"model": "a"}) != build_cache_key({"model": "b"})


def test_cache_round_trip(tmp_path: Path) -> None:
    write_cache("abc", {"total_tokens": 5}, tmp_path)
    assert read_cache("abc", tmp_path) == {"total_tokens": 5}


def test_missing_entry_returns_none(tmp_path: Path) -> None:
    assert read_cache("unknown", tmp_path) is None


def test_damaged_file_counts_as_missing(tmp_path: Path) -> None:
    (tmp_path / "abc.json").write_text("{not json", encoding="utf-8")
    assert read_cache("abc", tmp_path) is None


def test_second_identical_call_uses_the_cache(tmp_path: Path) -> None:
    client = FakeClient()

    first = chat(HELLO, model="m", client=client, cache_dir=tmp_path)
    second = chat(HELLO, model="m", client=client, cache_dir=tmp_path)

    assert client.calls == 1
    assert first.from_cache is False
    assert second.from_cache is True
    assert second.message == first.message
    assert second.total_tokens == 12


def test_different_model_calls_the_api_again(tmp_path: Path) -> None:
    client = FakeClient()

    chat(HELLO, model="a", client=client, cache_dir=tmp_path)
    chat(HELLO, model="b", client=client, cache_dir=tmp_path)

    assert client.calls == 2


def test_errors_are_not_cached(tmp_path: Path) -> None:
    failing = FakeClient(error=RuntimeError("temporary failure"))
    with pytest.raises(RuntimeError):
        chat(HELLO, model="m", client=failing, cache_dir=tmp_path)

    working = FakeClient()
    reply = chat(HELLO, model="m", client=working, cache_dir=tmp_path)

    assert working.calls == 1
    assert reply.from_cache is False
