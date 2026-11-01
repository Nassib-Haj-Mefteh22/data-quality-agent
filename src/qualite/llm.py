"""The one place where the project calls the LLM, with a disk cache.

WHY one module: every call follows the same rules, such as temperature 0 and caching.
WHY a cache: identical requests do not use the free quota again, and reruns give the same reply.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from qualite.config import CACHE_DIR, Settings, load_settings


@dataclass(frozen=True)
class ChatReply:
    """A model reply.

    WHY keep total_tokens on cached replies: Step 8 measures what a run costs, even when
    this particular rerun used no quota. from_cache tells the two situations apart.
    """

    message: dict[str, Any]
    total_tokens: int
    from_cache: bool


def make_client(settings: Settings) -> OpenAI:
    """Create the Groq client.

    WHY the OpenAI client: Groq has an OpenAI-compatible API, so only the address changes.
    """
    return OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)


def build_cache_key(request: dict[str, Any]) -> str:
    """Turn a full request into a fixed ID.

    WHY the whole request: changing the model, one word or one tool must give a new ID.
    WHY sort_keys: the same request must give the same ID, whatever the key order.
    """
    text = json.dumps(request, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_cache(key: str, cache_dir: Path) -> dict[str, Any] | None:
    """Return a saved reply, or None if there is none.

    WHY treat a damaged file as missing: the program can simply ask the model again.
    """
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_cache(key: str, payload: dict[str, Any], cache_dir: Path) -> None:
    """Save one reply as a readable JSON file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
    model: str | None = None,
    client: OpenAI | None = None,
    cache_dir: Path = CACHE_DIR,
) -> ChatReply:
    """Answer from the cache if possible, otherwise call the model and save the reply.

    WHY client is a parameter: tests pass a fake client, so they need no network or quota.
    WHY errors are not caught here: an exception stops before write_cache, so a failure
    is never saved and cannot come back on the next run.
    """
    if model is None or client is None:
        settings = load_settings()
        model = model or settings.model
        client = client or make_client(settings)

    request: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0}
    if tools is not None:
        request["tools"] = tools
    if response_format is not None:
        request["response_format"] = response_format

    key = build_cache_key(request)
    cached = read_cache(key, cache_dir)
    if cached is not None:
        return ChatReply(cached["message"], cached["total_tokens"], from_cache=True)

    response = client.chat.completions.create(**request)
    payload = {
        "message": response.choices[0].message.model_dump(exclude_none=True),
        "total_tokens": response.usage.total_tokens,
    }
    write_cache(key, payload, cache_dir)
    return ChatReply(payload["message"], payload["total_tokens"], from_cache=False)


if __name__ == "__main__":
    hello = [{"role": "user", "content": "Say hello in five words or fewer."}]
    for attempt in (1, 2):
        reply = chat(hello)
        source = "cache" if reply.from_cache else "Groq"
        quota_used = 0 if reply.from_cache else reply.total_tokens
        print(f"Call {attempt}: from {source}, quota used: {quota_used} tokens")
        print(f"Reply: {reply.message.get('content')}")
