"""Ollama chat client — accumulates streamed tokens for one assistant reply."""

from __future__ import annotations

from typing import Iterable

from ollama import Client

from config import Settings


def _client(settings: Settings) -> Client:
    # ollama package accepts host like http://127.0.0.1:11434
    return Client(host=settings.ollama_host)


def chat(settings: Settings, messages: list[dict[str, str]], *, stream: bool = True) -> str:
    """Return full assistant text. Streams from Ollama by default for lower time-to-first-token."""
    client = _client(settings)
    if stream:
        parts: list[str] = []
        for chunk in client.chat(
            model=settings.ollama_model,
            messages=messages,
            stream=True,
        ):
            msg = chunk.get("message") or {}
            c = msg.get("content")
            if c:
                parts.append(c)
        return "".join(parts).strip()

    resp = client.chat(
        model=settings.ollama_model,
        messages=messages,
        stream=False,
    )
    msg = resp.get("message") or {}
    return str(msg.get("content", "")).strip()


def chat_stream_chunks(
    settings: Settings, messages: list[dict[str, str]]
) -> Iterable[str]:
    """Yield content fragments as they arrive (reserved for streaming TTS)."""
    client = _client(settings)
    for chunk in client.chat(
        model=settings.ollama_model,
        messages=messages,
        stream=True,
    ):
        msg = chunk.get("message") or {}
        c = msg.get("content")
        if c:
            yield c
