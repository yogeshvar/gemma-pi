"""Ollama chat client — streaming chat and multi-turn tool loop for web_search."""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Mapping, Sequence, cast

from ollama import Client

from config import Settings

from .web_search import search_web

log = logging.getLogger(__name__)

WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current events, weather, news, sports, prices, or other "
            "time-sensitive or factual information you cannot answer from memory alone."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Concise search query (keywords, not a full sentence).",
                }
            },
            "required": ["query"],
        },
    },
}


def _client(settings: Settings) -> Client:
    return Client(host=settings.ollama_host)


def _tool_args(arguments: Mapping[str, Any] | str | None) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        if not arguments.strip():
            return {}
        try:
            return cast(dict[str, Any], json.loads(arguments))
        except json.JSONDecodeError:
            log.warning("Invalid tool arguments JSON: %s", arguments[:200])
            return {}
    return dict(arguments)


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


def chat_with_tools(
    settings: Settings,
    messages: Sequence[Mapping[str, Any]],
) -> str:
    """
    Chat with web_search tool. Uses non-streaming completions so tool_calls parse reliably.
    """
    client = _client(settings)
    tools: list[Any] = [WEB_SEARCH_TOOL]
    messages_work: list[Any] = [dict(m) for m in messages]

    for round_i in range(settings.web_search_max_tool_rounds):
        resp = client.chat(
            model=settings.ollama_model,
            messages=messages_work,
            tools=tools,
            stream=False,
        )
        msg = resp.message
        tool_calls = msg.tool_calls

        if not tool_calls:
            text = (msg.content or "").strip()
            return text or "I didn't catch that."

        assistant_dict = msg.model_dump(exclude_none=True)
        messages_work.append(assistant_dict)

        for tc in tool_calls:
            fn = tc.function
            name = fn.name
            args = _tool_args(fn.arguments)
            if name == "web_search":
                result = search_web(settings, str(args.get("query", "")))
            else:
                result = f"Unknown tool {name!r}."
            messages_work.append(
                {
                    "role": "tool",
                    "content": result,
                    "tool_name": name,
                }
            )

        log.debug("web_search tool round %d completed", round_i + 1)

    return (
        "I used the search tool too many times for one question. "
        "Please ask something simpler or more specific."
    )
