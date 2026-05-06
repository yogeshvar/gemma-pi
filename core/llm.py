"""Ollama chat client — streaming chat and multi-turn tool loop for web_search."""

from __future__ import annotations

import json
import logging
import time
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


def _model_tags_from_list_response(resp: Any) -> list[str]:
    """Installed model tags from Client.list(), sorted for stable display and default pick."""
    names: list[str] = []
    for item in getattr(resp, "models", []) or []:
        tag = getattr(item, "model", None)
        if isinstance(tag, str) and tag.strip():
            names.append(tag.strip())
    return sorted(set(names))


def sync_ollama_model_from_server(settings: Settings) -> None:
    """
    Query Ollama for installed tags and set ``settings.ollama_model``.

    Preference: use configured tag if it appears in ``ollama list``; otherwise use the
    first sorted installed tag (so you can swap models without editing .env).
    """
    client = _client(settings)
    try:
        resp = client.list()
    except Exception as e:
        log.warning(
            "Ollama list() failed at %s (leaving model=%r): %s",
            settings.ollama_host,
            settings.ollama_model,
            e,
        )
        return

    names = _model_tags_from_list_response(resp)
    if not names:
        log.warning(
            "Ollama at %s returned no models (leaving model=%r); run `ollama pull <tag>`",
            settings.ollama_host,
            settings.ollama_model,
        )
        return

    log.info(
        "Ollama at %s — installed: %s",
        settings.ollama_host,
        ", ".join(names),
    )

    pref = (settings.ollama_model or "").strip()
    if pref and pref in names:
        chosen = pref
        log.info("Using Ollama model %r (matches configuration)", chosen)
    elif pref:
        chosen = names[0]
        log.warning(
            "Configured model %r is not installed; using %r instead",
            pref,
            chosen,
        )
    else:
        chosen = names[0]
        log.info(
            "No model configured (empty PI_ASSISTANT_OLLAMA_MODEL); using %r",
            chosen,
        )
    settings.ollama_model = chosen


def _model_rejects_tools(exc: Exception) -> bool:
    """True when Ollama returns 400 because the model cannot use tools."""
    code = getattr(exc, "status_code", None)
    text = str(exc).lower()
    if code != 400:
        return False
    return "does not support tools" in text or "not support tools" in text


def _messages_for_plain_chat(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Strip to role/content strings for /api/chat without tools."""
    out: list[dict[str, str]] = []
    for m in messages:
        role = str(m.get("role", "user"))
        raw = m.get("content")
        if raw is None:
            content = ""
        elif isinstance(raw, str):
            content = raw
        else:
            content = str(raw)
        out.append({"role": role, "content": content})
    return out


def _think_kw(settings: Settings) -> dict[str, Any]:
    """Map ``Settings.ollama_think`` to Ollama ``Client.chat(..., think=...)`` (omit if None)."""
    if settings.ollama_think is None:
        return {}
    return {"think": settings.ollama_think}


def _chat_with_think_fallback(client: Client, settings: Settings, **kwargs: Any) -> Any:
    """Call ``client.chat`` with ``think`` when configured; retry without on old ollama-python."""
    extra = _think_kw(settings)
    if not extra:
        return client.chat(**kwargs)
    try:
        return client.chat(**kwargs, **extra)
    except TypeError as e:
        if "think" in str(e).lower():
            log.warning(
                "ollama Python package rejected `think` (install ollama>=0.4); retrying without: %s",
                e,
            )
            return client.chat(**kwargs)
        raise


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
    t0 = time.monotonic()
    log.info(
        "[pipeline] LLM: chat model=%r host=%s stream=%s think=%r",
        settings.ollama_model,
        settings.ollama_host,
        stream,
        settings.ollama_think,
    )
    if stream:
        parts: list[str] = []
        for chunk in _chat_with_think_fallback(
            client,
            settings,
            model=settings.ollama_model,
            messages=messages,
            stream=True,
        ):
            msg = chunk.get("message") or {}
            c = msg.get("content")
            if c:
                parts.append(c)
        text = "".join(parts).strip()
        log.info(
            "[pipeline] LLM: reply in %.2fs (%d chars)",
            time.monotonic() - t0,
            len(text),
        )
        return text

    resp = _chat_with_think_fallback(
        client,
        settings,
        model=settings.ollama_model,
        messages=messages,
        stream=False,
    )
    msg = resp.get("message") or {}
    text = str(msg.get("content", "")).strip()
    log.info(
        "[pipeline] LLM: reply in %.2fs (%d chars)",
        time.monotonic() - t0,
        len(text),
    )
    return text


def chat_stream_chunks(
    settings: Settings, messages: list[dict[str, str]]
) -> Iterable[str]:
    """Yield content fragments as they arrive (reserved for streaming TTS)."""
    client = _client(settings)
    for chunk in _chat_with_think_fallback(
        client,
        settings,
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
    t0 = time.monotonic()
    log.info(
        "[pipeline] LLM: chat_with_tools model=%r host=%s max_rounds=%s think=%r",
        settings.ollama_model,
        settings.ollama_host,
        settings.web_search_max_tool_rounds,
        settings.ollama_think,
    )

    for round_i in range(settings.web_search_max_tool_rounds):
        try:
            resp = _chat_with_think_fallback(
                client,
                settings,
                model=settings.ollama_model,
                messages=messages_work,
                tools=tools,
                stream=False,
            )
        except Exception as e:
            if round_i == 0 and _model_rejects_tools(e):
                log.warning(
                    "Model %r does not support tools; answering without web_search. "
                    "Pull a tool-capable model (e.g. llama3.2) or set "
                    "PI_ASSISTANT_WEB_SEARCH_ENABLED=false.",
                    settings.ollama_model,
                )
                plain = _messages_for_plain_chat(messages)
                return chat(settings, plain, stream=True)
            raise
        msg = resp.message
        tool_calls = msg.tool_calls

        if not tool_calls:
            text = (msg.content or "").strip()
            text = text or "I didn't catch that."
            log.info(
                "[pipeline] LLM: tools done in %.2fs (%d chars)",
                time.monotonic() - t0,
                len(text),
            )
            return text

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

    log.warning(
        "[pipeline] LLM: web_search max tool rounds exhausted after %.2fs",
        time.monotonic() - t0,
    )
    return (
        "I used the search tool too many times for one question. "
        "Please ask something simpler or more specific."
    )
