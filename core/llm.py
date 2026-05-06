"""llama-server client — OpenAI-compatible chat, streaming, and web_search tool loop."""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Mapping, Sequence, cast

import requests

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


class LlamaServerError(Exception):
    """HTTP error from llama-server /chat/completions."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"llama-server HTTP {status_code}: {body[:500]}")


def _chat_completions_url(settings: Settings) -> str:
    base = settings.llamacpp_base_url.rstrip("/")
    return f"{base}/chat/completions"


def _model_rejects_tools(status_code: int, body: str) -> bool:
    """True when the server/model cannot use tools (first round only)."""
    if status_code != 400:
        return False
    text = body.lower()
    return (
        "tool" in text
        and ("not support" in text or "unsupported" in text or "no tools" in text)
    ) or "does not support tools" in text


def _messages_for_plain_chat(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Strip to role/content for chat without tools."""
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


def _tool_args(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return dict(arguments)
    if isinstance(arguments, str):
        if not arguments.strip():
            return {}
        try:
            return cast(dict[str, Any], json.loads(arguments))
        except json.JSONDecodeError:
            log.warning("Invalid tool arguments JSON: %s", arguments[:200])
            return {}
    return {}


def _post_chat_completions(
    settings: Settings,
    *,
    messages: list[Any],
    tools: list[Any] | None = None,
    stream: bool,
    timeout: float = 300.0,
) -> requests.Response:
    url = _chat_completions_url(settings)
    payload: dict[str, Any] = {
        "model": settings.llamacpp_model,
        "messages": messages,
        "stream": stream,
    }
    if tools is not None:
        payload["tools"] = tools
    return requests.post(
        url,
        json=payload,
        stream=stream,
        timeout=timeout,
        headers={"Content-Type": "application/json"},
    )


def _parse_non_stream_response(resp: requests.Response) -> dict[str, Any]:
    if not resp.ok:
        raise LlamaServerError(resp.status_code, resp.text)
    data = resp.json()
    if not isinstance(data, dict):
        raise LlamaServerError(500, "non-JSON or invalid response")
    return data


def _assistant_message_from_choice(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices")
    if not choices or not isinstance(choices, list):
        raise LlamaServerError(500, "missing choices in response")
    first = choices[0]
    if not isinstance(first, dict):
        raise LlamaServerError(500, "invalid choice")
    msg = first.get("message")
    if not isinstance(msg, dict):
        raise LlamaServerError(500, "missing message in choice")
    return msg


def _iter_sse_deltas(resp: requests.Response) -> Iterable[str]:
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices")
        if not choices:
            continue
        delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
        if not isinstance(delta, dict):
            continue
        piece = delta.get("content")
        if piece:
            yield str(piece)


def chat(settings: Settings, messages: list[dict[str, str]], *, stream: bool = True) -> str:
    """Return full assistant text. Streams from llama-server by default for lower TTFT."""
    if stream:
        with _post_chat_completions(
            settings, messages=cast(list[Any], messages), tools=None, stream=True
        ) as resp:
            if not resp.ok:
                raise LlamaServerError(resp.status_code, resp.text)
            return "".join(_iter_sse_deltas(resp)).strip()

    resp = _post_chat_completions(
        settings, messages=cast(list[Any], messages), tools=None, stream=False
    )
    data = _parse_non_stream_response(resp)
    msg = _assistant_message_from_choice(data)
    return str(msg.get("content") or "").strip()


def chat_stream_chunks(
    settings: Settings, messages: list[dict[str, str]]
) -> Iterable[str]:
    """Yield content fragments as they arrive (reserved for streaming TTS)."""
    with _post_chat_completions(
        settings, messages=cast(list[Any], messages), tools=None, stream=True
    ) as resp:
        if not resp.ok:
            raise LlamaServerError(resp.status_code, resp.text)
        yield from _iter_sse_deltas(resp)


def chat_with_tools(
    settings: Settings,
    messages: Sequence[Mapping[str, Any]],
) -> str:
    """
    Chat with web_search tool. Uses non-streaming completions so tool_calls parse reliably.
    """
    tools: list[Any] = [WEB_SEARCH_TOOL]
    messages_work: list[Any] = [dict(m) for m in messages]

    for round_i in range(settings.web_search_max_tool_rounds):
        resp = _post_chat_completions(
            settings,
            messages=messages_work,
            tools=tools,
            stream=False,
        )
        if not resp.ok:
            body = resp.text
            if round_i == 0 and _model_rejects_tools(resp.status_code, body):
                log.warning(
                    "Model %r or server rejected tools; answering without web_search. "
                    "Use a tool-capable setup (see llama.cpp function-calling docs) or set "
                    "PI_ASSISTANT_WEB_SEARCH_ENABLED=false.",
                    settings.llamacpp_model,
                )
                plain = _messages_for_plain_chat(messages)
                return chat(settings, plain, stream=True)
            raise LlamaServerError(resp.status_code, body)

        data = _parse_non_stream_response(resp)
        msg = _assistant_message_from_choice(data)
        tool_calls = msg.get("tool_calls")

        if not tool_calls:
            text = str(msg.get("content") or "").strip()
            return text or "I didn't catch that."

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": msg.get("content"),
            "tool_calls": tool_calls,
        }
        if assistant_msg["content"] is None:
            assistant_msg.pop("content")
        messages_work.append(assistant_msg)

        if not isinstance(tool_calls, list):
            log.warning("Unexpected tool_calls shape")
            return str(msg.get("content") or "").strip() or "I didn't catch that."

        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            if not isinstance(fn, dict):
                continue
            name = str(fn.get("name") or "")
            args = _tool_args(fn.get("arguments"))
            tid = str(tc.get("id") or "")
            if name == "web_search":
                result = search_web(settings, str(args.get("query", "")))
            else:
                result = f"Unknown tool {name!r}."
            tool_msg: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": tid,
                "content": result,
            }
            messages_work.append(tool_msg)

        log.debug("web_search tool round %d completed", round_i + 1)

    return (
        "I used the search tool too many times for one question. "
        "Please ask something simpler or more specific."
    )
