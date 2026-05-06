"""Web search backends for the web_search tool (Brave, Tavily, DuckDuckGo)."""

from __future__ import annotations

import logging
import requests

from config import Settings

log = logging.getLogger(__name__)

_BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
_TAVILY_URL = "https://api.tavily.com/search"


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _format_hits(lines: list[str], max_chars: int) -> str:
    body = "\n".join(lines).strip()
    return _truncate(body, max_chars)


def _search_brave(settings: Settings, query: str) -> str:
    if not (settings.brave_api_key or "").strip():
        return "Search error: Brave Search requires PI_ASSISTANT_BRAVE_API_KEY."
    try:
        r = requests.get(
            _BRAVE_URL,
            params={
                "q": query,
                "count": settings.web_search_max_results,
            },
            headers={"X-Subscription-Token": settings.brave_api_key.strip()},
            timeout=settings.web_search_timeout_seconds,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        log.warning("Brave search failed: %s", e)
        return f"Search error: {e}"

    results = data.get("web", {}).get("results") or []
    lines: list[str] = []
    for i, item in enumerate(results[: settings.web_search_max_results], start=1):
        title = (item.get("title") or "").strip()
        desc = (item.get("description") or "").strip()
        url = (item.get("url") or "").strip()
        if title or desc:
            lines.append(f"{i}. {title}\n   {desc}\n   {url}")
    if not lines:
        return "No web results returned."
    return _format_hits(lines, settings.web_search_max_chars)


def _search_tavily(settings: Settings, query: str) -> str:
    if not (settings.tavily_api_key or "").strip():
        return "Search error: Tavily requires PI_ASSISTANT_TAVILY_API_KEY."
    try:
        r = requests.post(
            _TAVILY_URL,
            json={
                "api_key": settings.tavily_api_key.strip(),
                "query": query,
                "max_results": settings.web_search_max_results,
            },
            timeout=settings.web_search_timeout_seconds,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        log.warning("Tavily search failed: %s", e)
        return f"Search error: {e}"

    results = data.get("results") or []
    lines: list[str] = []
    for i, item in enumerate(results[: settings.web_search_max_results], start=1):
        title = (item.get("title") or "").strip()
        content = (item.get("content") or "").strip()
        url = (item.get("url") or "").strip()
        if title or content:
            lines.append(f"{i}. {title}\n   {content}\n   {url}")
    if not lines:
        return "No web results returned."
    return _format_hits(lines, settings.web_search_max_chars)


def _search_ddgs(settings: Settings, query: str) -> str:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "Search error: duckduckgo-search is not installed."

    lines: list[str] = []
    try:
        with DDGS() as ddgs:
            gen = ddgs.text(query, max_results=settings.web_search_max_results)
            if gen is None:
                return "No web results returned."
            for i, item in enumerate(gen, start=1):
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "").strip()
                body = str(item.get("body") or "").strip()
                href = str(item.get("href") or "").strip()
                if title or body:
                    lines.append(f"{i}. {title}\n   {body}\n   {href}")
                if i >= settings.web_search_max_results:
                    break
    except Exception as e:
        log.warning("DDG search failed: %s", e)
        return f"Search error: {e}"

    if not lines:
        return "No web results returned."
    return _format_hits(lines, settings.web_search_max_chars)


def search_web(settings: Settings, query: str) -> str:
    """
    Run a web search and return a plain-text blob for the model (truncated).
    """
    q = (query or "").strip()
    if not q:
        return "Search error: empty query."

    provider = settings.web_search_provider
    if provider == "brave":
        return _search_brave(settings, q)
    if provider == "tavily":
        return _search_tavily(settings, q)
    if provider == "ddgs":
        return _search_ddgs(settings, q)
    return f"Search error: unknown provider {provider!r}."
