"""Load markdown system prompts from prompts/ (lexicographic order)."""

from __future__ import annotations

import logging
from pathlib import Path

from config import Settings

log = logging.getLogger(__name__)

_SEPARATOR = "\n\n---\n\n"

# Used only when prompt_dir is missing or contains no usable .md files.
_FALLBACK_SYSTEM = (
    "You are Pi, a friendly offline voice assistant on a Raspberry Pi. "
    "Keep answers concise and conversational unless the user asks for detail."
)


def _merge_markdown_files(directory: Path) -> str:
    """Merge *.md in directory (sorted), excluding README.md and _*.md."""
    if not directory.is_dir():
        return ""

    paths = sorted(
        p
        for p in directory.glob("*.md")
        if p.is_file()
        and p.name.lower() != "readme.md"
        and not p.name.startswith("_")
    )
    chunks: list[str] = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8").strip()
        except OSError as e:
            log.warning("Skip prompt file %s: %s", p, e)
            continue
        if text:
            chunks.append(text)
    if not chunks:
        return ""
    return _SEPARATOR.join(chunks)


def load_merged_system_prompt(settings: Settings) -> str:
    """
    Merge all *.md directly under prompt_dir (sorted), excluding README.md and _*.md.
    Returns non-empty string; falls back to _FALLBACK_SYSTEM if nothing loaded.
    """
    root = settings.prompt_dir.expanduser().resolve()
    if not root.is_dir():
        log.warning("Prompt dir missing: %s — using fallback system prompt", root)
        return _FALLBACK_SYSTEM

    merged = _merge_markdown_files(root)
    if not merged:
        log.warning("No prompt *.md under %s — using fallback system prompt", root)
        return _FALLBACK_SYSTEM
    return merged


def load_web_prompts(settings: Settings) -> str:
    """Merge prompts/web/*.md when web search is enabled; empty if disabled or missing."""
    if not settings.web_search_enabled:
        return ""
    root = settings.prompt_dir.expanduser().resolve()
    web_dir = root / "web"
    return _merge_markdown_files(web_dir)


def build_system_message(settings: Settings) -> str:
    """File-based prompts, optional web-search risk/tool markdown, plus PI_ASSISTANT_SYSTEM_PROMPT last."""
    base = load_merged_system_prompt(settings)
    web = load_web_prompts(settings)
    if web:
        base = base + _SEPARATOR + web
    extra = (settings.system_prompt or "").strip()
    if not extra:
        return base
    return base + _SEPARATOR + extra
