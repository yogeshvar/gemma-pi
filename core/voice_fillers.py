"""Short spoken phrases while waiting on STT / LLM (cancelable playback)."""

from __future__ import annotations

import random

ACK_PHRASES: tuple[str, ...] = (
    "Got it.",
    "Okay.",
    "Mm-hmm.",
    "Right.",
    "Sure.",
)

THINKING_PHRASES: tuple[str, ...] = (
    "Hmm, let me think.",
    "Good question — one sec.",
    "Interesting, hang on.",
    "Working on that.",
    "Just a moment.",
    "Let me see.",
    "Okay, thinking.",
)


def random_ack() -> str:
    return random.choice(ACK_PHRASES)


def random_thinking() -> str:
    return random.choice(THINKING_PHRASES)
