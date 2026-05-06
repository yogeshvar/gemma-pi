"""Blink, drift, and tween helpers for the face."""

from __future__ import annotations

import math
import random
import time


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


class BlinkController:
    def __init__(self) -> None:
        self._next_blink = time.monotonic() + random.uniform(2.5, 5.0)
        self._blink_start: float | None = None

    def update(self, now: float) -> float:
        """Return eye-open scale 0..1 (0 = fully closed)."""
        if self._blink_start is None and now >= self._next_blink:
            self._blink_start = now
        if self._blink_start is not None:
            dt = now - self._blink_start
            if dt < 0.1:
                return lerp(1.0, 0.0, smoothstep(dt / 0.1))
            if dt < 0.2:
                return lerp(0.0, 1.0, smoothstep((dt - 0.1) / 0.1))
            self._blink_start = None
            self._next_blink = now + random.uniform(2.5, 5.0)
        return 1.0


def breathing_scale(now: float, hz: float = 0.25) -> float:
    return 1.0 + 0.02 * math.sin(now * hz * 2.0 * math.pi)


def pupil_offset(state_name: str, now: float) -> tuple[float, float]:
    """Subtle look direction per assistant state."""
    t = now * 0.7
    if state_name == "THINKING":
        return (2.0, -6.0)
    if state_name == "SPEAKING":
        return (math.sin(t) * 2.0, 2.0)
    if state_name == "LISTENING":
        return (0.0, 0.0)
    return (math.sin(t * 0.4) * 3.0, math.cos(t * 0.35) * 2.0)
