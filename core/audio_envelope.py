"""Simple amplitude envelope for mouth animation during TTS."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def wav_envelope(path: Path, bins: int = 48) -> list[float]:
    with wave.open(str(path), "rb") as wf:
        sw = wf.getsampwidth()
        ch = wf.getnchannels()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if sw != 2:
        return [0.0] * bins
    audio = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    audio = np.abs(audio.astype(np.float64))
    n = audio.size
    if n == 0:
        return [0.0] * bins
    hop = max(1, n // bins)
    out: list[float] = []
    for i in range(bins):
        start = i * hop
        end = min(n, start + hop)
        seg = audio[start:end]
        v = float(seg.mean() / 32768.0) if seg.size else 0.0
        out.append(min(1.0, v * 4.0))
    return out
