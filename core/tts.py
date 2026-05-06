"""Piper TTS subprocess wrapper."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from config import Settings

log = logging.getLogger(__name__)


def synthesize(settings: Settings, text: str) -> Path:
    if not settings.piper_bin.is_file():
        raise FileNotFoundError(f"Piper binary not found: {settings.piper_bin}")
    if not settings.piper_voice.is_file():
        raise FileNotFoundError(f"Piper voice not found: {settings.piper_voice}")

    out = Path(tempfile.mkstemp(suffix=".wav", prefix="piper_")[1])
    cmd = [
        str(settings.piper_bin),
        "--model",
        str(settings.piper_voice),
        "--output_file",
        str(out),
    ]
    log.debug("Running TTS: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        capture_output=True,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        log.error("piper stderr: %s", err)
        raise RuntimeError(f"Piper failed ({proc.returncode}): {err.strip()}")
    if not out.is_file() or out.stat().st_size < 64:
        raise RuntimeError("Piper produced empty or missing WAV")
    return out
