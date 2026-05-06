"""Piper TTS subprocess wrapper."""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from pathlib import Path

from config import Settings

log = logging.getLogger(__name__)


def _resolve_piper_executable(path: Path) -> Path:
    """
    Piper is often unpacked as a directory containing a `piper` executable
    (e.g. ~/piper/piper/piper) while PI_ASSISTANT_PIPER_BIN may point at the folder.
    """
    if path.is_file():
        return path
    nested = path / "piper"
    if nested.is_file():
        return nested
    raise FileNotFoundError(
        f"Piper binary not found at {path} or {nested}. "
        "Set PI_ASSISTANT_PIPER_BIN to the piper executable "
        "(e.g. /home/you/piper/piper/piper)."
    )


def synthesize(settings: Settings, text: str) -> Path:
    piper_exe = _resolve_piper_executable(settings.piper_bin.expanduser().resolve())
    if not settings.piper_voice.expanduser().resolve().is_file():
        raise FileNotFoundError(
            f"Piper voice not found: {settings.piper_voice}. "
            "Set PI_ASSISTANT_PIPER_VOICE to the .onnx file path."
        )

    out = Path(tempfile.mkstemp(suffix=".wav", prefix="piper_")[1])
    voice = settings.piper_voice.expanduser().resolve()
    cmd = [
        str(piper_exe),
        "--model",
        str(voice),
        "--output_file",
        str(out),
    ]
    t0 = time.monotonic()
    log.info("TTS: piper (%d chars) → %s", len(text), out.name)
    log.debug("TTS cmd: %s", " ".join(cmd))
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
    sz = out.stat().st_size
    log.info("TTS: done in %.2fs (%d bytes)", time.monotonic() - t0, sz)
    return out
