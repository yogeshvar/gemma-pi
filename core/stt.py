"""whisper.cpp CLI wrapper."""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from pathlib import Path

from config import Settings

log = logging.getLogger(__name__)


def transcribe(settings: Settings, wav_path: Path) -> str:
    if not settings.whisper_cli.is_file():
        raise FileNotFoundError(f"whisper-cli not found: {settings.whisper_cli}")
    if not settings.whisper_model.is_file():
        raise FileNotFoundError(f"Whisper model not found: {settings.whisper_model}")

    with tempfile.NamedTemporaryFile(
        prefix="whisper_out_", delete=False, dir=None
    ) as tmp:
        out_base = Path(tmp.name)

    cmd = [
        str(settings.whisper_cli),
        "-m",
        str(settings.whisper_model),
        "-f",
        str(wav_path),
        "-of",
        str(out_base),
        "-otxt",
        "-nt",
        "-np",
    ]
    t0 = time.monotonic()
    try:
        sz = wav_path.stat().st_size
    except OSError:
        sz = -1
    log.info("STT: whisper-cli on %s (~%d bytes)", wav_path.name, sz)
    log.debug("STT cmd: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log.error("whisper-cli stderr: %s", proc.stderr)
        raise RuntimeError(f"whisper-cli failed ({proc.returncode}): {proc.stderr.strip()}")

    text = ""
    for p in out_base.parent.glob(out_base.name + "*"):
        if p.suffix.lower() == ".txt" and p.is_file():
            text = p.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                break

    for p in list(out_base.parent.glob(out_base.name + "*")):
        try:
            if p.is_file():
                p.unlink(missing_ok=True)
        except OSError:
            pass

    out = text.strip()
    log.info("STT: finished in %.2fs (%d chars)", time.monotonic() - t0, len(out))
    return out
