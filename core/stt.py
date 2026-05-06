"""whisper.cpp CLI wrapper."""

from __future__ import annotations

import logging
import subprocess
import tempfile
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
    log.debug("Running STT: %s", " ".join(cmd))
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

    return text.strip()
