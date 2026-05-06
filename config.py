"""Central configuration — paths, thresholds, UI colors (override via env / .env)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PI_ASSISTANT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama
    ollama_host: str = Field(default="http://127.0.0.1:11434")
    ollama_model: str = Field(default="gemma3:1b")
    # Appended after merged markdown under prompt_dir (quick experiments / overrides).
    system_prompt: str = Field(default="")

    prompt_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent / "prompts",
        description="Directory of *.md files merged into the system message (lexicographic order)",
    )

    # whisper.cpp
    whisper_cli: Path = Field(
        default_factory=lambda: Path.home() / "whisper.cpp/build/bin/whisper-cli"
    )
    whisper_model: Path = Field(
        default_factory=lambda: Path.home() / "whisper.cpp/models/ggml-base.en.bin"
    )

    # Piper
    piper_bin: Path = Field(default_factory=lambda: Path.home() / "piper/piper")
    piper_voice: Path = Field(
        default_factory=lambda: Path.home() / "piper/piper/en_US-lessac-medium.onnx"
    )

    # Memory
    memory_db_path: Path = Field(
        default_factory=lambda: Path.home() / ".pi-assistant/memory.db"
    )
    context_turns: int = Field(default=10, ge=1, le=50)
    session_idle_minutes: float = Field(default=10.0, ge=0.5)

    # Audio / VAD
    target_sample_rate: int = Field(default=16_000)
    vad_aggressiveness: int = Field(default=2, ge=0, le=3)
    vad_frame_ms: int = Field(default=30, description="webrtcvad frame size: 10, 20, or 30")
    end_of_speech_ms: int = Field(default=800, description="Silence duration after speech to stop")
    max_record_seconds: int = Field(default=120)
    input_device: int | None = Field(default=None, description="sounddevice index; None = default")
    output_device: int | None = Field(default=None)

    # Display
    screen_width: int = Field(default=800)
    screen_height: int = Field(default=480)
    fullscreen: bool = Field(default=True)

    # Logging
    log_level: str = Field(default="INFO")


def get_settings() -> Settings:
    return Settings()
