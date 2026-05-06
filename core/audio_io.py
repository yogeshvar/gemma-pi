"""Microphone capture with VAD end-of-utterance detection; optional RMS for UI."""

from __future__ import annotations

import logging
import threading
import time
import wave
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd

from config import Settings

from .vad import StreamingMonoResampler, VoiceActivityDetector

log = logging.getLogger(__name__)

RmsCallback = Callable[[float], None] | None

_ERR_NO_INPUT_DEFAULT = (
    "No default audio input device (PortAudio reports default index -1). "
    "Use a graphical session with PipeWire/Pulse running, set PI_ASSISTANT_INPUT_DEVICE "
    "to a device index or name (e.g. pipewire), and check: pactl get-default-source"
)

_ERR_NO_OUTPUT_DEFAULT = (
    "No default audio output device (PortAudio reports default index -1). "
    "Set PI_ASSISTANT_OUTPUT_DEVICE (e.g. pipewire) and check: pactl get-default-sink"
)


def resolve_input_device(settings: Settings) -> int | str:
    """PortAudio input id or host/device name substring for sounddevice."""
    if settings.input_device is not None:
        return settings.input_device
    idx = sd.default.device[0]
    if idx is None or int(idx) < 0:
        raise RuntimeError(_ERR_NO_INPUT_DEFAULT)
    return int(idx)


def resolve_output_device(settings: Settings) -> int | str:
    if settings.output_device is not None:
        return settings.output_device
    idx = sd.default.device[1]
    if idx is None or int(idx) < 0:
        raise RuntimeError(_ERR_NO_OUTPUT_DEFAULT)
    return int(idx)


def verify_audio_devices(settings: Settings) -> None:
    """Log resolved devices; raise RuntimeError with hints if they are not usable."""
    in_dev = resolve_input_device(settings)
    out_dev = resolve_output_device(settings)
    try:
        in_info = sd.query_devices(in_dev, "input")
        out_info = sd.query_devices(out_dev, "output")
    except Exception as e:
        raise RuntimeError(
            f"Audio device not available (input={in_dev!r}, output={out_dev!r}). "
            "Run `python -c \"import sounddevice as sd; print(sd.query_devices()); "
            'print(sd.default.device)"` and set PI_ASSISTANT_INPUT_DEVICE / OUTPUT_DEVICE.'
        ) from e
    log.info(
        "Audio input: %s (~%s Hz)",
        in_info["name"],
        int(in_info["default_samplerate"]),
    )
    log.info(
        "Audio output: %s (~%s Hz)",
        out_info["name"],
        int(out_info["default_samplerate"]),
    )


def _write_wav_mono16(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.astype("<i2", copy=False).tobytes())


def record_until_silence(
    settings: Settings,
    vad: VoiceActivityDetector,
    *,
    rms_callback: RmsCallback = None,
    cancel_event: threading.Event | None = None,
) -> Path:
    """
    Record from default input until silence after speech, or max duration.
    Returns path to a temporary 16 kHz mono WAV suitable for whisper.cpp.
    """
    in_dev = resolve_input_device(settings)
    device_info = sd.query_devices(in_dev, "input")
    device_sr = int(device_info["default_samplerate"])
    block = max(256, device_sr // 50)  # ~20ms at native rate

    silence_frames_needed = max(
        1, (settings.end_of_speech_ms + vad.frame_ms - 1) // vad.frame_ms
    )
    max_samples_native = settings.max_record_seconds * device_sr

    resampler = StreamingMonoResampler(device_sr, vad.sample_rate)
    buffer_16k = bytearray()

    heard_speech = False
    silence_run = 0
    total_native = 0
    exit_reason = "unknown"

    def append_16k(chunk: np.ndarray) -> None:
        if chunk.size:
            buffer_16k.extend(chunk.astype("<i2", copy=False).tobytes())

    stream_kwargs: dict = {
        "samplerate": device_sr,
        "channels": 1,
        "dtype": "int16",
        "blocksize": block,
        "device": in_dev,
    }

    log.info(
        "Recording: device_sr=%s block=%s max_record_s=%s",
        device_sr,
        block,
        settings.max_record_seconds,
    )
    with sd.InputStream(**stream_kwargs) as stream:
        start = time.monotonic()
        while total_native < max_samples_native:
            if cancel_event and cancel_event.is_set():
                exit_reason = "cancelled"
                raise RuntimeError("recording_cancelled")
            try:
                data, _overflowed = stream.read(block)
            except Exception as e:
                log.warning("Mic stream read failed (stopping capture): %s", e)
                exit_reason = "read_error"
                break
            mono = data[:, 0].copy() if data.ndim > 1 else data.reshape(-1).copy()
            total_native += mono.size

            if rms_callback and mono.size:
                rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)) / 32768.0)
                rms_callback(rms)

            pcm_16k = resampler.process(mono)
            append_16k(pcm_16k)

            # Never hold memoryview(buffer_16k) across iterations — extend() would
            # fail with "Existing exports of data: object cannot be re-sized" (Py 3.13+).
            fb = vad.frame_bytes
            if len(buffer_16k) < fb:
                continue
            last_frame = bytes(buffer_16k[-fb:])
            is_sp = vad.is_speech(last_frame)
            if not heard_speech:
                if is_sp:
                    heard_speech = True
                    silence_run = 0
            else:
                if is_sp:
                    silence_run = 0
                else:
                    silence_run += 1
                    if silence_run >= silence_frames_needed:
                        exit_reason = "silence_after_speech"
                        break

            if time.monotonic() - start > settings.max_record_seconds:
                exit_reason = "max_wall_seconds"
                break

        if exit_reason == "unknown" and total_native >= max_samples_native:
            exit_reason = "max_samples"

    append_16k(resampler.end())

    wall = time.monotonic() - start
    samples_16k = len(buffer_16k) // 2 if buffer_16k else 0
    log.info(
        "Recording done: reason=%s wall=%.2fs heard_speech=%s samples_16k=%d",
        exit_reason,
        wall,
        heard_speech,
        samples_16k,
    )

    if not buffer_16k:
        raise RuntimeError("no_audio_captured")

    samples = np.frombuffer(buffer_16k, dtype=np.int16)
    out = Path("/tmp") / f"pi_assistant_{int(time.time() * 1000)}.wav"
    _write_wav_mono16(out, samples, vad.sample_rate)
    return out


def play_wav(
    path: Path,
    settings: Settings,
    *,
    cancel_event: threading.Event | None = None,
) -> None:
    with wave.open(str(path), "rb") as wf:
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if sw != 2:
        raise ValueError("only 16-bit WAV playback supported")
    audio = np.frombuffer(frames, dtype=np.int16)
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1).astype(np.int16)

    out_dev = resolve_output_device(settings)
    stream_kwargs: dict = {
        "samplerate": sr,
        "channels": 1,
        "dtype": "int16",
        "device": out_dev,
    }

    dur_s = len(audio) / float(sr) if sr else 0.0
    t0 = time.monotonic()
    log.info("Playback: %s @ %d Hz, ~%.2fs of audio", path.name, sr, dur_s)
    with sd.OutputStream(**stream_kwargs) as stream:
        chunk = int(sr * 0.05)
        for i in range(0, len(audio), chunk):
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("playback_cancelled")
            stream.write(audio[i : i + chunk].reshape(-1, 1))
    log.info("Playback finished in %.2fs", time.monotonic() - t0)


class RmsRingBuffer:
    """Thread-safe recent RMS values for waveform UI."""

    def __init__(self, maxlen: int = 64) -> None:
        self._dq: deque[float] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, v: float) -> None:
        with self._lock:
            self._dq.append(v)

    def snapshot(self) -> list[float]:
        with self._lock:
            return list(self._dq)
