"""Microphone capture with VAD end-of-utterance detection; optional RMS for UI."""

from __future__ import annotations

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

RmsCallback = Callable[[float], None] | None


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
    in_dev = settings.input_device
    if in_dev is None:
        in_dev = sd.default.device[0]
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

    with sd.InputStream(**stream_kwargs) as stream:
        start = time.monotonic()
        while total_native < max_samples_native:
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("recording_cancelled")
            try:
                data, _overflowed = stream.read(block)
            except Exception:
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
                        break

            if time.monotonic() - start > settings.max_record_seconds:
                break

    append_16k(resampler.end())

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

    out_dev = settings.output_device
    stream_kwargs: dict = {"samplerate": sr, "channels": 1, "dtype": "int16"}
    if out_dev is not None:
        stream_kwargs["device"] = out_dev

    with sd.OutputStream(**stream_kwargs) as stream:
        chunk = int(sr * 0.05)
        for i in range(0, len(audio), chunk):
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("playback_cancelled")
            stream.write(audio[i : i + chunk].reshape(-1, 1))


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
