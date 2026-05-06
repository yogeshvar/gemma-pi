"""Voice activity detection using webrtcvad on fixed-size PCM frames."""

from __future__ import annotations

import audioop

import numpy as np
import webrtcvad

from config import Settings


def _frame_samples(sample_rate: int, frame_ms: int) -> int:
    if frame_ms not in (10, 20, 30):
        raise ValueError("webrtcvad frame_ms must be 10, 20, or 30")
    return (sample_rate * frame_ms) // 1000


class StreamingMonoResampler:
    """Incremental audioop.ratecv for live mic → 16 kHz."""

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        self._src_rate = src_rate
        self._dst_rate = dst_rate
        self._state: tuple | None = None

    def process(self, pcm_int16: np.ndarray) -> np.ndarray:
        if pcm_int16.size == 0:
            return np.empty(0, dtype=np.int16)
        data = pcm_int16.astype("<i2", copy=False).tobytes()
        out, self._state = audioop.ratecv(
            data, 2, 1, self._src_rate, self._dst_rate, self._state
        )
        return np.frombuffer(out, dtype=np.int16) if out else np.empty(0, dtype=np.int16)

    def end(self) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for _ in range(256):
            out, self._state = audioop.ratecv(
                b"", 2, 1, self._src_rate, self._dst_rate, self._state
            )
            if out:
                chunks.append(np.frombuffer(out, dtype=np.int16))
            if not out:
                break
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int16)


def resample_mono_int16(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample mono int16 using stdlib audioop (good enough for VAD / whisper)."""
    if src_rate == dst_rate:
        return pcm.astype(np.int16, copy=False)
    data = pcm.astype("<i2").tobytes()
    state = None
    out_chunks: list[bytes] = []
    # ratecv processes in chunks; feed whole buffer
    fragment, state = audioop.ratecv(data, 2, 1, src_rate, dst_rate, state)
    out_chunks.append(fragment)
    while state and state[3] != 0:
        fragment, state = audioop.ratecv(b"", 2, 1, src_rate, dst_rate, state)
        if fragment:
            out_chunks.append(fragment)
    out = b"".join(out_chunks)
    return np.frombuffer(out, dtype=np.int16)


class VoiceActivityDetector:
    def __init__(self, settings: Settings) -> None:
        self._vad = webrtcvad.Vad(settings.vad_aggressiveness)
        self._rate = settings.target_sample_rate
        if settings.vad_frame_ms not in (10, 20, 30):
            raise ValueError("PI_ASSISTANT_VAD_FRAME_MS must be 10, 20, or 30")
        self._frame_ms = settings.vad_frame_ms
        self._frame_bytes = _frame_samples(self._rate, self._frame_ms) * 2

    @property
    def frame_ms(self) -> int:
        return self._frame_ms

    @property
    def frame_bytes(self) -> int:
        return self._frame_bytes

    @property
    def sample_rate(self) -> int:
        return self._rate

    def is_speech(self, frame_int16_mono: bytes) -> bool:
        if len(frame_int16_mono) != self._frame_bytes:
            raise ValueError(
                f"VAD frame must be {self._frame_bytes} bytes, got {len(frame_int16_mono)}"
            )
        return bool(self._vad.is_speech(frame_int16_mono, self._rate))


def pcm_bytes_to_frames(pcm_16k: bytes, frame_bytes: int) -> list[bytes]:
    return [pcm_16k[i : i + frame_bytes] for i in range(0, len(pcm_16k) - frame_bytes + 1, frame_bytes)]
