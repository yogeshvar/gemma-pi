"""State machine: IDLE → LISTENING → THINKING → SPEAKING, wired to workers."""

from __future__ import annotations

import logging
import threading
import time
import wave
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum, auto
from pathlib import Path

from config import Settings

from .daemon_executor import DaemonThreadPoolExecutor
from .audio_envelope import wav_envelope
from .audio_io import RmsRingBuffer, play_wav, record_until_silence
from .llm import chat, chat_with_tools
from .memory import MemoryStore
from .prompts import build_system_message
from .stt import transcribe
from .tts import synthesize
from .vad import VoiceActivityDetector
from .voice_fillers import random_ack, random_thinking

log = logging.getLogger(__name__)


class AssistantState(Enum):
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()
    ERROR = auto()


@dataclass
class UIFrame:
    state: AssistantState
    caption: str
    subtitle: str
    waveform: list[float] = field(default_factory=list)
    envelope: list[float] = field(default_factory=list)
    speak_progress: float = 0.0


class AssistantController:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.memory = MemoryStore(settings.memory_db_path)
        self.vad = VoiceActivityDetector(settings)
        self.rms = RmsRingBuffer(maxlen=64)

        self._state = AssistantState.IDLE
        self._lock = threading.Lock()
        self._caption = "Tap the screen or press Enter to talk"
        self._subtitle = ""
        self._envelope: list[float] = []
        self._speak_start = 0.0
        self._speak_duration = 1.0

        self._cancel_record = threading.Event()
        self._cancel_play = threading.Event()
        self._filler_cancel = threading.Event()
        self._filler_thread: threading.Thread | None = None
        self._executor = DaemonThreadPoolExecutor(max_workers=2, thread_name_prefix="pi_ast")
        self._record_future: Future | None = None
        self._conversation_id = 0
        self._system_message = build_system_message(settings)

    def close(self) -> None:
        self._cancel_record.set()
        self._cancel_play.set()
        self._stop_thinking_fillers()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.memory.close()

    def _stop_thinking_fillers(self) -> None:
        self._filler_cancel.set()
        th = self._filler_thread
        self._filler_thread = None
        if th is not None and th.is_alive():
            th.join(timeout=2.5)

    def _start_thinking_fillers(self) -> None:
        if not self.settings.voice_fillers_enabled:
            return
        self._stop_thinking_fillers()
        self._filler_cancel.clear()
        th = threading.Thread(
            target=self._thinking_filler_runner,
            name="voice_fillers",
            daemon=True,
        )
        self._filler_thread = th
        th.start()

    def _thinking_filler_runner(self) -> None:
        settings = self.settings
        for phrase in (random_ack(), random_thinking()):
            if self._filler_cancel.is_set():
                return
            path: Path | None = None
            try:
                path = synthesize(settings, phrase)
                play_wav(path, settings, cancel_event=self._filler_cancel)
            except Exception as e:
                log.debug("Voice filler skipped: %s", e)
            finally:
                if path is not None:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass

    def _get_state_locked(self) -> AssistantState:
        with self._lock:
            return self._state

    @property
    def state(self) -> AssistantState:
        return self._get_state_locked()

    def _set_state(
        self,
        s: AssistantState,
        caption: str,
        subtitle: str = "",
        *,
        envelope: list[float] | None = None,
        speak_duration: float | None = None,
    ) -> None:
        with self._lock:
            self._state = s
            self._caption = caption
            self._subtitle = subtitle
            if envelope is not None:
                self._envelope = envelope
            if speak_duration is not None:
                self._speak_duration = max(0.05, speak_duration)
            if s == AssistantState.SPEAKING:
                self._speak_start = time.monotonic()

    def ui_snapshot(self, now_mono: float | None = None) -> UIFrame:
        now = time.monotonic() if now_mono is None else now_mono
        with self._lock:
            st = self._state
            cap = self._caption
            sub = self._subtitle
            env = list(self._envelope)
            wf = self.rms.snapshot()
            dur = self._speak_duration
            start = self._speak_start
        prog = 0.0
        if st == AssistantState.SPEAKING and dur > 0:
            prog = min(1.0, max(0.0, (now - start) / dur))
        return UIFrame(
            state=st,
            caption=cap,
            subtitle=sub,
            waveform=wf,
            envelope=env,
            speak_progress=prog,
        )

    def _idle_caption(self) -> str:
        return "Tap the screen or press Enter to talk"

    def dismiss_error(self) -> None:
        """Leave ERROR and return to normal idle (no new listen)."""
        if self._get_state_locked() != AssistantState.ERROR:
            return
        self._set_state(AssistantState.IDLE, self._idle_caption(), "")

    def _enter_error_state(self, headline: str, exc: Exception | str) -> None:
        detail = str(exc).strip() if not isinstance(exc, str) else exc
        if len(detail) > 140:
            detail = detail[:137] + "..."
        hint = "Tap the face to try again, or press R to reset."
        sub = f"{detail}  {hint}" if detail else hint
        self._set_state(AssistantState.ERROR, headline, sub)

    def on_enter_from_idle(self) -> None:
        st = self._get_state_locked()
        if st == AssistantState.ERROR:
            self.dismiss_error()
            self._begin_listening()
            return
        if st != AssistantState.IDLE:
            return
        self._begin_listening()

    def on_tap(self) -> None:
        st = self._get_state_locked()
        if st == AssistantState.ERROR:
            self.dismiss_error()
            self._begin_listening()
            return
        if st == AssistantState.IDLE:
            self._begin_listening()
        elif st == AssistantState.LISTENING:
            self._cancel_listening()
        elif st == AssistantState.SPEAKING:
            self._cancel_play.set()

    def forget_last_exchange(self) -> None:
        with self._lock:
            cid = self._conversation_id
        if cid:
            n = self.memory.forget_last_exchange(cid)
            log.info("Forgot %d turn(s) from conversation %s", n, cid)

    def _begin_listening(self) -> None:
        self._cancel_record.clear()
        self._cancel_play.clear()
        self.rms = RmsRingBuffer(maxlen=64)
        self._set_state(
            AssistantState.LISTENING,
            "Listening…",
            "Speak when you're ready.",
        )
        idle = timedelta(minutes=self.settings.session_idle_minutes)
        self._conversation_id = self.memory.get_or_create_conversation(idle)

        def job() -> Path:
            return record_until_silence(
                self.settings,
                self.vad,
                rms_callback=self.rms.push,
                cancel_event=self._cancel_record,
            )

        self._record_future = self._executor.submit(job)
        self._record_future.add_done_callback(self._on_record_done)

    def _cancel_listening(self) -> None:
        self._cancel_record.set()
        self._set_state(AssistantState.IDLE, self._idle_caption(), "")

    def _on_record_done(self, fut: Future) -> None:
        if self._get_state_locked() != AssistantState.LISTENING:
            return
        try:
            wav = fut.result()
        except Exception as e:
            if str(e) == "recording_cancelled":
                self._set_state(AssistantState.IDLE, self._idle_caption(), "")
                return
            log.exception("Recording failed: %s", e)
            self._enter_error_state("I couldn't hear you — mic problem.", e)
            return
        self._set_state(
            AssistantState.THINKING,
            "Thinking…",
            "",
        )
        self._start_thinking_fillers()
        pipe = self._executor.submit(self._pipeline, wav)
        pipe.add_done_callback(self._on_pipeline_done)

    def _pipeline(self, wav_path: Path) -> tuple[Path, str, str]:
        user_text = transcribe(self.settings, wav_path)
        if not user_text.strip():
            user_text = "(silence)"
        self.memory.add_turn(self._conversation_id, "user", user_text)

        ctx = self.memory.get_context_messages(
            self._conversation_id, self.settings.context_turns
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_message},
            *ctx,
        ]
        if self.settings.web_search_enabled:
            reply = chat_with_tools(self.settings, messages)
        else:
            reply = chat(self.settings, messages, stream=True)
        if not reply:
            reply = "I didn't catch that."
        self.memory.add_turn(self._conversation_id, "assistant", reply)

        out_wav = synthesize(self.settings, reply)
        try:
            wav_path.unlink(missing_ok=True)
        except OSError:
            pass
        return out_wav, reply, user_text

    def _on_pipeline_done(self, fut: Future) -> None:
        if self._get_state_locked() != AssistantState.THINKING:
            return
        self._stop_thinking_fillers()
        try:
            wav_path, reply, _user_text = fut.result()
        except Exception as e:
            log.exception("Pipeline failed: %s", e)
            self._enter_error_state("Oh no — I'm a bit broken right now.", e)
            return

        env = wav_envelope(wav_path, bins=48)
        speak_dur = 3.0
        try:
            with wave.open(str(wav_path), "rb") as wf:
                frames = wf.getnframes()
                sr = wf.getframerate() or 22050
                speak_dur = frames / float(sr)
        except Exception:
            pass

        preview = reply.replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:117] + "..."
        self._set_state(
            AssistantState.SPEAKING,
            "Speaking...",
            preview,
            envelope=env,
            speak_duration=speak_dur,
        )
        self._cancel_play.clear()

        def play_job() -> None:
            play_wav(wav_path, self.settings, cancel_event=self._cancel_play)

        play_fut = self._executor.submit(play_job)
        play_fut.add_done_callback(lambda f: self._on_play_done(f, wav_path))

    def _on_play_done(self, fut: Future, wav_path: Path) -> None:
        err = fut.exception()
        try:
            wav_path.unlink(missing_ok=True)
        except OSError:
            pass
        if err and str(err) != "playback_cancelled":
            log.error("Playback error: %s", err)
            self._enter_error_state("Couldn't play my reply — speaker problem?", err)
            return
        self._set_state(AssistantState.IDLE, self._idle_caption(), "")
