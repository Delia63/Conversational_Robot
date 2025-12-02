from __future__ import annotations

import os
import queue
import struct
import tempfile
import threading
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from rapidfuzz import fuzz

from src.audio.devices import choose_input_device
from src.audio.vad import VAD
from src.utils.textnorm import normalize_text


def _float_to_int16(audio_f32: np.ndarray) -> np.ndarray:
    audio_f32 = np.clip(audio_f32, -1.0, 1.0)
    return (audio_f32 * 32767.0).astype(np.int16)


class TextStopListener:
    """
    Background listener that captures short utterances while TTS is speaking,
    transcribes them via the main ASR engine, and fires a callback when
    fuzzy-matching phrases (e.g., "stop robot") are detected.
    """

    def __init__(
        self,
        cfg_audio: dict,
        asr_engine,
        phrases_norm: Iterable[str],
        fuzzy_threshold: int,
        logger=None,
        on_detect: Optional[Callable[[str], None]] = None,
    ):
        self.cfg_audio = cfg_audio or {}
        self.asr = asr_engine
        self.logger = logger
        self.on_detect = on_detect
        self.sample_rate = int(self.cfg_audio.get("sample_rate", 16000))
        self.block_ms = int(self.cfg_audio.get("block_ms", 20))
        self.block_size = int(self.sample_rate * (self.block_ms / 1000.0))
        self.vad = VAD(self.sample_rate, self.cfg_audio.get("vad_aggressiveness", 3), self.block_ms)
        self.phrases_norm: List[str] = [normalize_text(p) for p in phrases_norm if p]
        self.fuzzy = max(0, min(100, int(fuzzy_threshold or 90)))
        self.min_voice_ms = int(self.cfg_audio.get("stop_text_min_voice_ms", 200))
        self.silence_ms_to_end = int(self.cfg_audio.get("stop_text_silence_ms", 350))
        self.max_record_ms = int(self.cfg_audio.get("stop_text_max_ms", 1800))
        self.trigger_after_ms = max(
            self.min_voice_ms,
            int(self.cfg_audio.get("stop_text_trigger_ms", self.min_voice_ms + 250)),
        )
        self.overlap_ms = max(0, int(self.cfg_audio.get("stop_text_overlap_ms", 180)))
        self.debug_log = bool(self.cfg_audio.get("stop_text_debug", False))
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)
        self._stream: Optional[sd.InputStream] = None
        self._active = False
        self._buffer: List[np.ndarray] = []
        self._voiced_ms = 0
        self._silence_ms = 0

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._open_stream()
        self._thread = threading.Thread(target=self._run, name="TextStopListener", daemon=True)
        self._thread.start()
        if self.logger:
            self.logger.debug("TextStopListener started (ASR stop phrases).")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.5)
            self._thread = None
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        self._stream = None
        self._buffer.clear()
        self._active = False
        if self.logger:
            self.logger.debug("TextStopListener stopped.")

    def _open_stream(self):
        dev_index = choose_input_device(
            prefer_echo_cancel=bool(self.cfg_audio.get("prefer_echo_cancel", True)),
            hint=str(self.cfg_audio.get("input_device_hint", "") or ""),
            logger=self.logger,
        )

        def _callback(indata, frames, time_info, status):
            if self._stop.is_set():
                return
            if status and self.logger:
                self.logger.debug(f"TextStopListener input status: {status}")
            try:
                self._queue.put_nowait(indata.copy())
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(indata.copy())
                except queue.Full:
                    pass

        self._stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype="float32",
            callback=_callback,
            device=dev_index,
        )
        self._stream.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                block = self._queue.get(timeout=0.3)
            except queue.Empty:
                continue
            pcm_i16 = _float_to_int16(block[:, 0])
            pcm_bytes = struct.pack("<%dh" % len(pcm_i16), *pcm_i16)
            if self.vad.is_speech(pcm_bytes):
                self._buffer.append(pcm_i16)
                self._voiced_ms += self.block_ms
                self._silence_ms = 0
                self._active = True
                if self._voiced_ms >= self.trigger_after_ms:
                    self._finalize_buffer(force=True)
                if self._voiced_ms >= self.max_record_ms:
                    self._finalize_buffer(force=True)
            else:
                if self._active:
                    self._buffer.append(pcm_i16)
                    self._silence_ms += self.block_ms
                    if self._silence_ms >= self.silence_ms_to_end:
                        self._finalize_buffer()
                else:
                    self._buffer = []
                    self._voiced_ms = 0

    def _finalize_buffer(self, force: bool = False):
        if not self._buffer:
            return
        total_ms = self._voiced_ms
        if total_ms < self.min_voice_ms:
            return
        audio = np.concatenate(self._buffer, axis=0)
        keep_tail = None
        keep_tail_ms = 0
        if force and self.overlap_ms > 0:
            keep_tail_ms = min(self.overlap_ms, total_ms)
            if keep_tail_ms > 0:
                tail_samples = int(self.sample_rate * (keep_tail_ms / 1000.0))
                if tail_samples > 0:
                    keep_tail = audio[-tail_samples:].copy()
        self._buffer = []
        self._voiced_ms = 0
        self._silence_ms = 0
        self._active = False
        try:
            with tempfile.NamedTemporaryFile(prefix="stop_phrase_", suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio, self.sample_rate, subtype="PCM_16")
                tmp_path = tmp.name
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Stop text: cannot write temp wav ({e})")
            return
        try:
            result = {}
            try:
                result = self.asr.transcribe(tmp_path, language_override="en")
            except Exception:
                result = self.asr.transcribe(tmp_path)
            text = (result.get("text") or "").strip()
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        norm = normalize_text(text)
        if not norm:
            return
        best_phrase = ""
        best_ratio = 0.0
        matched = False
        for phrase in self.phrases_norm:
            if not phrase:
                continue
            ratio = fuzz.ratio(norm, phrase)
            if ratio > best_ratio:
                best_ratio = ratio
                best_phrase = phrase
            if norm == phrase or ratio >= self.fuzzy:
                matched = True
                if self.logger:
                    self.logger.info(f"🛑 Stop via ASR detected: '{text}' (norm='{norm}', target='{phrase}')")
                if callable(self.on_detect):
                    try:
                        self.on_detect(norm)
                    except Exception as cb_err:
                        if self.logger:
                            self.logger.warning(f"Stop text callback error: {cb_err}")
                break
        if self.logger and self.debug_log:
            status = "MATCH" if matched else "no-match"
            self.logger.info(
                f"[StopText] transcript='{text}' norm='{norm}' best='{best_phrase or '-'}' "
                f"ratio={best_ratio:.1f} thr={self.fuzzy} ({status})"
            )
        if force and keep_tail is not None:
            self._buffer.append(keep_tail)
            self._voiced_ms = keep_tail_ms
            self._active = True
