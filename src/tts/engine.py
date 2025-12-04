# src/tts/engine.py
from __future__ import annotations
from typing import Dict, Optional, Iterable, Callable
import threading, re, os, shutil, subprocess, tempfile, time, queue
import soundfile as sf
import sounddevice as sd

from src.telemetry.metrics import tts_speak_calls

_SENT_SPLIT = re.compile(r'([.!?…:;]+)\s+')

# -------------------- PYTTSX3 BACKEND --------------------
class _Pyttsx3TTS:
    def __init__(self, cfg: Dict, logger):
        import pyttsx3
        self.log = logger
        self.eng = pyttsx3.init()
        self.rate = int(cfg.get("rate", 170))
        self.volume = float(cfg.get("volume", 1.0))
        self.voice_ro_hint = cfg.get("voice_ro_hint", "ro")
        self.voice_en_hint = cfg.get("voice_en_hint", "en")
        self.eng.setProperty("rate", self.rate)
        self.eng.setProperty("volume", self.volume)
        self._voices = self.eng.getProperty("voices")

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._speaking = threading.Event()
        self._speak_th: Optional[threading.Thread] = None

    def _pick_voice(self, lang: str) -> Optional[str]:
        target = (self.voice_ro_hint if lang.startswith("ro") else self.voice_en_hint or "").lower()
        for v in self._voices:
            name = (getattr(v, "name", "") or "").lower()
            _id  = (getattr(v, "id", "") or "").lower()
            if target and (target in name or target in _id):
                return v.id
        return self._voices[0].id if self._voices else None

    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def say(self, text: str, lang: str = "en"):
        vid = self._pick_voice(lang)
        if vid: self.eng.setProperty("voice", vid)
        else:   self.log.warning("⚠️ Nicio voce potrivită (pyttsx3) – folosesc default.")
        tts_speak_calls.inc()
        self._speaking.set()
        try:
            self.eng.say(text)
            self.eng.runAndWait()
        finally:
            self._speaking.clear()

    def say_async_stream(
        self,
        token_iter: Iterable[str],
        lang: str = "en",
        on_first_speak: Optional[Callable[[], None]] = None,
        min_chunk_chars: int = 80,
        on_done: Optional[Callable[[], None]] = None,
    ):
        def worker():
            first_spoken = False
            buf = ""
            vid = self._pick_voice(lang)
            if vid: self.eng.setProperty("voice", vid)
            tts_speak_calls.inc()
            self._speaking.set()
            try:
                for tok in token_iter:
                    if self._stop.is_set():
                        break
                    buf += tok

                    parts = _SENT_SPLIT.split(buf)
                    out = []
                    if len(parts) >= 2:
                        for i in range(0, len(parts)-1, 2):
                            frag, punct = parts[i], parts[i+1]
                            s = (frag + punct).strip()
                            if s: out.append(s)
                        buf = parts[-1] if (len(parts) % 2 == 1) else ""

                    if not out and len(buf) >= min_chunk_chars:
                        last_space = buf.rfind(" ")
                        if last_space > 20:
                            out.append(buf[:last_space].strip())
                            buf = buf[last_space+1:]

                    for sentence in out:
                        if self._stop.is_set():
                            break
                        if on_first_speak and not first_spoken:
                            first_spoken = True
                            try: on_first_speak()
                            except Exception: pass
                        self.eng.say(sentence)
                        self.eng.runAndWait()

                if not self._stop.is_set() and buf.strip():
                    if on_first_speak and not first_spoken:
                        first_spoken = True
                        try: on_first_speak()
                        except Exception: pass
                    self.eng.say(buf.strip())
                    self.eng.runAndWait()
            except Exception as e:
                self.log.error(f"TTS stream error (pyttsx3): {e}")
            finally:
                self._speaking.clear()
                if on_done:
                    try: on_done()
                    except Exception: pass

        self.stop()
        self._stop.clear()
        self._speak_th = threading.Thread(target=worker, daemon=True)
        self._speak_th.start()
        return self._speaking

    def stop(self):
        with self._lock:
            self._stop.set()
            try: self.eng.stop()
            except Exception: pass
        self._speaking.clear()

# -------------------- PIPER (CLI) BACKEND — DOUBLE BUFFER --------------------
class _PiperCmdTTS:
    """
    Piper backend cu dublu-buffer:
      - Producer-ul segmentează stream-ul LLM în propoziții/bucăți, sintetizează WAV-urile următoare
        și le pune într-o coadă cu max 2 elemente (A/B).
      - Consumer-ul redă în timp real fișierul curent, în timp ce următorul e deja prefăcut.
      - Loguri:
          🧠  LLM→TTS chunk: <text>   (înainte de sinteză)
          🔊  TTS play start: <N>     (când începe redarea)
    """
    def __init__(self, cfg: Dict, logger):
        self.log = logger
        self.cfg = cfg or {}
        self.p = self.cfg.get("piper") or {}
        self.exe = self.p.get("exe") or shutil.which("piper")
        self.model_ro = self.p.get("model_ro")
        self.config_ro = self.p.get("config_ro")
        self.model_en = self.p.get("model_en")
        self.config_en = self.p.get("config_en")
        self.speaker_id = self.p.get("speaker_id", None)
        self.length_scale = float(self.p.get("length_scale", 1.0))
        self.noise_scale = float(self.p.get("noise_scale", 0.667))
        self.noise_w = float(self.p.get("noise_w", 0.8))
        self.sentence_silence_ms = int(self.p.get("sentence_silence_ms", 80))
        self.warmup_enabled = bool(self.p.get("warmup_enabled", True))
        self.warmup_text = (self.p.get("warmup_text") or "").strip()
        self.warmup_lang = (self.p.get("warmup_lang") or "en").lower()

        # Control
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._speaking = threading.Event()
        self._warmup_lock = threading.Lock()
        self._warmed_up = False

        # Double buffer queue (A/B)
        self._q: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=2)
        self._producer_th: Optional[threading.Thread] = None
        self._consumer_th: Optional[threading.Thread] = None
        self._coord_th: Optional[threading.Thread] = None
        self._play_proc: Optional[subprocess.Popen] = None
        self._staged_paths: set[str] = set()

        if not self.exe or not os.path.exists(self.exe):
            raise RuntimeError("Piper executable not found. Set tts.piper.exe or install piper-tts.")
        self._ensure_warm()

    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def _pick_model(self, lang: str):
        lang = (lang or "").lower()
        if lang.startswith("ro"):
            return self.model_ro, self.config_ro
        return self.model_en, self.config_en

    def _resolve_warmup_lang(self, lang_hint: Optional[str]) -> str:
        lang = (self.warmup_lang or lang_hint or "en").lower()
        if lang.startswith("ro"):
            return "ro"
        return "en"

    def _ensure_warm(self, lang_hint: Optional[str] = None):
        if not self.warmup_enabled or self._warmed_up:
            return
        text = (self.warmup_text or "").strip()
        if not text:
            self.log.info("Piper warm-up: text gol, sar peste.")
            self._warmed_up = True
            return
        with self._warmup_lock:
            if not self.warmup_enabled or self._warmed_up:
                return
            lang = self._resolve_warmup_lang(lang_hint)
            try:
                self.log.info(f"🔥 Piper warm-up start (lang={lang})")
                wav = self._synth_to_wav(text, lang)
                try:
                    os.remove(wav)
                except Exception:
                    pass
                self._warmed_up = True
                self.log.info("✅ Piper warm-up gata")
            except Exception as e:
                self.log.warning(f"Piper warm-up eșuat: {e}")

    def _synth_to_wav(self, text: str, lang: str) -> str:
        model, cfg = self._pick_model(lang)
        if not (model and os.path.exists(model)):
            raise RuntimeError("Piper model not set/found for selected language.")
        fd, path = tempfile.mkstemp(prefix=f"piper_{lang}_", suffix=".wav")
        os.close(fd)

        cmd = [self.exe, "--model", model, "--output_file", path]
        if cfg and os.path.exists(cfg):
            cmd += ["--config", cfg]
        if self.speaker_id is not None:
            cmd += ["--speaker", str(self.speaker_id)]
        # length/noise se pot lăsa în .json dacă binarul nu suportă flag-urile

        try:
            subprocess.run(cmd, input=text.encode("utf-8"), check=True)
            return path
        except subprocess.CalledProcessError as e:
            self.log.error(f"Piper synth failed: {e}")
            raise

    def _play_wav(self, wav_path: str):
        # 1) paplay (PulseAudio/PipeWire)
        player = shutil.which("paplay")
        if player:
            self._play_proc = subprocess.Popen([player, wav_path])
            while self._play_proc.poll() is None:
                if self._stop.is_set():
                    self._play_proc.terminate()
                    break
                time.sleep(0.02)
            return

        # 2) aplay (ALSA)
        player = shutil.which("aplay")
        if player:
            self._play_proc = subprocess.Popen([player, "-q", wav_path])
            while self._play_proc.poll() is None:
                if self._stop.is_set():
                    self._play_proc.terminate()
                    break
                time.sleep(0.02)
            return

        # 3) fallback Python (sounddevice)
        try:
            data, sr = sf.read(wav_path, dtype="float32")
            sd.play(data, sr)
            while sd.get_stream() and sd.get_stream().active:
                if self._stop.is_set():
                    sd.stop()
                    break
                time.sleep(0.02)
            sd.wait()
        except Exception as e:
            self.log.error(f"Audio playback error: {e}")

    # ---------- FIX: producer robust + sentinel garantat ----------
    def _producer(self, token_iter: Iterable[str], lang: str, min_chunk_chars: int):
        try:
            buf = ""
            for tok in token_iter:
                if self._stop.is_set():
                    break
                buf += tok

                parts = _SENT_SPLIT.split(buf)
                out = []
                if len(parts) >= 2:
                    for i in range(0, len(parts) - 1, 2):
                        frag, punct = parts[i], parts[i + 1]
                        s = (frag + punct).strip()
                        if s:
                            out.append(s)
                    buf = parts[-1] if (len(parts) % 2 == 1) else ""

                if not out and len(buf) >= min_chunk_chars:
                    last_space = buf.rfind(" ")
                    if last_space > 20:
                        out.append(buf[:last_space].strip())
                        buf = buf[last_space + 1:]

                for s in out:
                    if self._stop.is_set():
                        break
                    self.log.info(f"🧠 LLM→TTS chunk [{len(s)}c]: {s}")
                    wav = self._synth_to_wav(s, lang)
                    self._staged_paths.add(wav)
                    while not self._stop.is_set():
                        try:
                            self._q.put(wav, timeout=0.1)
                            break
                        except queue.Full:
                            continue

            tail = buf.strip()
            if (not self._stop.is_set()) and tail:
                self.log.info(f"🧠 LLM→TTS chunk [{len(tail)}c]: {tail}")
                wav = self._synth_to_wav(tail, lang)
                self._staged_paths.add(wav)
                while not self._stop.is_set():
                    try:
                        self._q.put(wav, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except Exception as e:
            self.log.error(f"Piper producer error: {e}")
        finally:
            # Sentinel garantat: livrează None chiar dacă coada e plină
            while not self._stop.is_set():
                try:
                    self._q.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def _consumer(self, on_first_speak: Optional[Callable[[], None]]):
        first = True
        n = 0
        try:
            while not self._stop.is_set():
                try:
                    item = self._q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    break
                wav = item
                n += 1
                if on_first_speak and first:
                    first = False
                    try:
                        on_first_speak()
                    except Exception:
                        pass
                self.log.info(f"🔊 TTS play start (chunk {n})")
                try:
                    self._play_wav(wav)
                finally:
                    try:
                        if wav in self._staged_paths:
                            os.remove(wav)
                            self._staged_paths.discard(wav)
                    except Exception:
                        pass

                # mic gap între bucăți, dacă e configurat
                if self.sentence_silence_ms > 0 and not self._stop.is_set():
                    t0 = time.time()
                    while (time.time() - t0) * 1000 < self.sentence_silence_ms and not self._stop.is_set():
                        time.sleep(0.003)
        except Exception as e:
            self.log.error(f"Piper consumer error: {e}")

    def say(self, text: str, lang: str = "en"):
        """Sinteză blocking pe propoziții (fără stream din LLM)."""
        self._ensure_warm(lang)
        tts_speak_calls.inc()
        self._speaking.set()
        try:
            parts = _SENT_SPLIT.split(text)
            sentences = []
            if len(parts) >= 2:
                for i in range(0, len(parts)-1, 2):
                    frag, punct = parts[i], parts[i+1]
                    s = (frag + punct).strip()
                    if s: sentences.append(s)
                tail = parts[-1].strip() if (len(parts) % 2 == 1) else ""
                if tail: sentences.append(tail)
            else:
                if text.strip():
                    sentences = [text.strip()]

            for s in sentences:
                if self._stop.is_set(): break
                self.log.info(f"🧠 LLM→TTS chunk [{len(s)}c]: {s}")
                wav = self._synth_to_wav(s, lang)
                try:
                    self.log.info("🔊 TTS play start (blocking)")
                    self._play_wav(wav)
                finally:
                    try: os.remove(wav)
                    except Exception: pass
                if self.sentence_silence_ms > 0:
                    t0 = time.time()
                    while (time.time() - t0) * 1000 < self.sentence_silence_ms and not self._stop.is_set():
                        time.sleep(0.003)
        finally:
            self._speaking.clear()

    # ---------- FIX: coordonatorul asigură oprirea corectă + revenire în listen ----------
    def say_async_stream(
        self,
        token_iter: Iterable[str],
        lang: str = "en",
        on_first_speak: Optional[Callable[[], None]] = None,
        min_chunk_chars: int = 80,
        on_done: Optional[Callable[[], None]] = None,
    ):
        self._ensure_warm(lang)
        def coordinator():
            try:
                self._speaking.set()
                tts_speak_calls.inc()

                # Pornește producer + consumer
                self._producer_th = threading.Thread(
                    target=self._producer,
                    args=(token_iter, lang, int(min_chunk_chars)),
                    daemon=True,
                )
                self._consumer_th = threading.Thread(
                    target=self._consumer,
                    args=(on_first_speak,),
                    daemon=True,
                )
                self._producer_th.start()
                self._consumer_th.start()

                # Așteaptă producer-ul, apoi injectează sentinel dacă mai e nevoie
                self._producer_th.join()
                while self._consumer_th.is_alive() and not self._stop.is_set():
                    try:
                        self._q.put(None, timeout=0.1)
                        break
                    except queue.Full:
                        time.sleep(0.05)
                        continue

                self._consumer_th.join()
            finally:
                self._speaking.clear()
                if on_done:
                    try: on_done()
                    except Exception: pass

        # reset pipeline
        self.stop()
        self._stop.clear()
        self._q = queue.Queue(maxsize=2)

        self._coord_th = threading.Thread(target=coordinator, daemon=True)
        self._coord_th.start()
        return self._speaking

    def stop(self):
        with self._lock:
            self._stop.set()
            try:
                if self._play_proc and self._play_proc.poll() is None:
                    self._play_proc.terminate()
            except Exception:
                pass
        # șterge WAV-urile neconsumate
        for p in list(self._staged_paths):
            try:
                os.remove(p)
            except Exception:
                pass
            self._staged_paths.discard(p)
        self._speaking.clear()


# -------------------- FACADE --------------------
class TTSLocal:
    """
    Alege backend-ul în funcție de configs/tts.yaml:
      - backend: piper  -> _PiperCmdTTS (cu dublu-buffer)
      - altfel         -> _Pyttsx3TTS (fallback)
    """
    def __init__(self, cfg: Dict, logger):
        self.log = logger
        backend = (cfg.get("backend") or "pyttsx3").lower()
        try:
            if backend == "piper":
                self.impl = _PiperCmdTTS(cfg, logger)
                self.log.info("TTS backend: Piper (double-buffer)")
            else:
                raise RuntimeError("force pyttsx3")
        except Exception as e:
            self.log.warning(f"Piper indisponibil ({e}). Revin pe pyttsx3.")
            self.impl = _Pyttsx3TTS(cfg, logger)
            self.log.info("TTS backend: pyttsx3")

    def is_speaking(self) -> bool:
        return self.impl.is_speaking()

    def say(self, text: str, lang: str = "en"):
        return self.impl.say(text, lang)

    def say_async_stream(
        self,
        token_iter: Iterable[str],
        lang: str = "en",
        on_first_speak: Optional[Callable[[], None]] = None,
        min_chunk_chars: int = 80,
        on_done: Optional[Callable[[], None]] = None,
    ):
        return self.impl.say_async_stream(token_iter, lang, on_first_speak, min_chunk_chars, on_done)

    def stop(self):
        return self.impl.stop()
