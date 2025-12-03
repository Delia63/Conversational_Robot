from __future__ import annotations
import time
from rapidfuzz import fuzz
from typing import Any, Optional, Dict
from src.utils.textnorm import normalize_text

class FastExit:
    """
    Detectează pe *parțiale ASR* expresii de tip "ok bye"/"pa" și oprește imediat
    TTS + LLM stream, apoi trece în STANDBY. Non-intruziv, cu debounce și (opțional)
    verificare că vorbește userul (nu ecoul TTS) via barge listener.
    """

    def __init__(
        self,
        tts: Any,
        llm: Any,
        state: Any,
        logger: Any,
        cfg: dict,
        barge: Optional[Any] = None,
    ):
        self.tts = tts
        self.llm = llm
        self.state = state
        self.log = logger

        self.enabled = bool(cfg.get("enabled", True))
        self.phrases = [p.lower() for p in cfg.get("phrases", [])]
        self.fuzzy = int(cfg.get("fuzzy", 90))
        self.debounce_ms = int(cfg.get("debounce_ms", 120))
        self.min_chars = int(cfg.get("min_chars", 2))
        self.confirm_cfg = cfg.get("confirm_tts", "See you later!")
        if isinstance(self.confirm_cfg, dict):
            self.confirm_default = self.confirm_cfg.get("default") or self.confirm_cfg.get("en") or next(iter(self.confirm_cfg.values()), "")
        else:
            self.confirm_default = str(self.confirm_cfg or "")
            self.confirm_cfg = None
        raw_langs: Dict[str, str] = {}
        for k, v in (cfg.get("phrase_langs") or {}).items():
            try:
                nk = normalize_text(k).lower().strip()
                if nk:
                    raw_langs[nk] = (v or "").lower().strip() or "en"
            except Exception:
                continue
        self.phrase_langs = raw_langs
        self.use_barge = bool(cfg.get("use_barge_check", True))
        self.barge = barge

        self._last_hit_ms = 0
        self._aborted = False

    # ——— API simplu pentru orchestrator ———
    def reset(self):
        self._aborted = False

    def pending(self) -> bool:
        """Dacă e True, orice stream TTS/LLM ar trebui să se oprească."""
        return self._aborted

    # ——— Callbacks ———
    def on_partial(self, text: str) -> bool:
        """
        Întoarce True dacă a declanșat exitul (evenimentul a fost "consumat"),
        altfel False (lasă pipeline-ul să continue normal).
        """
        if not self.enabled or not text:
            return False

        now = time.time() * 1000
        if (now - self._last_hit_ms) < self.debounce_ms:
            return False

        s = normalize_text(text).lower().strip()
        if len(s) < self.min_chars:
            return False

        # Evită trigger pe ecou dacă ai barge-in detector (near-end only)
        if self.use_barge and self.barge and hasattr(self.barge, "user_is_speaking"):
            try:
                if not self.barge.user_is_speaking():
                    return False
            except Exception:
                pass

        for target in self.phrases:
            if s == target or fuzz.ratio(s, target) >= self.fuzzy:
                self._last_hit_ms = now
                self._trigger_exit(s)
                return True
        return False

    def on_final(self, text: str) -> bool:
        """Reciclează aceeași logică și pe transcriptul final (fallback)."""
        return self.on_partial(text)

    # ——— Poți apela manual pentru test ———
    def trigger_exit(self, reason: str = "manual"):
        self._trigger_exit(reason)

    # ——— Intern ———
    def _trigger_exit(self, matched: str):
        if self._aborted:
            return
        self._aborted = True
        try:
            self.log.info(f"[FAST_EXIT] Triggered by: '{matched}'")
        except Exception:
            pass

        # 1) Oprește orice audio/generare
        for obj in (self.tts, self.llm):
            for fn_name in ("stop", "cancel_stream", "cancel", "abort"):
                fn = getattr(obj, fn_name, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception as e:
                        try:
                            self.log.warning(f"[FAST_EXIT] {fn_name} error: {e}")
                        except Exception:
                            pass

        # 2) Feedback auditiv minimal (opțional)
        confirm_msg = self._select_confirm_message(matched)
        if confirm_msg and hasattr(self.tts, "say") and callable(self.tts.say):
            try:
                self.tts.say(confirm_msg)
            except Exception:
                pass

        # 3) Treci în STANDBY
        # Preferă un setter dedicat; altfel încearcă fallback.
        for setter in ("set_standby", "to_standby", "go_standby"):
            s = getattr(self.state, setter, None)
            if callable(s):
                try:
                    s()
                    return
                except Exception:
                    pass
        # Fallback hard dacă folosești enum BotState
        try:
            from src.core.states import BotState
            self.state = BotState.STANDBY
        except Exception:
            pass

    def _select_confirm_message(self, matched: str) -> str:
        if isinstance(self.confirm_cfg, dict):
            lang = self._lang_for_phrase(matched)
            if lang and lang in self.confirm_cfg:
                return self.confirm_cfg[lang]
            return self.confirm_default
        return self.confirm_default

    def _lang_for_phrase(self, matched: str) -> Optional[str]:
        try:
            norm = normalize_text(matched or "").lower().strip()
        except Exception:
            norm = ""
        if norm and norm in self.phrase_langs:
            return self.phrase_langs[norm]
        return None
