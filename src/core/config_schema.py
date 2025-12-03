# src/core/config_schema.py
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List, Dict, Any, Literal

class AudioCfg(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    sample_rate: int = Field(16000, ge=8000, le=48000)
    block_ms: int = Field(20)
    vad_aggressiveness: int = Field(2, ge=0, le=3)
    silence_ms_to_end: int = Field(500, ge=100, le=5000)
    max_record_seconds: int = Field(15, ge=1, le=300)
    session_idle_seconds: Optional[int] = Field(12, ge=3, le=120)
    min_valid_seconds: Optional[float] = Field(0.5, ge=0.0, le=3.0)

    @field_validator("block_ms")
    @classmethod
    def check_block_ms(cls, v: int):
        if v not in (10, 20, 30):
            raise ValueError("audio.block_ms must be one of: 10, 20, 30")
        return v

class ASRCfg(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    provider: str = Field("faster")                 # faster | openai
    model_size: str = Field("base")
    compute_type: Optional[str] = Field("int8")     # int8 | float16 | int8_float16
    device: str = Field("cpu")                      # cpu | cuda
    beam_size: Optional[int] = Field(1, ge=1, le=8)
    force_language: Optional[str] = None
    vad_min_silence_ms: int = Field(300, ge=100, le=1500)

class LLMCfg(BaseModel):
    provider: str = Field("ollama")
    host: str = Field("http://127.0.0.1:11434")
    model: str = Field("llama3.2")
    max_tokens: int = Field(120, ge=16, le=4096)
    temperature: float = Field(0.4, ge=0.0, le=1.5)
    language_policy: Optional[str] = Field("auto")
    system_prompt: Optional[str] = ""
    default_mode: Optional[str] = Field("precise")
    strict_facts: Optional[bool] = Field(True)

class PiperCfg(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), extra="allow")
    exe: Optional[str] = None
    model_ro: Optional[str] = None
    config_ro: Optional[str] = None
    model_en: Optional[str] = None
    config_en: Optional[str] = None
    speaker_id: Optional[int] = None
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w: float = 0.8
    sentence_silence_ms: int = 80

class TTSCfg(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    backend: str = Field("pyttsx3")
    rate: int = Field(180, ge=60, le=400)
    volume: float = Field(0.9, ge=0.0, le=1.0)
    voice_ro_hint: Optional[str] = Field("ro")
    voice_en_hint: Optional[str] = Field("en")
    piper: Optional[PiperCfg] = None

class PorcupineCfg(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    enabled: bool = False
    access_key: Optional[str] = None
    ppn_path: Optional[str] = None
    sensitivity: float = 0.6
    lang_hint: Optional[str] = Field("auto")  # "auto" | "en" | "ro"

class WakeCfg(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    wake_phrases: List[str]
    acknowledgement: Dict[str, str]
    porcupine: Optional[PorcupineCfg] = None

class RouteCfg(BaseModel):
    rules: List[Dict[str, Any]] = []

class PathsCfg(BaseModel):
    data: str
    models: str

class FastExitCfg(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    enabled: bool = True
    phrases: List[str] = Field(default_factory=lambda: ["ok bye", "ok, bye"])
    fuzzy: int = Field(90, ge=0, le=100)
    debounce_ms: int = Field(120, ge=0, le=2000)
    min_chars: int = Field(2, ge=1, le=16)
    confirm_tts: Optional[Dict[str, str] | str] = None
    use_barge_check: bool = True
    phrase_langs: Dict[str, str] = Field(default_factory=dict)

class AppCfg(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    audio: AudioCfg
    asr: ASRCfg
    llm: LLMCfg
    tts: TTSCfg
    wake: WakeCfg
    route: RouteCfg
    paths: PathsCfg
    fast_exit: Optional[FastExitCfg] = None
    core: Dict[str, Any] = Field(default_factory=dict)

def validate_all(raw: dict) -> dict:
    model = AppCfg(**raw)
    return model.model_dump()
