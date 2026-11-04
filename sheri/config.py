import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    api_key: str = ""
    base_url: str = "https://routerai.ru/api/v1"
    model: str = "openai/gpt-6-luna"
    reasoning_effort: str = "low"
    max_completion_tokens: int = 1000
    # При достижении high история одним шагом обрезается примерно до low.
    # Обрезать по чуть-чуть нельзя: каждый сдвиг начала истории сбрасывает кэш.
    history_high_tokens: int = 30_000
    history_low_tokens: int = 15_000
    # Пачка ждёт следующее сообщение debounce секунд, но в сумме не дольше max_wait.
    debounce_seconds: float = 3.0
    batch_max_wait_seconds: float = 8.0
    max_batch: int = 20
    max_tool_rounds: int = 5
    prompts_dir: Path = ROOT / "prompts"
    data_dir: Path = ROOT / "data"
    twitch_client_id: str = ""
    # Голос: сервер распознавания слушает клиент, который снимает микрофон.
    voice_host: str = "0.0.0.0"
    voice_port: int = 8765
    voice_token: str = ""
    voice_speaker: str = "Mo0nlaght"
    voice_context_seconds: float = 90.0
    voice_followup_seconds: float = 20.0
    stt_model: str = "v3_e2e_rnnt"


def load_config() -> Config:
    load_dotenv(ROOT / ".env")
    return Config(
        api_key=os.getenv("ROUTERAI_API_KEY", ""),
        model=os.getenv("SHERI_MODEL", Config.model),
        reasoning_effort=os.getenv("SHERI_REASONING_EFFORT", Config.reasoning_effort),
        twitch_client_id=os.getenv("TWITCH_CLIENT_ID", ""),
        voice_port=int(os.getenv("VOICE_PORT", Config.voice_port)),
        voice_token=os.getenv("VOICE_TOKEN", ""),
        stt_model=os.getenv("STT_MODEL", Config.stt_model),
    )
