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


def load_config() -> Config:
    load_dotenv(ROOT / ".env")
    return Config(
        api_key=os.getenv("ROUTERAI_API_KEY", ""),
        model=os.getenv("SHERI_MODEL", Config.model),
        reasoning_effort=os.getenv("SHERI_REASONING_EFFORT", Config.reasoning_effort),
        twitch_client_id=os.getenv("TWITCH_CLIENT_ID", ""),
    )
