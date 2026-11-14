import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Protocol

from .config import Config
from .events import Event, EventQueue, Role
from .memory import Memory
from .tools import Tools, TurnContext

log = logging.getLogger(__name__)

COMPACTION_NOTICE = (
    "Служебное сообщение: начало истории скоро будет забыто. Сохрани через memory_write всё, "
    "что стоит помнить дальше. В чат ничего не пиши, закончи ход через stay_silent."
)


SCREENSHOT_CAPTION = "Снимок экрана стрима от look_at_screen, сделан в {time}:"
# В истории живёт только последний снимок и недолго: каждый запрос с ним
# стоит пару тысяч токенов, пусть и из кэша. В history.json он не попадает.
SCREENSHOT_GONE = "[здесь был снимок экрана стрима, картинка уже удалена]"


class LLM(Protocol):
    async def complete(self, messages: list[dict], tools: list[dict]) -> tuple[dict, int]: ...


def build_system_prompt(prompts_dir: Path, memory: Memory) -> str:
    system = (prompts_dir / "system.md").read_text(encoding="utf-8").strip()
    people = (prompts_dir / "people.md").read_text(encoding="utf-8").strip()
    return f"{system}\n\n{people}\n\n# Индекс памяти\n\n{memory.index()}"


class Agent:
    """Одна общая история на все источники: чат, консоль, голос, а потом донаты."""

    def __init__(self, llm: LLM, tools: Tools, memory: Memory, config: Config):
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.config = config
        self.history_path = config.data_dir / "history.json"
        self.history: list[dict] = self._load_history()
        # Системный промпт собирается редко: индекс памяти внутри него нельзя
        # обновлять на каждый ход, иначе кэш токенов будет сбрасываться.
        self.system_prompt = self._build_system_prompt()
        self.last_prompt_tokens = 0
        # Когда сделан снимок, который сейчас лежит в истории.
        self.screenshot_taken: float | None = None
        self.clock = time.monotonic

    def _build_system_prompt(self) -> str:
        return build_system_prompt(self.config.prompts_dir, self.memory)

    def _load_history(self) -> list[dict]:
        if self.history_path.exists():
            return json.loads(self.history_path.read_text(encoding="utf-8"))
        return []

    def _save_history(self) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.history_path.with_suffix(".tmp")
        saved = [
            {**message, "content": SCREENSHOT_GONE} if _is_screenshot(message) else message
            for message in self.history
        ]
        temporary.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, self.history_path)

    async def run(self, queue: EventQueue) -> None:
        while True:
            events = await queue.next_batch()
            try:
                await self.handle(events)
            except Exception:
                # События уже в истории: модель увидит их на следующем ходу.
                log.exception("Ход агента упал")

    async def handle(self, events: list[Event]) -> None:
        # Права хода берутся по самому слабому участнику пачки. Автор с правами
        # попадает в пачку только со своими же сообщениями, см. EventQueue.
        role = min(event.role for event in events)
        if len({event.user for event in events}) == 1:
            ctx = TurnContext(actor=events[0].user, role=role)
        else:
            ctx = TurnContext(actor="чат", role=role)
        await self._turn("\n".join(event.to_line() for event in events), ctx)
        if self.last_prompt_tokens >= self.config.history_high_tokens:
            await self._compact()

    async def _turn(self, content: str, ctx: TurnContext) -> None:
        if (
            self.screenshot_taken is not None
            and self.clock() - self.screenshot_taken > self.config.screenshot_ttl_seconds
        ):
            # Удаление сбрасывает кэш с места картинки, но через 5 минут
            # кэш обычно уже истёк сам.
            self._drop_screenshot()
        self.history.append({"role": "user", "content": content})
        done: set[tuple[str, str]] = set()
        try:
            for _ in range(self.config.max_tool_rounds):
                messages = [{"role": "system", "content": self.system_prompt}, *self.history]
                message, self.last_prompt_tokens = await self.llm.complete(messages, self.tools.schemas)
                self.history.append(message)
                calls = message.get("tool_calls") or []
                if not calls:
                    break
                finished = False
                images: list[str] = []
                for call in calls:
                    key = (call["function"]["name"], call["function"]["arguments"])
                    if key in done:
                        # Модель иногда повторяет уже сделанное действие вместо завершения хода.
                        text = "уже выполнено в этом ходе, не повторяй. Заверши ход: say или stay_silent"
                    else:
                        result = await self.tools.call(*key, ctx)
                        text = result.text
                        finished = finished or result.ends_turn
                        if result.image:
                            images.append(result.image)
                        if not text.startswith(("ошибка", "отказано")):
                            done.add(key)
                    self.history.append({"role": "tool", "tool_call_id": call["id"], "content": text})
                if images:
                    # В ответе инструмента картинку роутер молча выбрасывает,
                    # модель видит её только в сообщении пользователя.
                    self._drop_screenshot()
                    caption = SCREENSHOT_CAPTION.format(time=datetime.now().strftime("%H:%M:%S"))
                    self.history.append({"role": "user", "content": [
                        {"type": "text", "text": caption},
                        {"type": "image_url", "image_url": {"url": images[-1]}},
                    ]})
                    self.screenshot_taken = self.clock()
                if finished:
                    break
        finally:
            self._save_history()

    def _drop_screenshot(self) -> None:
        for message in self.history:
            if _is_screenshot(message):
                message["content"] = SCREENSHOT_GONE
        self.screenshot_taken = None

    async def _compact(self) -> None:
        log.info("История дошла до %s токенов, обрезаю", self.last_prompt_tokens)
        ctx = TurnContext(actor="шери", role=Role.BROADCASTER, maintenance=True)
        try:
            await self._turn(COMPACTION_NOTICE, ctx)
        except Exception:
            log.exception("Не удалось сохранить память перед обрезкой")
        # Обрезка всё равно сбрасывает кэш, а base64 картинки исказил бы её подсчёт размеров.
        self._drop_screenshot()
        self._trim()
        self.system_prompt = self._build_system_prompt()
        self.last_prompt_tokens = 0
        self._save_history()

    def _trim(self) -> None:
        """Оставляет хвост нужного размера. Резать можно только по сообщению
        пользователя, чтобы не разорвать пару tool_call и её результат."""
        sizes = [len(json.dumps(message, ensure_ascii=False)) for message in self.history]
        budget = sum(sizes) * self.config.history_low_tokens / self.config.history_high_tokens
        cut = None
        kept = 0
        for index in range(len(self.history) - 1, -1, -1):
            kept += sizes[index]
            if kept > budget and cut is not None:
                break
            if self.history[index]["role"] == "user":
                cut = index
        if cut:
            del self.history[:cut]


def _is_screenshot(message: dict) -> bool:
    return message["role"] == "user" and isinstance(message.get("content"), list)
