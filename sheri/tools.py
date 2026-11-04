import json
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

from .events import Role
from .memory import Memory, MemoryError

log = logging.getLogger(__name__)

MAX_CHAT_MESSAGE = 450
MAX_TIMEOUT_SECONDS = 1_209_600
DONE_HINT = "Готово, повторять не нужно: заверши ход через say или stay_silent."


class Channel(Protocol):
    async def send_chat(self, text: str) -> None: ...
    async def timeout_user(self, login: str, seconds: int, reason: str) -> None: ...
    async def ban_user(self, login: str, reason: str) -> None: ...
    async def set_title(self, title: str) -> None: ...
    async def get_channel(self) -> str: ...


class ToolError(Exception):
    pass


@dataclass
class TurnContext:
    actor: str
    role: Role
    # Служебный ход перед обрезкой истории: разрешена только работа с памятью.
    maintenance: bool = False


@dataclass(frozen=True)
class ToolResult:
    text: str
    ends_turn: bool = False


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    min_role: Role
    handler: Callable[[dict, TurnContext], Awaitable[str]]
    # Успешный вызов завершает ход без лишнего запроса к модели.
    ends_turn: bool = False


def _schema(optional: tuple[str, ...] = (), **properties: str) -> dict:
    return {
        "type": "object",
        "properties": {name: {"type": "string", "description": text} for name, text in properties.items()},
        "required": [name for name in properties if name not in optional],
        "additionalProperties": False,
    }


class Tools:
    def __init__(self, channel: Channel, memory: Memory):
        self.channel = channel
        self.memory = memory
        self._tools = {tool.name: tool for tool in self._build()}
        # Список не меняется в течение жизни бота: его смена сбросила бы кэш токенов.
        self.schemas = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def _build(self) -> list[Tool]:
        return [
            Tool("say", "Сказать в чат. Сообщение также озвучивается на стриме. Завершает ход.",
                 _schema(text="Текст сообщения"), Role.VIEWER, self._say, ends_turn=True),
            Tool("stay_silent", "Промолчать и завершить ход. Комментарий никто не увидит.",
                 _schema(optional=("comment",), comment="Мысль про себя, необязательно"),
                 Role.VIEWER, self._stay_silent, ends_turn=True),
            Tool("memory_read", "Прочитать заметку из памяти.",
                 _schema(name="Имя заметки"), Role.VIEWER, self._memory_read),
            Tool("memory_write", "Создать или перезаписать заметку.",
                 _schema(
                     name="Короткое имя: буквы, цифры, _ и -",
                     description="Одна строка для индекса, до 120 символов",
                     content="Текст заметки, до 2000 символов",
                 ), Role.VIEWER, self._memory_write),
            Tool("memory_delete", "Удалить заметку.",
                 _schema(name="Имя заметки"), Role.MODERATOR, self._memory_delete),
            Tool("get_channel", "Узнать текущее название стрима и игру.",
                 _schema(), Role.VIEWER, self._get_channel),
            Tool("timeout_user", "Выдать зрителю таймаут.",
                 _schema(login="Ник зрителя", seconds="Длительность в секундах, числом",
                         reason="Причина"), Role.MODERATOR, self._timeout_user),
            Tool("ban_user", "Забанить зрителя.",
                 _schema(login="Ник зрителя", reason="Причина"), Role.MODERATOR, self._ban_user),
            Tool("set_title", "Изменить название стрима.",
                 _schema(title="Новое название"), Role.BROADCASTER, self._set_title),
        ]

    async def call(self, name: str, raw_args: str, ctx: TurnContext) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(f"ошибка: инструмента {name} нет")
        if ctx.maintenance and not (name.startswith("memory_") or name == "stay_silent"):
            return ToolResult("отказано: в служебном ходе можно работать только с памятью")
        if ctx.role < tool.min_role:
            return ToolResult(
                f"отказано в доступе: {name} требует уровень {tool.min_role.name.lower()}, "
                f"а у {ctx.actor} уровень {ctx.role.name.lower()}"
            )
        try:
            text = await tool.handler(json.loads(raw_args or "{}"), ctx)
        except (ToolError, MemoryError) as error:
            return ToolResult(f"ошибка: {error}")
        except Exception as error:
            log.exception("Инструмент %s упал", name)
            return ToolResult(f"ошибка: {type(error).__name__}: {error}")
        log.info("%s(%s) от %s: %s", name, raw_args, ctx.actor, text)
        return ToolResult(text, tool.ends_turn)

    async def _say(self, args: dict, ctx: TurnContext) -> str:
        text = args["text"].strip()
        if not text:
            raise ToolError("пустое сообщение")
        if len(text) > MAX_CHAT_MESSAGE:
            raise ToolError(f"длиннее {MAX_CHAT_MESSAGE} символов, сократи")
        await self.channel.send_chat(text)
        return "отправлено"

    async def _stay_silent(self, args: dict, ctx: TurnContext) -> str:
        return "молчу"

    async def _memory_read(self, args: dict, ctx: TurnContext) -> str:
        return self.memory.read(args["name"])

    async def _memory_write(self, args: dict, ctx: TurnContext) -> str:
        author = self.memory.author(args["name"])
        if ctx.role < Role.MODERATOR and author is not None and author != ctx.actor:
            raise ToolError(f"отказано в доступе: заметку написал {author}, зритель может перезаписать только свою")
        self.memory.write(args["name"], args["description"], args["content"], ctx.actor)
        return "записано"

    async def _memory_delete(self, args: dict, ctx: TurnContext) -> str:
        self.memory.delete(args["name"])
        return "удалено"

    async def _get_channel(self, args: dict, ctx: TurnContext) -> str:
        return await self.channel.get_channel()

    async def _timeout_user(self, args: dict, ctx: TurnContext) -> str:
        seconds = int(args["seconds"])
        if not 1 <= seconds <= MAX_TIMEOUT_SECONDS:
            raise ToolError(f"таймаут от 1 до {MAX_TIMEOUT_SECONDS} секунд")
        login = _login(args["login"])
        await self.channel.timeout_user(login, seconds, args["reason"])
        return f"{login} получил таймаут на {seconds} с. {DONE_HINT}"

    async def _ban_user(self, args: dict, ctx: TurnContext) -> str:
        login = _login(args["login"])
        await self.channel.ban_user(login, args["reason"])
        return f"{login} забанен. {DONE_HINT}"

    async def _set_title(self, args: dict, ctx: TurnContext) -> str:
        await self.channel.set_title(args["title"])
        return f"название изменено на «{args['title']}». {DONE_HINT}"


def _login(value: str) -> str:
    return value.strip().lstrip("@").lower()
