"""Прогоняет реплики людей из записи чата через Шери на настоящей модели.

Ответы прежнего бота выкидываются, Шери отвечает заново. Работает во временной
папке: история и память бота не трогаются.

Формат чата: JSON-список {"t": секунды от начала, "user": ник, "text": текст}.

    python scripts/replay_chat.py chat.json --from 5 --to 20 --prompts prompts --out out.txt
"""
import argparse
import asyncio
import logging
import shutil
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from sheri.agent import Agent
from sheri.config import load_config
from sheri.events import Event, Role, _same_turn
from sheri.llm import RouterAI
from sheri.memory import Memory
from sheri.tools import Tools

BOT = "sheri_cat"
SKIP = {BOT, "jeetbot"}
ROLES = {"mo0nlaght": Role.BROADCASTER, "ashllanderr": Role.MODERATOR}
START = datetime(2026, 11, 13, 20, 0)


class Recorder:
    """Канал, который записывает действия Шери в общий лог."""

    title = "Dead by Daylight с котом"

    def __init__(self, log: list[str]):
        self.log = log

    async def send_chat(self, text: str) -> None:
        self.log.append(f"    >>> ШЕРИ: {text}")

    async def timeout_user(self, login: str, seconds: int, reason: str) -> None:
        self.log.append(f"    [таймаут {login} {seconds} с: {reason}]")

    async def ban_user(self, login: str, reason: str) -> None:
        self.log.append(f"    [бан {login}: {reason}]")

    async def set_title(self, title: str) -> None:
        self.log.append(f"    [название: {title}]")

    async def get_channel(self) -> str:
        return f"название: {self.title}; игра: Dead by Daylight"


class CostingRouterAI(RouterAI):
    """RouterAI, который считает стоимость запросов по полю cost из ответа."""

    def __init__(self, config):
        super().__init__(config)
        self.cost = 0.0
        create = self.client.chat.completions.create

        async def counted(*args, **kwargs):
            response = await create(*args, **kwargs)
            self.cost += float(getattr(response.usage, "cost", 0) or 0)
            return response

        self.client.chat.completions.create = counted


def batches(messages: list[dict], debounce: float, max_wait: float) -> list[list[Event]]:
    """Склеивает сообщения так же, как EventQueue в живом боте."""
    result: list[list[Event]] = []
    first_t = last_t = None
    for message in messages:
        role = ROLES.get(message["user"], Role.VIEWER)
        event = Event("twitch", role, message["user"], message["text"], START + timedelta(seconds=message["t"]))
        current = result[-1] if result else None
        if (
            current
            and _same_turn(current[0], event)
            and message["t"] - last_t <= debounce
            and message["t"] - first_t <= max_wait
        ):
            current.append(event)
        else:
            result.append([event])
            first_t = message["t"]
        last_t = message["t"]
    return result


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("chat", type=Path)
    parser.add_argument("--from", dest="start", type=float, default=0, help="минута начала")
    parser.add_argument("--to", dest="end", type=float, default=1e9, help="минута конца")
    parser.add_argument("--prompts", type=Path, default=None, help="папка с system.md и people.md")
    parser.add_argument("--memory", type=Path, default=None, help="папка с заметками памяти")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    import json

    chat = json.loads(args.chat.read_text(encoding="utf-8"))
    messages = [
        m for m in chat
        if m["user"] not in SKIP and args.start * 60 <= m["t"] < args.end * 60 and m["text"].strip()
    ]
    config = replace(load_config(), data_dir=Path(tempfile.mkdtemp()))
    if args.prompts:
        config = replace(config, prompts_dir=args.prompts)
    if args.memory:
        shutil.copytree(args.memory, config.data_dir / "memory")
    memory = Memory(config.data_dir / "memory")
    log: list[str] = []
    llm = CostingRouterAI(config)
    agent = Agent(llm, Tools(Recorder(log), memory), memory, config)

    for batch in batches(messages, config.debounce_seconds, config.batch_max_wait_seconds):
        for event in batch:
            minute = int((event.time - START).total_seconds() // 60)
            log.append(f"{minute:3}m {event.user}: {event.text}")
        size = len(log)
        try:
            await agent.handle(batch)
        except Exception as error:
            log.append(f"    [ошибка: {error}]")
        if len(log) == size:
            log.append("    ... (молчит)")

    log.append(f"\nстоимость: {llm.cost:.2f}")
    text = "\n".join(log)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    asyncio.run(main())
