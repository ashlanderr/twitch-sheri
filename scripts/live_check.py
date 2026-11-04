"""Прогон сценария на настоящей модели во временной папке: история и память не трогаются."""
import asyncio
import logging
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from sheri.agent import Agent
from sheri.config import load_config
from sheri.console import ConsoleChannel
from sheri.events import Event, Role
from sheri.llm import RouterAI
from sheri.memory import Memory
from sheri.tools import Tools

SCRIPT = [
    (Role.MODERATOR, "ashllanderr", "Шери, привет"),
    (Role.MODERATOR, "ashllanderr", "@Mo0nlaght во что сегодня играешь?"),
    (Role.BROADCASTER, "Mo0nlaght", "@ashllanderr пока не решила"),
    (Role.VIEWER, "vasya", "кот, ты толстый"),
]


async def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config = replace(load_config(), data_dir=Path(tempfile.mkdtemp()))
    memory = Memory(config.data_dir / "memory")
    agent = Agent(RouterAI(config), Tools(ConsoleChannel(), memory), memory, config)
    for role, user, text in SCRIPT:
        print(f"\n{user}> {text}")
        await agent.handle([Event("console", role, user, text)])


asyncio.run(main())
