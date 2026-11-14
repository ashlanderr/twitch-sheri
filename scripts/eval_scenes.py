"""Прогоняет короткие сцены на настоящей модели, каждую с чистой историей.

Сцены: JSON {"название": [["ник", "текст"], ...]}. Каждая реплика идёт отдельным
ходом, Шери отвечает сам. Реплики не должны опираться на конкретный ответ Шери.

    python scripts/eval_scenes.py evals/scenes.json --memory evals/memory --out out.txt
"""
import argparse
import asyncio
import json
import logging
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from replay_chat import ROLES, START, CostingRouterAI, Recorder

from sheri.agent import Agent
from sheri.config import load_config
from sheri.events import Event, Role
from sheri.memory import Memory
from sheri.tools import Tools


async def run_scene(name, lines, config, memory_dir, llm, limit):
    async with limit:
        config = replace(config, data_dir=Path(tempfile.mkdtemp()))
        if memory_dir:
            shutil.copytree(memory_dir, config.data_dir / "memory")
        memory = Memory(config.data_dir / "memory")
        log = [f"## {name}"]
        agent = Agent(llm, Tools(Recorder(log), memory), memory, config)
        for user, text in lines:
            log.append(f"{user}: {text}")
            size = len(log)
            try:
                await agent.handle([Event("twitch", ROLES.get(user, Role.VIEWER), user, text, START)])
            except Exception as error:
                log.append(f"    [ошибка: {error}]")
            if len(log) == size:
                log.append("    ... (молчит)")
        return "\n".join(log)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", type=Path)
    parser.add_argument("--prompts", type=Path, default=None)
    parser.add_argument("--memory", type=Path, default=None)
    parser.add_argument("--only", default=None, help="подстрока в названии сцены")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    scenes = json.loads(args.scenes.read_text(encoding="utf-8"))
    if args.only:
        scenes = {k: v for k, v in scenes.items() if args.only in k}
    config = load_config()
    if args.prompts:
        config = replace(config, prompts_dir=args.prompts)
    llm = CostingRouterAI(config)
    limit = asyncio.Semaphore(6)
    parts = await asyncio.gather(
        *(run_scene(name, lines, config, args.memory, llm, limit) for name, lines in scenes.items())
    )
    text = "\n\n".join(parts) + f"\n\nстоимость: {llm.cost:.2f}"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    asyncio.run(main())
