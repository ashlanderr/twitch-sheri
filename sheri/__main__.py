import argparse
import asyncio
import logging
import sys

from .agent import Agent
from .config import Config, load_config
from .console import ConsoleChannel, read_console
from .events import EventQueue
from .llm import RouterAI
from .memory import Memory
from .tools import Tools
from .twitch import TwitchChannel
from .twitch_auth import TokenStore, device_login


def build_agent(config: Config, channel) -> tuple[Agent, EventQueue]:
    if not config.api_key:
        sys.exit("Укажи ROUTERAI_API_KEY в файле .env")
    memory = Memory(config.data_dir / "memory")
    agent = Agent(RouterAI(config), Tools(channel, memory), memory, config)
    return agent, EventQueue(config.debounce_seconds, config.batch_max_wait_seconds, config.max_batch)


async def run_console(config: Config) -> None:
    agent, queue = build_agent(config, ConsoleChannel())
    worker = asyncio.create_task(agent.run(queue))
    try:
        await read_console(queue)
    finally:
        worker.cancel()


async def run_twitch(config: Config) -> None:
    if not config.twitch_client_id:
        sys.exit("Укажи TWITCH_CLIENT_ID в файле .env")
    tokens_dir = config.data_dir / "tokens"
    missing = [name for name in ("bot", "broadcaster") if not (tokens_dir / f"{name}.json").exists()]
    if missing:
        sys.exit("Сначала выполни вход: " + ", ".join(f"python -m sheri login {name}" for name in missing))
    channel = TwitchChannel(
        config.twitch_client_id,
        TokenStore(tokens_dir / "bot.json", config.twitch_client_id),
        TokenStore(tokens_dir / "broadcaster.json", config.twitch_client_id),
    )
    agent, queue = build_agent(config, channel)
    await asyncio.gather(agent.run(queue), channel.read_chat(queue))


def main() -> None:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="sheri")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("console", help="проверка без Twitch")
    commands.add_parser("twitch", help="работа в чате Twitch")
    login = commands.add_parser("login", help="вход в аккаунт Twitch")
    login.add_argument("account", choices=["bot", "broadcaster"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config = load_config()
    if args.command == "console":
        asyncio.run(run_console(config))
    elif args.command == "twitch":
        asyncio.run(run_twitch(config))
    else:
        if not config.twitch_client_id:
            sys.exit("Укажи TWITCH_CLIENT_ID в файле .env")
        asyncio.run(device_login(args.account, config.twitch_client_id, config.data_dir / "tokens"))


if __name__ == "__main__":
    main()
