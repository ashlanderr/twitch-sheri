import argparse
import asyncio
import getpass
import logging
import sys

from .agent import Agent
from .config import Config, load_config
from .console import ConsoleChannel, read_console
from .events import EventQueue
from .hearing import Hearing
from .llm import RouterAI
from .memory import Memory
from .tools import Tools
from .twitch import TwitchChannel
from .twitch_auth import TokenStore, device_login, import_token


def build_agent(config: Config, channel, screen=None) -> tuple[Agent, EventQueue]:
    if not config.api_key:
        sys.exit("Укажи ROUTERAI_API_KEY в файле .env")
    memory = Memory(config.data_dir / "memory")
    agent = Agent(RouterAI(config), Tools(channel, memory, screen), memory, config)
    return agent, EventQueue(config.debounce_seconds, config.batch_max_wait_seconds, config.max_batch)


async def serve_stream_pc(config: Config, queue: EventQueue, screen) -> None:
    # Тяжёлые импорты только тут: стрим-компу и чату torch не нужен.
    from .stream_pc.server import serve_stream_pc as serve
    from .voice.stt import GigaAMTranscriber

    hearing = Hearing(
        queue, config.voice_speaker, config.voice_context_seconds, config.voice_followup_seconds
    )
    transcriber = await asyncio.to_thread(GigaAMTranscriber, config.stt_model)
    await serve(
        config.stream_pc_host, config.stream_pc_port, config.stream_pc_token,
        transcriber, hearing.hear, screen,
    )


def make_screen(stream_pc: bool):
    """Экран Шери видит только через стрим-комп."""
    if not stream_pc:
        return None
    from .stream_pc.server import StreamScreen

    return StreamScreen()


async def run_console(config: Config, stream_pc: bool) -> None:
    screen = make_screen(stream_pc)
    agent, queue = build_agent(config, ConsoleChannel(), screen)
    workers = [asyncio.create_task(agent.run(queue))]
    if stream_pc:
        workers.append(asyncio.create_task(serve_stream_pc(config, queue, screen)))
    try:
        await read_console(queue)
    finally:
        for worker in workers:
            worker.cancel()


async def run_twitch(config: Config, stream_pc: bool) -> None:
    if not config.twitch_client_id:
        sys.exit("Укажи TWITCH_CLIENT_ID в файле .env")
    tokens_dir = config.data_dir / "tokens"
    missing = [name for name in ("bot", "broadcaster") if not (tokens_dir / f"{name}.json").exists()]
    if missing:
        sys.exit("Сначала выполни вход: " + ", ".join(f"python -m sheri login {name} (или import-token {name})" for name in missing))
    channel = TwitchChannel(
        config.twitch_client_id,
        TokenStore(tokens_dir / "bot.json", config.twitch_client_id),
        TokenStore(tokens_dir / "broadcaster.json", config.twitch_client_id),
    )
    screen = make_screen(stream_pc)
    agent, queue = build_agent(config, channel, screen)
    tasks = [agent.run(queue), channel.read_chat(queue)]
    if stream_pc:
        tasks.append(serve_stream_pc(config, queue, screen))
    await asyncio.gather(*tasks)


def run_stream_pc(config: Config, args: argparse.Namespace) -> None:
    from .stream_pc.client import list_devices, run_client
    from .stream_pc.obs import OBS

    if args.list_devices:
        list_devices()
        return
    if not config.stream_pc_token:
        sys.exit("Укажи в .env тот же STREAM_PC_TOKEN, что и у бота")
    device = int(args.device) if args.device and args.device.isdigit() else args.device
    obs = None if args.no_screen else OBS(args.obs, config.obs_password, args.screen_width)
    asyncio.run(run_client(args.bot, config.stream_pc_token, device, obs))


def run_import_token(config: Config, account: str) -> None:
    # getpass не показывает ввод, чтобы токены не остались на экране.
    access_token = getpass.getpass("Access Token: ").strip()
    refresh_token = getpass.getpass("Refresh Token: ").strip()
    try:
        client_id = asyncio.run(import_token(account, access_token, refresh_token, config.data_dir / "tokens"))
    except RuntimeError as error:
        sys.exit(str(error))
    if client_id != config.twitch_client_id:
        print(f"Укажи в .env: TWITCH_CLIENT_ID={client_id}")


def main() -> None:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="sheri")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("console", "проверка без Twitch"), ("twitch", "работа в чате Twitch")):
        command = commands.add_parser(name, help=text)
        command.add_argument(
            "--stream-pc", action="store_true",
            help="ждать подключения стрим-компа: голос Любы и снимки экрана",
        )
    login = commands.add_parser("login", help="вход в аккаунт Twitch")
    login.add_argument("account", choices=["bot", "broadcaster"])
    imported = commands.add_parser("import-token", help="токены с twitchtokengenerator.com")
    imported.add_argument("account", choices=["bot", "broadcaster"])
    client = commands.add_parser(
        "stream-pc", help="запускать на стрим-компе: отдаёт боту микрофон и снимки экрана"
    )
    client.add_argument("--bot", default="ws://127.0.0.1:8765", help="адрес компа, где запущен бот")
    client.add_argument("--device", help="номер или имя микрофона")
    client.add_argument("--list-devices", action="store_true", help="показать микрофоны")
    client.add_argument("--obs", default="ws://127.0.0.1:4455", help="адрес WebSocket-сервера OBS")
    client.add_argument("--screen-width", type=int, default=1920, help="ширина снимка экрана")
    client.add_argument("--no-screen", action="store_true", help="не давать Шери смотреть на экран")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config = load_config()
    if args.command == "console":
        asyncio.run(run_console(config, args.stream_pc))
    elif args.command == "twitch":
        asyncio.run(run_twitch(config, args.stream_pc))
    elif args.command == "stream-pc":
        run_stream_pc(config, args)
    elif args.command == "import-token":
        run_import_token(config, args.account)
    else:
        if not config.twitch_client_id:
            sys.exit("Укажи TWITCH_CLIENT_ID в файле .env")
        asyncio.run(device_login(args.account, config.twitch_client_id, config.data_dir / "tokens"))


if __name__ == "__main__":
    main()
