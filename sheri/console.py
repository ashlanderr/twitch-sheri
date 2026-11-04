import asyncio

from .events import Event, EventQueue, Role

HELP = """Обычная строка уходит Шери от текущего пользователя.
/as <viewer|moderator|broadcaster> <ник>  сменить пользователя
/quit  выйти"""


class ConsoleChannel:
    """Канал для проверки без Twitch: действия только печатаются."""

    title = "Тестовый стрим"

    async def send_chat(self, text: str) -> None:
        print(f"Шери: {text}")

    async def timeout_user(self, login: str, seconds: int, reason: str) -> None:
        print(f"[таймаут {login} на {seconds} с: {reason}]")

    async def ban_user(self, login: str, reason: str) -> None:
        print(f"[бан {login}: {reason}]")

    async def set_title(self, title: str) -> None:
        self.title = title
        print(f"[название: {title}]")

    async def get_channel(self) -> str:
        return f"название: {self.title}; игра: не указана"


async def read_console(queue: EventQueue) -> None:
    print(HELP)
    role, user = Role.VIEWER, "zritel"
    while True:
        line = (await asyncio.to_thread(input, f"{user}> ")).strip()
        if line == "/quit":
            return
        if line.startswith("/as"):
            parts = line.split()
            if len(parts) == 3 and parts[1].upper() in Role.__members__:
                role, user = Role[parts[1].upper()], parts[2]
            else:
                print(HELP)
        elif line:
            queue.put(Event("console", role, user, line))
