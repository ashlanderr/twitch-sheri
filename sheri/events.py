import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum


class Role(IntEnum):
    VIEWER = 0
    MODERATOR = 1
    BROADCASTER = 2


@dataclass
class Event:
    source: str
    role: Role
    user: str
    text: str
    time: datetime = field(default_factory=datetime.now)

    def to_line(self) -> str:
        # JSON экранирует текст, поэтому зритель не подделает строку чужого события.
        return json.dumps(
            {
                "time": self.time.strftime("%d.%m %H:%M"),
                "source": self.source,
                "role": self.role.name.lower(),
                "user": self.user,
                "text": self.text,
            },
            ensure_ascii=False,
        )


def _same_turn(first: Event, event: Event) -> bool:
    """Зрители склеиваются друг с другом, а автор с правами только сам с собой,
    чтобы у каждого хода был однозначный уровень доступа."""
    if first.role == Role.VIEWER:
        return event.role == Role.VIEWER
    return event.user == first.user and event.role == first.role


class EventQueue:
    """Отдаёт события пачками: пачка закрывается, когда новых сообщений нет
    debounce секунд подряд или когда она собирается дольше max_wait."""

    def __init__(self, debounce_seconds: float, max_wait_seconds: float, max_batch: int):
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._held: Event | None = None
        self._debounce = debounce_seconds
        self._max_wait = max_wait_seconds
        self._max_batch = max_batch

    def put(self, event: Event) -> None:
        self._queue.put_nowait(event)

    async def next_batch(self) -> list[Event]:
        first, self._held = self._held, None
        if first is None:
            first = await self._queue.get()

        batch = [first]
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._max_wait
        while len(batch) < self._max_batch:
            timeout = min(self._debounce, deadline - loop.time())
            if timeout <= 0:
                break
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout)
            except TimeoutError:
                break
            if not _same_turn(first, event):
                self._held = event
                break
            batch.append(event)
        return batch
