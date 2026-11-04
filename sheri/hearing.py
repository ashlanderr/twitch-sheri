import re
import time
from collections import deque
from typing import Callable

from .events import Event, EventQueue, Role

# «Шери» в разных написаниях распознавания: Шери, Шерри, Шэри, Шерик, Шеричка, Sherry.
WAKE_RE = re.compile(
    r"(?<![а-яёa-z])(?:ш[еэ]р+и(?:к\w*|чк\w*)?|sherr?y|cherry|sheri)(?![а-яёa-z])",
    re.IGNORECASE,
)


class Hearing:
    """Слух Шери. Речь, обращённая не к нему, копится в буфере и хода не запускает.
    Когда звучит его имя, буфер уходит агенту вместе с обращением, чтобы модель
    по контексту поняла, к кому обращались: к ней или к настоящему коту."""

    def __init__(
        self,
        queue: EventQueue,
        speaker: str,
        context_seconds: float,
        followup_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.queue = queue
        self.speaker = speaker
        self.context_seconds = context_seconds
        self.followup_seconds = followup_seconds
        self.clock = clock
        self._buffer: deque[tuple[float, Event]] = deque()
        # Пока окно открыто, реплики без имени тоже адресованы Шери:
        # «Шери, забань Васю» — «Банить vasya_2007?» — «да».
        self._followup_until = float("-inf")

    def hear(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        now = self.clock()
        while self._buffer and self._buffer[0][0] < now - self.context_seconds:
            self._buffer.popleft()

        if not (WAKE_RE.search(text) or now < self._followup_until):
            self._buffer.append((now, Event("room", Role.BROADCASTER, self.speaker, text)))
            return

        for _, event in self._buffer:
            self.queue.put(event)
        self._buffer.clear()
        self.queue.put(Event("voice", Role.BROADCASTER, self.speaker, text))
        self._followup_until = now + self.followup_seconds
