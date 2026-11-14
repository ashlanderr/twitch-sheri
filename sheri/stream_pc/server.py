import asyncio
import hmac
import json
import logging
import uuid
from typing import Callable, Protocol

import numpy as np
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed, ConnectionClosedError

from ..tools import ToolError

log = logging.getLogger(__name__)

# Снимок экрана 1080p в JPEG и base64 занимает около 300 КБ, запас на яркие сцены.
MAX_MESSAGE_BYTES = 8 * 1024 * 1024
SCREENSHOT_TIMEOUT_SECONDS = 10.0


class Transcriber(Protocol):
    def transcribe(self, pcm: np.ndarray) -> str: ...


class StreamScreen:
    """Снимки экрана через клиент на стрим-компе. Сервер шлёт текстом
    {"type": "screenshot", "id"}, клиент отвечает {"type": "screenshot", "id",
    "image"} с data URL картинки или {"type": "screenshot", "id", "error"}."""

    def __init__(self) -> None:
        self._connection: ServerConnection | None = None
        self._pending: dict[str, asyncio.Future[dict]] = {}

    def attach(self, connection: ServerConnection) -> None:
        self._connection = connection

    def detach(self, connection: ServerConnection) -> None:
        if self._connection is connection:
            self._connection = None

    async def capture(self) -> str:
        connection = self._connection
        if connection is None:
            raise ToolError("клиент на стрим-компе не подключён, экран не видно")
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await connection.send(json.dumps({"type": "screenshot", "id": request_id}))
            async with asyncio.timeout(SCREENSHOT_TIMEOUT_SECONDS):
                reply = await future
        except TimeoutError:
            raise ToolError("клиент на стрим-компе не прислал снимок вовремя") from None
        except ConnectionClosed:
            raise ToolError("связь с клиентом на стрим-компе оборвалась") from None
        finally:
            self._pending.pop(request_id, None)
        if reply.get("error"):
            raise ToolError(f"снимок не получился: {reply['error']}")
        return reply["image"]

    def receive(self, message: str) -> None:
        try:
            reply = json.loads(message)
        except json.JSONDecodeError:
            log.warning("Непонятное сообщение от клиента: %.100s", message)
            return
        future = self._pending.get(reply.get("id")) if isinstance(reply, dict) else None
        if future is not None and not future.done():
            future.set_result(reply)


async def serve_stream_pc(
    host: str,
    port: int,
    token: str,
    transcriber: Transcriber,
    on_text: Callable[[str], None],
    screen: StreamScreen | None = None,
) -> None:
    """Связь с клиентом на стрим-компе. Бинарное сообщение — одна фраза с
    микрофона, 16 кГц, моно, int16: распознанный текст уходит в on_text.
    Текстовые сообщения — ответы на запросы снимков экрана."""
    screen = screen or StreamScreen()
    if not token:
        raise RuntimeError(
            "Укажи STREAM_PC_TOKEN в .env: без него голос и снимки может прислать кто угодно в сети"
        )
    # Модель на видеокарте обрабатывает фразы по одной.
    lock = asyncio.Lock()

    async def handle(connection: ServerConnection) -> None:
        given = connection.request.headers.get("Authorization", "")
        if not hmac.compare_digest(given, f"Bearer {token}"):
            log.warning("Стрим-комп %s с неверным токеном", connection.remote_address)
            await connection.close(4001, "bad token")
            return
        log.info("Стрим-комп подключился: %s", connection.remote_address)
        # Распознавание идёт отдельно от чтения, чтобы ответ со снимком
        # не ждал, пока модель дослушает фразу. Порядок фраз сохраняется.
        phrases: asyncio.Queue[np.ndarray] = asyncio.Queue()

        async def transcribe() -> None:
            while True:
                pcm = await phrases.get()
                try:
                    async with lock:
                        text = await asyncio.to_thread(transcriber.transcribe, pcm)
                    log.info("Голос (%.1f с): %s", len(pcm) / 16_000, text)
                    on_text(text)
                except Exception:
                    log.exception("Не удалось распознать фразу")
                finally:
                    phrases.task_done()

        worker = asyncio.create_task(transcribe())
        screen.attach(connection)
        try:
            async for message in connection:
                if isinstance(message, str):
                    screen.receive(message)
                else:
                    phrases.put_nowait(np.frombuffer(message, dtype=np.int16))
        except ConnectionClosedError:
            # Клиент закрылся без прощания: выключили комп или оборвалась сеть.
            pass
        finally:
            screen.detach(connection)
            # Уже принятые фразы дослушиваем, потом останавливаемся.
            await phrases.join()
            worker.cancel()
        log.info("Стрим-комп отключился: %s", connection.remote_address)

    async with serve(handle, host, port, max_size=MAX_MESSAGE_BYTES) as server:
        log.info("Жду стрим-комп на %s:%s", host, port)
        await server.serve_forever()
