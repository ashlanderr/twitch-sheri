import asyncio
import hmac
import logging
from typing import Callable, Protocol

import numpy as np
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosedError

log = logging.getLogger(__name__)

MAX_MESSAGE_BYTES = 2 * 1024 * 1024


class Transcriber(Protocol):
    def transcribe(self, pcm: np.ndarray) -> str: ...


async def serve_voice(
    host: str,
    port: int,
    token: str,
    transcriber: Transcriber,
    on_text: Callable[[str], None],
) -> None:
    """Принимает фразы от клиента: одно бинарное сообщение = одна фраза,
    16 кГц, моно, int16. Распознанный текст отдаёт в on_text."""
    if not token:
        raise RuntimeError("Укажи VOICE_TOKEN в .env: без него голос может прислать кто угодно в сети")
    # Модель на видеокарте обрабатывает фразы по одной.
    lock = asyncio.Lock()

    async def handle(connection: ServerConnection) -> None:
        given = connection.request.headers.get("Authorization", "")
        if not hmac.compare_digest(given, f"Bearer {token}"):
            log.warning("Голосовой клиент %s с неверным токеном", connection.remote_address)
            await connection.close(4001, "bad token")
            return
        log.info("Голосовой клиент подключился: %s", connection.remote_address)
        try:
            async for message in connection:
                if isinstance(message, str):
                    continue
                pcm = np.frombuffer(message, dtype=np.int16)
                async with lock:
                    text = await asyncio.to_thread(transcriber.transcribe, pcm)
                log.info("Голос (%.1f с): %s", len(pcm) / 16_000, text)
                on_text(text)
        except ConnectionClosedError:
            # Клиент закрылся без прощания: выключили комп или оборвалась сеть.
            pass
        log.info("Голосовой клиент отключился: %s", connection.remote_address)

    async with serve(handle, host, port, max_size=MAX_MESSAGE_BYTES) as server:
        log.info("Сервер голоса слушает %s:%s", host, port)
        await server.serve_forever()
