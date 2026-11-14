"""Клиент для стрим-компа: снимает микрофон, вырезает фразы и отправляет их боту,
а по его запросу присылает снимок стрима из OBS.
Тяжёлого здесь ничего нет: VAD маленький и работает на процессоре."""

import asyncio
import json
import logging

import numpy as np
import sounddevice as sd
import websockets

from ..voice.segmenter import FRAME, SAMPLE_RATE, Segmenter
from ..voice.vad import SileroVAD
from .obs import OBS, OBSError

log = logging.getLogger(__name__)

RECONNECT_DELAY_SECONDS = 3


async def answer_screenshot(socket, request: dict, obs: OBS | None) -> None:
    reply = {"type": "screenshot", "id": request.get("id")}
    if obs is None:
        reply["error"] = "снимки экрана выключены на стрим-компе"
    else:
        try:
            reply["image"] = await obs.screenshot()
            log.info("Отправлен снимок экрана, %d КБ", len(reply["image"]) * 3 // 4 // 1024)
        except OBSError as error:
            log.warning("Снимок экрана: %s", error)
            reply["error"] = str(error)
    await socket.send(json.dumps(reply))


async def run_client(url: str, token: str, device: int | str | None, obs: OBS | None = None) -> None:
    loop = asyncio.get_running_loop()
    frames: asyncio.Queue[np.ndarray] = asyncio.Queue()
    segments: asyncio.Queue[np.ndarray] = asyncio.Queue()
    connected = asyncio.Event()

    def on_audio(indata, frame_count, time_info, status) -> None:
        if status:
            log.warning("Микрофон: %s", status)
        loop.call_soon_threadsafe(frames.put_nowait, indata[:, 0].copy())

    async def cut_phrases() -> None:
        vad, segmenter = SileroVAD(), Segmenter()
        while True:
            frame = await frames.get()
            segment = segmenter.push(frame, vad(frame))
            if segment is None:
                continue
            if connected.is_set():
                segments.put_nowait(segment)
            else:
                log.warning("Бот недоступен, фраза %.1f с потеряна", len(segment) / SAMPLE_RATE)

    async def send_phrases(socket) -> None:
        while True:
            segment = await segments.get()
            await socket.send(segment.tobytes())
            log.info("Отправлена фраза %.1f с", len(segment) / SAMPLE_RATE)

    async def serve_requests(socket) -> None:
        async for message in socket:
            request = json.loads(message) if isinstance(message, str) else {}
            if request.get("type") == "screenshot":
                await answer_screenshot(socket, request, obs)

    async def stay_connected() -> None:
        headers = {"Authorization": f"Bearer {token}"}
        while True:
            try:
                async with websockets.connect(url, additional_headers=headers) as socket:
                    log.info("Подключён к боту %s", url)
                    connected.set()
                    sender = asyncio.create_task(send_phrases(socket))
                    try:
                        # Цикл чтения заканчивается, когда сервер закрыл соединение.
                        await serve_requests(socket)
                    finally:
                        sender.cancel()
            except (websockets.WebSocketException, OSError) as error:
                log.warning("Нет связи с ботом: %s", error)
            connected.clear()
            await asyncio.sleep(RECONNECT_DELAY_SECONDS)

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME,
        device=device, callback=on_audio,
    ):
        log.info("Слушаю микрофон: %s", sd.query_devices(device, "input")["name"])
        await asyncio.gather(cut_phrases(), stay_connected())


def list_devices() -> None:
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] > 0:
            api = sd.query_hostapis(info["hostapi"])["name"]
            print(f"{index:3}  {info['name']}  ({api})")
