"""Клиент для стрим-компа: снимает микрофон, вырезает фразы и отправляет их на сервер.
Тяжёлого здесь ничего нет: VAD маленький и работает на процессоре."""

import asyncio
import logging

import numpy as np
import sounddevice as sd
import websockets

from .segmenter import FRAME, SAMPLE_RATE, Segmenter
from .vad import SileroVAD

log = logging.getLogger(__name__)

RECONNECT_DELAY_SECONDS = 3


async def run_client(url: str, token: str, device: int | str | None) -> None:
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
                log.warning("Сервер недоступен, фраза %.1f с потеряна", len(segment) / SAMPLE_RATE)

    async def send_phrases() -> None:
        headers = {"Authorization": f"Bearer {token}"}
        while True:
            try:
                async with websockets.connect(url, additional_headers=headers) as socket:
                    log.info("Подключён к серверу голоса %s", url)
                    connected.set()
                    while True:
                        segment = await segments.get()
                        await socket.send(segment.tobytes())
                        log.info("Отправлена фраза %.1f с", len(segment) / SAMPLE_RATE)
            except (websockets.WebSocketException, OSError) as error:
                log.warning("Нет связи с сервером голоса: %s", error)
            connected.clear()
            await asyncio.sleep(RECONNECT_DELAY_SECONDS)

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME,
        device=device, callback=on_audio,
    ):
        log.info("Слушаю микрофон: %s", sd.query_devices(device, "input")["name"])
        await asyncio.gather(cut_phrases(), send_phrases())


def list_devices() -> None:
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] > 0:
            api = sd.query_hostapis(info["hostapi"])["name"]
            print(f"{index:3}  {info['name']}  ({api})")
