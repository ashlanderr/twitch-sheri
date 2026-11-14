import asyncio
import contextlib
import itertools
import json
import time

import numpy as np
import pytest
import websockets

from sheri.tools import ToolError
from sheri.stream_pc.server import StreamScreen, serve_stream_pc

PORTS = itertools.count(18765)


class FakeTranscriber:
    def __init__(self):
        self.lengths = []

    def transcribe(self, pcm):
        self.lengths.append(len(pcm))
        return f"фраза из {len(pcm)} отсчётов"


@contextlib.asynccontextmanager
async def running_server(token="secret"):
    port = next(PORTS)
    heard, transcriber = [], FakeTranscriber()
    server = asyncio.create_task(serve_stream_pc("127.0.0.1", port, token, transcriber, heard.append))
    try:
        async with asyncio.timeout(5):
            while True:
                try:
                    reader, writer = await asyncio.open_connection("127.0.0.1", port)
                    writer.close()
                    break
                except OSError:
                    await asyncio.sleep(0.05)
        yield f"ws://127.0.0.1:{port}", heard, transcriber
    finally:
        server.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server


async def test_segment_is_transcribed_and_passed_on():
    async with asyncio.timeout(10), running_server() as (url, heard, _):
        headers = {"Authorization": "Bearer secret"}
        async with websockets.connect(url, additional_headers=headers) as socket:
            await socket.send(np.zeros(16_000, dtype=np.int16).tobytes())
            await socket.send("текст игнорируется")
            await socket.send(np.zeros(8_000, dtype=np.int16).tobytes())
            while len(heard) < 2:
                await asyncio.sleep(0.02)
        assert heard == ["фраза из 16000 отсчётов", "фраза из 8000 отсчётов"]


async def test_wrong_token_is_rejected():
    async with asyncio.timeout(10), running_server() as (url, heard, transcriber):
        headers = {"Authorization": "Bearer wrong"}
        async with websockets.connect(url, additional_headers=headers) as socket:
            with pytest.raises(websockets.ConnectionClosed) as closed:
                await socket.recv()
        assert closed.value.rcvd.code == 4001
        assert transcriber.lengths == []


async def test_abrupt_disconnect_keeps_server_running(caplog):
    async with asyncio.timeout(10), running_server() as (url, heard, _):
        headers = {"Authorization": "Bearer secret"}
        socket = await websockets.connect(url, additional_headers=headers)
        await socket.send(np.zeros(1_600, dtype=np.int16).tobytes())
        while not heard:
            await asyncio.sleep(0.02)
        socket.transport.abort()
        await asyncio.sleep(0.2)
        async with websockets.connect(url, additional_headers=headers) as again:
            await again.send(np.zeros(3_200, dtype=np.int16).tobytes())
            while len(heard) < 2:
                await asyncio.sleep(0.02)
    assert not [r for r in caplog.records if r.levelname == "ERROR"]


async def test_server_refuses_to_start_without_token():
    with pytest.raises(RuntimeError, match="STREAM_PC_TOKEN"):
        await serve_stream_pc("127.0.0.1", next(PORTS), "", FakeTranscriber(), print)


async def test_screenshot_round_trip_through_client():
    port = next(PORTS)
    screen = StreamScreen()
    server = asyncio.create_task(
        serve_stream_pc("127.0.0.1", port, "secret", FakeTranscriber(), print, screen)
    )
    try:
        async with asyncio.timeout(10):
            with pytest.raises(ToolError, match="не подключён"):
                await screen.capture()
            while True:
                try:
                    socket = await websockets.connect(
                        f"ws://127.0.0.1:{port}", additional_headers={"Authorization": "Bearer secret"}
                    )
                    break
                except OSError:
                    await asyncio.sleep(0.05)

            async def client():
                for answer in ({"image": "data:image/jpeg;base64,QQ=="}, {"error": "OBS выключен"}):
                    request = json.loads(await socket.recv())
                    assert request["type"] == "screenshot"
                    await socket.send(json.dumps({"type": "screenshot", "id": request["id"], **answer}))

            answering = asyncio.create_task(client())
            assert await screen.capture() == "data:image/jpeg;base64,QQ=="
            with pytest.raises(ToolError, match="OBS выключен"):
                await screen.capture()
            await answering
            await socket.close()
            await asyncio.sleep(0.1)
            with pytest.raises(ToolError, match="не подключён"):
                await screen.capture()
    finally:
        server.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server


async def test_screenshot_does_not_wait_for_slow_transcription():
    class SlowTranscriber(FakeTranscriber):
        def transcribe(self, pcm):
            time.sleep(1.5)
            return super().transcribe(pcm)

    port, screen, heard = next(PORTS), StreamScreen(), []
    server = asyncio.create_task(
        serve_stream_pc("127.0.0.1", port, "secret", SlowTranscriber(), heard.append, screen)
    )
    try:
        async with asyncio.timeout(10):
            while True:
                try:
                    socket = await websockets.connect(
                        f"ws://127.0.0.1:{port}", additional_headers={"Authorization": "Bearer secret"}
                    )
                    break
                except OSError:
                    await asyncio.sleep(0.05)
            await socket.send(np.zeros(1_600, dtype=np.int16).tobytes())
            await asyncio.sleep(0.1)

            async def client():
                request = json.loads(await socket.recv())
                await socket.send(json.dumps({"type": "screenshot", "id": request["id"], "image": "x"}))

            answering = asyncio.create_task(client())
            started = time.monotonic()
            assert await screen.capture() == "x"
            assert time.monotonic() - started < 1.0
            await answering
            # Отключение посреди распознавания не теряет фразу.
            await socket.close()
            while not heard:
                await asyncio.sleep(0.05)
    finally:
        server.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server
