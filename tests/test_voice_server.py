import asyncio
import contextlib
import itertools

import numpy as np
import pytest
import websockets

from sheri.voice.server import serve_voice

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
    server = asyncio.create_task(serve_voice("127.0.0.1", port, token, transcriber, heard.append))
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
    with pytest.raises(RuntimeError, match="VOICE_TOKEN"):
        await serve_voice("127.0.0.1", next(PORTS), "", FakeTranscriber(), print)
