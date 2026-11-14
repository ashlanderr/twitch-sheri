import asyncio
import contextlib
import itertools
import json

import pytest
from websockets.asyncio.server import serve

from sheri.voice.obs import OBS, OBSError, _auth

PORTS = itertools.count(19455)


@contextlib.asynccontextmanager
async def fake_obs(password="pw"):
    """Минимальный obs-websocket 5: вход по паролю и два запроса снимка."""
    requests = []

    async def handle(socket):
        await socket.send(json.dumps({"op": 0, "d": {"rpcVersion": 1, "authentication": {
            "salt": "salt", "challenge": "challenge"}}}))
        identify = json.loads(await socket.recv())["d"]
        if identify.get("authentication") != _auth(password, "salt", "challenge"):
            await socket.close(4009, "Authentication failed.")
            return
        await socket.send(json.dumps({"op": 2, "d": {"negotiatedRpcVersion": 1}}))
        async for message in socket:
            request = json.loads(message)["d"]
            requests.append(request)
            data = {"currentProgramSceneName": "Игра"}
            if request["requestType"] == "GetSourceScreenshot":
                data = {"imageData": "data:image/jpg;base64,QUJD"}
            # Событие посреди ответа не должно сбивать клиента.
            await socket.send(json.dumps({"op": 5, "d": {"eventType": "Шум"}}))
            await socket.send(json.dumps({"op": 7, "d": {
                "requestType": request["requestType"], "requestId": request["requestId"],
                "requestStatus": {"result": True, "code": 100}, "responseData": data}}))

    port = next(PORTS)
    async with serve(handle, "127.0.0.1", port, subprotocols=["obswebsocket.json"]):
        yield f"ws://127.0.0.1:{port}", requests


async def test_screenshot_of_program_scene():
    async with asyncio.timeout(10), fake_obs() as (url, requests):
        image = await OBS(url, "pw", 1920).screenshot()
    assert image == "data:image/jpeg;base64,QUJD"
    assert requests[1]["requestData"]["sourceName"] == "Игра"
    assert requests[1]["requestData"]["imageWidth"] == 1920


async def test_wrong_password_is_reported():
    async with asyncio.timeout(10), fake_obs() as (url, _):
        with pytest.raises(OBSError, match="не пустил"):
            await OBS(url, "nope").screenshot()


async def test_obs_not_running_is_reported():
    with pytest.raises(OBSError, match="нет связи с OBS"):
        await OBS(f"ws://127.0.0.1:{next(PORTS)}", "pw").screenshot()
