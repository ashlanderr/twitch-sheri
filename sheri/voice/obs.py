"""Снимок сцены, которая сейчас идёт в эфир, через obs-websocket 5 (встроен в OBS 28+).
Снимается картинка стрима, а не мониторы: второй монитор с личными окнами
модель не видит, и неважно, на каком мониторе игра."""

import asyncio
import base64
import hashlib
import json

import websockets

# Перки и мелкий текст оверлея на 720p читаются хуже, а 1080p всего вдвое дороже.
DEFAULT_WIDTH = 1920
JPEG_QUALITY = 80
TIMEOUT_SECONDS = 5.0


class OBSError(Exception):
    pass


def _auth(password: str, salt: str, challenge: str) -> str:
    secret = base64.b64encode(hashlib.sha256((password + salt).encode()).digest()).decode()
    return base64.b64encode(hashlib.sha256((secret + challenge).encode()).digest()).decode()


class OBS:
    def __init__(self, url: str, password: str, width: int = DEFAULT_WIDTH):
        self.url = url
        self.password = password
        self.width = width

    async def screenshot(self) -> str:
        """Возвращает data URL с JPEG. Подключается заново на каждый снимок:
        снимки редкие, а OBS может перезапуститься между ними."""
        try:
            async with asyncio.timeout(TIMEOUT_SECONDS):
                async with websockets.connect(self.url, subprotocols=["obswebsocket.json"]) as socket:
                    await self._identify(socket)
                    scene = await self._request(socket, "GetCurrentProgramScene", {})
                    shot = await self._request(socket, "GetSourceScreenshot", {
                        "sourceName": scene["currentProgramSceneName"],
                        "imageFormat": "jpg",
                        "imageWidth": self.width,
                        "imageCompressionQuality": JPEG_QUALITY,
                    })
        except TimeoutError:
            raise OBSError("OBS не ответил вовремя") from None
        except (websockets.WebSocketException, OSError) as error:
            raise OBSError(f"нет связи с OBS ({error}), включи WebSocket-сервер в OBS") from None
        # OBS пишет нестандартный тип image/jpg.
        _, _, data = shot["imageData"].partition(",")
        return f"data:image/jpeg;base64,{data}"

    async def _identify(self, socket) -> None:
        hello = json.loads(await socket.recv())["d"]
        identify: dict = {"rpcVersion": 1, "eventSubscriptions": 0}
        auth = hello.get("authentication")
        if auth:
            if not self.password:
                raise OBSError("OBS просит пароль: укажи OBS_PASSWORD в .env")
            identify["authentication"] = _auth(self.password, auth["salt"], auth["challenge"])
        await socket.send(json.dumps({"op": 1, "d": identify}))
        try:
            reply = json.loads(await socket.recv())
        except websockets.ConnectionClosed as closed:
            raise OBSError(f"OBS не пустил: {closed.rcvd.reason if closed.rcvd else 'неверный пароль?'}") from None
        if reply.get("op") != 2:
            raise OBSError("OBS не подтвердил вход")

    async def _request(self, socket, kind: str, data: dict) -> dict:
        await socket.send(json.dumps({"op": 6, "d": {"requestType": kind, "requestId": kind, "requestData": data}}))
        while True:
            message = json.loads(await socket.recv())
            if message.get("op") == 7 and message["d"].get("requestId") == kind:
                break
        status = message["d"]["requestStatus"]
        if not status.get("result"):
            raise OBSError(f"{kind}: {status.get('comment') or status.get('code')}")
        return message["d"].get("responseData") or {}
