import asyncio
import json
import logging

import httpx
import websockets

from .events import Event, EventQueue, Role
from .twitch_auth import TokenStore

log = logging.getLogger(__name__)

HELIX = "https://api.twitch.tv/helix"
EVENTSUB_URL = "wss://eventsub.wss.twitch.tv/ws"
MODERATOR_BADGES = {"moderator", "lead_moderator"}
RECONNECT_DELAY_SECONDS = 5


def parse_chat_event(event: dict, bot_id: str) -> Event | None:
    """Роль берётся только из бейджей Twitch, а не из текста сообщения."""
    if event["chatter_user_id"] == bot_id:
        return None
    badges = {badge["set_id"] for badge in event.get("badges") or []}
    if "broadcaster" in badges:
        role = Role.BROADCASTER
    elif badges & MODERATOR_BADGES:
        role = Role.MODERATOR
    else:
        role = Role.VIEWER
    return Event("twitch", role, event["chatter_user_login"], event["message"]["text"])


class TwitchChannel:
    def __init__(self, client_id: str, bot: TokenStore, broadcaster: TokenStore):
        self.client_id = client_id
        self.bot = bot
        self.broadcaster = broadcaster
        self.http = httpx.AsyncClient(base_url=HELIX, timeout=30)

    async def _request(self, token: TokenStore, method: str, path: str, **kwargs) -> httpx.Response:
        for force_refresh in (False, True):
            access_token = await token.access_token(force_refresh=force_refresh)
            headers = {"Authorization": f"Bearer {access_token}", "Client-Id": self.client_id}
            response = await self.http.request(method, path, headers=headers, **kwargs)
            if response.status_code != 401:
                break
        if response.is_error:
            raise RuntimeError(f"Twitch {method} {path}: {response.status_code} {response.text}")
        return response

    async def send_chat(self, text: str) -> None:
        response = await self._request(
            self.bot, "POST", "/chat/messages",
            json={"broadcaster_id": self.broadcaster.user_id, "sender_id": self.bot.user_id, "message": text},
        )
        result = response.json()["data"][0]
        if not result["is_sent"]:
            raise RuntimeError(f"Twitch не принял сообщение: {result.get('drop_reason')}")

    async def _user_id(self, login: str) -> str:
        response = await self._request(self.bot, "GET", "/users", params={"login": login})
        users = response.json()["data"]
        if not users:
            raise RuntimeError(f"пользователя {login} нет")
        return users[0]["id"]

    async def _ban(self, login: str, reason: str, seconds: int | None) -> None:
        data = {"user_id": await self._user_id(login), "reason": reason[:500]}
        if seconds is not None:
            data["duration"] = seconds
        await self._request(
            self.bot, "POST", "/moderation/bans",
            params={"broadcaster_id": self.broadcaster.user_id, "moderator_id": self.bot.user_id},
            json={"data": data},
        )

    async def timeout_user(self, login: str, seconds: int, reason: str) -> None:
        await self._ban(login, reason, seconds)

    async def ban_user(self, login: str, reason: str) -> None:
        await self._ban(login, reason, None)

    async def set_title(self, title: str) -> None:
        await self._request(
            self.broadcaster, "PATCH", "/channels",
            params={"broadcaster_id": self.broadcaster.user_id}, json={"title": title},
        )

    async def get_channel(self) -> str:
        response = await self._request(
            self.bot, "GET", "/channels", params={"broadcaster_id": self.broadcaster.user_id}
        )
        info = response.json()["data"][0]
        return f"название: {info['title']}; игра: {info['game_name'] or 'не указана'}"

    async def read_chat(self, queue: EventQueue) -> None:
        url = EVENTSUB_URL
        # После session_reconnect подписки сохраняются на новом соединении.
        subscribe = True
        while True:
            try:
                next_url = await self._listen(url, subscribe, queue)
            except (websockets.WebSocketException, OSError, TimeoutError) as error:
                log.warning("EventSub отключился: %s", error)
                next_url = None
            if next_url:
                url, subscribe = next_url, False
            else:
                url, subscribe = EVENTSUB_URL, True
                await asyncio.sleep(RECONNECT_DELAY_SECONDS)

    async def _listen(self, url: str, subscribe: bool, queue: EventQueue) -> str | None:
        """Возвращает адрес нового соединения, если Twitch попросил переподключиться."""
        async with websockets.connect(url) as socket:
            keepalive = 30.0
            seen: set[str] = set()
            while True:
                message = json.loads(await asyncio.wait_for(socket.recv(), keepalive + 10))
                kind = message["metadata"]["message_type"]
                payload = message["payload"]
                if kind == "session_welcome":
                    keepalive = float(payload["session"]["keepalive_timeout_seconds"])
                    if subscribe:
                        await self._subscribe(payload["session"]["id"])
                        log.info("Подключён к чату %s как %s", self.broadcaster.login, self.bot.login)
                elif kind == "session_reconnect":
                    return payload["session"]["reconnect_url"]
                elif kind == "notification":
                    message_id = message["metadata"]["message_id"]
                    if message_id in seen:
                        continue
                    seen.add(message_id)
                    event = parse_chat_event(payload["event"], self.bot.user_id)
                    if event:
                        queue.put(event)

    async def _subscribe(self, session_id: str) -> None:
        await self._request(
            self.bot, "POST", "/eventsub/subscriptions",
            json={
                "type": "channel.chat.message",
                "version": "1",
                "condition": {
                    "broadcaster_user_id": self.broadcaster.user_id,
                    "user_id": self.bot.user_id,
                },
                "transport": {"method": "websocket", "session_id": session_id},
            },
        )
