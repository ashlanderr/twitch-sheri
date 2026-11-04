import asyncio
import json
import os
import time
from pathlib import Path

import httpx

SCOPES = {
    "bot": "user:bot user:read:chat user:write:chat moderator:manage:banned_users",
    "broadcaster": "channel:manage:broadcast",
}
DEVICE_URL = "https://id.twitch.tv/oauth2/device"
TOKEN_URL = "https://id.twitch.tv/oauth2/token"
VALIDATE_URL = "https://id.twitch.tv/oauth2/validate"
REFRESH_MARGIN_SECONDS = 300


class TokenStore:
    """Токен одного аккаунта. Refresh-токен одноразовый, поэтому новый
    сохраняется на диск сразу после обновления."""

    def __init__(self, path: Path, client_id: str):
        self.path = path
        self.client_id = client_id
        self.data = json.loads(path.read_text(encoding="utf-8"))
        self._lock = asyncio.Lock()

    @property
    def user_id(self) -> str:
        return self.data["user_id"]

    @property
    def login(self) -> str:
        return self.data["login"]

    async def access_token(self, *, force_refresh: bool = False) -> str:
        async with self._lock:
            if force_refresh or time.time() > self.data["expires_at"] - REFRESH_MARGIN_SECONDS:
                await self._refresh()
            return self.data["access_token"]

    async def _refresh(self) -> None:
        async with httpx.AsyncClient(timeout=30) as http:
            response = await http.post(
                TOKEN_URL,
                data={
                    "client_id": self.client_id,
                    "grant_type": "refresh_token",
                    "refresh_token": self.data["refresh_token"],
                },
            )
        if response.status_code != 200:
            raise RuntimeError(
                f"Токен {self.path.stem} не обновился ({response.status_code}), "
                f"повтори вход: python -m sheri login {self.path.stem}"
            )
        self.data.update(_token_fields(response.json()))
        save_token(self.path, self.data)


def _token_fields(payload: dict) -> dict:
    return {
        "access_token": payload["access_token"],
        "refresh_token": payload["refresh_token"],
        "expires_at": time.time() + payload["expires_in"],
    }


def save_token(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    os.replace(temporary, path)


async def device_login(account: str, client_id: str, tokens_dir: Path) -> None:
    async with httpx.AsyncClient(timeout=30) as http:
        response = await http.post(
            DEVICE_URL, data={"client_id": client_id, "scopes": SCOPES[account]}
        )
        response.raise_for_status()
        device = response.json()
        print(f"Войди в аккаунт ({account}) и подтверди код: {device['verification_uri']}")

        deadline = time.monotonic() + device["expires_in"]
        while time.monotonic() < deadline:
            await asyncio.sleep(device["interval"])
            response = await http.post(
                TOKEN_URL,
                data={
                    "client_id": client_id,
                    "scopes": SCOPES[account],
                    "device_code": device["device_code"],
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
            )
            if response.status_code == 200:
                break
            if response.json().get("message") != "authorization_pending":
                raise RuntimeError(f"Вход не удался: {response.text}")
        else:
            raise RuntimeError("Код устарел, запусти вход заново")

        data = _token_fields(response.json())
        validated = await http.get(
            VALIDATE_URL, headers={"Authorization": f"OAuth {data['access_token']}"}
        )
        validated.raise_for_status()
        info = validated.json()

    save_token(tokens_dir / f"{account}.json", {**data, "user_id": info["user_id"], "login": info["login"]})
    print(f"Готово: {account} = {info['login']}")
