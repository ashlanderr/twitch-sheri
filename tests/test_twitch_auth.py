import json
import time

import httpx
import pytest

from sheri import twitch_auth
from sheri.twitch_auth import TokenStore, import_token


@pytest.fixture
def twitch(monkeypatch):
    """Подменяет сеть: запросы попадают в handler, ответы берутся из routes."""
    requests, routes = [], {}

    def handler(request):
        requests.append(request)
        status, body = routes[str(request.url)]
        return httpx.Response(status, json=body)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        twitch_auth.httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(handler))
    )
    return requests, routes


def validated(scopes):
    return {"client_id": "ttg-app", "login": "sheri_cat", "user_id": "42", "scopes": scopes, "expires_in": 14000}


async def test_import_saves_token_and_returns_client_id(tmp_path, twitch):
    _, routes = twitch
    routes[twitch_auth.VALIDATE_URL] = (200, validated(twitch_auth.SCOPES["bot"].split()))

    client_id = await import_token("bot", "access", "refresh", tmp_path)

    saved = json.loads((tmp_path / "bot.json").read_text(encoding="utf-8"))
    assert client_id == "ttg-app"
    assert saved["login"] == "sheri_cat"
    assert saved["source"] == "ttg"
    assert saved["expires_at"] > time.time() + 13000


async def test_import_rejects_token_without_needed_scopes(tmp_path, twitch):
    _, routes = twitch
    routes[twitch_auth.VALIDATE_URL] = (200, validated(["user:read:chat"]))

    with pytest.raises(RuntimeError, match="user:write:chat"):
        await import_token("bot", "access", "refresh", tmp_path)
    assert not (tmp_path / "bot.json").exists()


async def test_imported_token_refreshes_through_ttg(tmp_path, twitch):
    requests, routes = twitch
    path = tmp_path / "bot.json"
    path.write_text(json.dumps({"access_token": "old", "refresh_token": "r1", "expires_at": 0, "source": "ttg"}))
    routes[twitch_auth.TTG_REFRESH_URL] = (
        200,
        {"access_token": "new", "refresh_token": "r2", "expires_in": 3600, "success": True},
    )

    assert await TokenStore(path, "ttg-app").access_token() == "new"
    assert json.loads(requests[0].content) == {"refresh_token": "r1"}
    assert json.loads(path.read_text(encoding="utf-8"))["refresh_token"] == "r2"
