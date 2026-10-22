"""Every endpoint must refuse requests without the token, including the WebSocket upgrade."""
import asyncio

from aiohttp.test_utils import TestClient, TestServer

from voice_agent import config, server


class FakeState:
    system_prompt = "x"
    recognizer = None
    brain = None


def _run(coro):
    return asyncio.run(coro)


async def _client():
    c = TestClient(TestServer(server.make_app(FakeState())))
    await c.start_server()
    return c


def test_everything_refuses_without_token():
    async def go():
        c = await _client()
        try:
            for path in ("/", "/api/health", "/api/voices", "/ws"):
                r = await c.get(path)
                assert r.status == 403, path
                r = await c.get(path + "?k=wrong")
                assert r.status == 403, path
        finally:
            await c.close()
    _run(go())


def test_token_in_query_is_accepted_and_sets_cookie():
    async def go():
        c = await _client()
        try:
            r = await c.get("/?k=" + config.VOICE_TOKEN)
            assert r.status == 200
            assert config.TOKEN_COOKIE in r.cookies
            r = await c.get("/api/voices")                 # cookie jar now has the token
            assert r.status == 200
        finally:
            await c.close()
    _run(go())


def test_no_token_configured_means_nobody_gets_in(monkeypatch):
    monkeypatch.setattr(config, "VOICE_TOKEN", "")

    async def go():
        c = await _client()
        try:
            assert (await c.get("/?k=")).status == 403
        finally:
            await c.close()
    _run(go())


def test_default_bind_is_loopback():
    assert config.HOST == "127.0.0.1"
