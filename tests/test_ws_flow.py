"""Full WebSocket conversation with fake recogniser / brain / voice (no models needed)."""
import asyncio
import types

from aiohttp.test_utils import TestClient, TestServer

from voice_agent import config, llm, server, tts


class FakeStream:
    confs = []

    def __init__(self, recognizer):
        pass

    def feed(self, pcm):
        return []

    def finish(self):
        return ["запишите", "меня", "на", "завтра"]

    def take_confidence(self):
        return 0.99


class FakeState:
    system_prompt = "test"
    recognizer = object()
    brain = llm.make("echo")


def _fake_asr():
    return types.SimpleNamespace(Stream=FakeStream, Recognizer=object)


def test_call_then_turn(monkeypatch):
    monkeypatch.setattr(server, "_asr", _fake_asr)
    monkeypatch.setattr(config, "VAD_ENABLED", False)
    monkeypatch.setattr(tts, "synth", lambda text, voice=None: (b"AUDIO:" + text.encode(), "audio/wav"))

    async def collect(ws, until):
        seen = []
        while True:
            msg = await asyncio.wait_for(ws.receive_json(), 5)
            seen.append(msg)
            if msg["type"] == until:
                return seen

    async def go():
        c = TestClient(TestServer(server.make_app(FakeState())))
        await c.start_server()
        try:
            ws = await c.ws_connect("/ws?k=" + config.VOICE_TOKEN)
            await ws.send_json({"type": "call"})
            seen = await collect(ws, "listen")
            assert [m["type"] for m in seen if m["type"] == "audio"] == ["audio"]
            assert seen[-2]["type"] == "done"

            await ws.send_json({"type": "end"})
            seen = await collect(ws, "done")
            said = [m for m in seen if m["type"] == "said"][0]
            assert said["text"] == "запишите меня на завтра"
            audio = [m for m in seen if m["type"] == "audio"]
            assert audio and audio[0]["engine"] == config.TTS_ENGINE
            done = seen[-1]
            assert done["reply"] == "Вы сказали: запишите меня на завтра"
            assert "first_audio_ms" in done["timings"]
            await ws.close()
        finally:
            await c.close()

    asyncio.run(go())


def test_unknown_voice_is_rejected(monkeypatch):
    monkeypatch.setattr(server, "_asr", _fake_asr)
    monkeypatch.setattr(config, "VAD_ENABLED", False)

    async def go():
        c = TestClient(TestServer(server.make_app(FakeState())))
        await c.start_server()
        try:
            ws = await c.ws_connect("/ws?k=" + config.VOICE_TOKEN)
            await ws.send_json({"type": "voice", "id": "someone-elses-voice"})
            msg = await asyncio.wait_for(ws.receive_json(), 5)
            assert msg["type"] == "error"
            await ws.close()
        finally:
            await c.close()

    asyncio.run(go())
