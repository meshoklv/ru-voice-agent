"""Web server: takes microphone audio over a WebSocket, answers with speech.

Security
- Listens on 127.0.0.1 by default. Put a TLS reverse proxy in front of it to expose it.
- EVERY endpoint (page, API, WebSocket) requires the access token. There is no default
  password: if VOICE_TOKEN is empty a random token is generated into data/token.
- It serves only audio and text. No shell, no file access, no uploads to disk.
- The access log is off on purpose: the token travels in the URL.
"""
import asyncio
import base64
import hmac
import json
import logging
import time
from pathlib import Path

from aiohttp import WSMsgType, web

from . import calllog as calllog_mod
from . import chunking, config
from . import llm as llm_mod
from . import tts as tts_mod
from . import vad as vad_mod
from .resample import Resampler

log = logging.getLogger("voice")


def _asr():
    """Imported lazily: the T-one package is heavy and not needed to test the server."""
    from . import asr
    return asr
HERE = Path(__file__).parent
CALL_NO = [0]
GREET_CACHE = {}


# ------------------------------------------------------------------- auth ---
def authorized(request) -> bool:
    """Token from ``?k=`` or from the cookie. Constant-time comparison: a plain ``==``
    stops at the first different character and leaks the token through timing."""
    if not config.VOICE_TOKEN:
        return False
    given = request.query.get("k") or request.cookies.get(config.TOKEN_COOKIE) or ""
    return hmac.compare_digest(given.encode(), config.VOICE_TOKEN.encode())


def deny():
    return web.Response(status=403, text="403: access token required\n",
                        content_type="text/plain", charset="utf-8")


class State:
    def __init__(self):
        self.recognizer = None
        self.brain = None
        self.system_prompt = config.load_system_prompt()

    async def warmup(self):
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        self.recognizer = await loop.run_in_executor(None, _asr().Recognizer)
        log.info("speech recognition ready in %.1f s", time.perf_counter() - t0)
        self.brain = llm_mod.make()
        voice = tts_mod.make()
        for who, obj in (("brain", self.brain), ("voice", voice)):
            if hasattr(obj, "warm"):
                ok = await loop.run_in_executor(None, obj.warm)
                log.info("%s warm: %s", who, "yes" if ok else "NO (check the configuration)")

    async def prewarm_greeting(self):
        """Synthesise the greeting for each voice in the background."""
        loop = asyncio.get_running_loop()
        spoken = tts_mod.prepare(config.greeting())
        for v in tts_mod.voices():
            if v["id"] in GREET_CACHE:
                continue
            try:
                GREET_CACHE[v["id"]] = await loop.run_in_executor(
                    None, tts_mod.synth, spoken, v["id"])
            except Exception as e:                             # noqa: BLE001
                log.warning("greeting for voice %s failed: %s", v["id"], e)


# ------------------------------------------------------------- http routes --
async def index(request):
    if not authorized(request):
        return deny()
    resp = web.FileResponse(HERE / "static" / "index.html")
    if request.query.get("k"):
        resp.set_cookie(config.TOKEN_COOKIE, request.query["k"], httponly=True,
                        samesite="Strict", max_age=86400 * 30)
    return resp


async def health(request):
    if not authorized(request):
        return deny()
    st = request.app[STATE_KEY]
    return web.json_response({
        "ok": True, "llm": config.LLM_PROVIDER, "tts": config.TTS_ENGINE,
        "brain_used": getattr(st.brain, "last_used", None),
        "tts_cache": tts_mod.cache_stats()})


async def voices(request):
    if not authorized(request):
        return deny()
    return web.json_response({"voices": tts_mod.voices(), "current": tts_mod.default_voice(),
                              "agent": config.MANAGER_NAME})


# ---------------------------------------------------------------- WebSocket --
async def ws_handler(request):
    if not authorized(request):
        return deny()                 # refuse BEFORE the upgrade
    state: State = request.app[STATE_KEY]
    ws = web.WebSocketResponse(heartbeat=25, max_msg_size=4 * 1024 * 1024)
    await ws.prepare(request)
    loop = asyncio.get_running_loop()

    history = []
    stream = await loop.run_in_executor(None, _asr().Stream, state.recognizer)
    said_parts = []
    chosen_voice = None
    default_voice = tts_mod.default_voice()
    clog = None
    resampler = None
    detector = vad_mod.make(8000) if config.VAD_ENABLED else None
    barge = {"stop": False}           # the caller started talking over the answer

    async def say(obj):
        if not ws.closed:
            await ws.send_json(obj)

    async def run_turn(said, t_end, unsure=False, conf=None):
        """One turn: think, then speak sentence by sentence while the model is writing."""
        timings = {"stt_ms": round((time.perf_counter() - t_end) * 1000)}
        messages = [{"role": "system", "content": state.system_prompt}]
        if unsure:
            messages.append({"role": "system", "content": config.UNSURE_HINT})
        messages += history[-config.MAX_TURNS:] + [{"role": "user", "content": said}]

        queue: asyncio.Queue = asyncio.Queue()

        def pump():
            try:
                for piece in state.brain.stream(messages):
                    loop.call_soon_threadsafe(queue.put_nowait, piece)
            except Exception as e:                             # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        barge["stop"] = False
        t0 = time.perf_counter()
        loop.run_in_executor(None, pump)

        # Synthesis is a PIPELINE: neighbouring sentences are synthesised overlapped but
        # sent strictly in order. The first sentence runs alone: on a shared CPU a second
        # job steals cores from it and makes the pause before the answer longer.
        buf, full, seq, first_audio, cuts = "", "", 0, None, 0
        spoken_all = []
        room = asyncio.Semaphore(max(1, config.TTS_PARALLEL))
        outbox: asyncio.Queue = asyncio.Queue()
        head_done = asyncio.Event()
        jobs = []

        async def voice_of(text, is_first=False):
            if not is_first:
                await head_done.wait()
            async with room:
                if barge["stop"]:
                    return None
                ta = time.perf_counter()
                try:
                    audio, mime = await loop.run_in_executor(
                        None, tts_mod.synth, text, chosen_voice)
                finally:
                    if is_first:
                        head_done.set()
                return audio, mime, round((time.perf_counter() - ta) * 1000)

        async def postman():
            nonlocal seq, first_audio
            while True:
                item = await outbox.get()
                if item is None:
                    return
                text, job = item
                try:
                    ready = await job
                except Exception as e:                         # noqa: BLE001
                    log.warning("synthesis failed: %s", e)
                    if clog:
                        clog.fail("tts", e)
                    await say({"type": "tts_failed", "text": text, "message": str(e)})
                    continue
                if ready is None or barge["stop"]:
                    continue
                audio, mime, tts_ms = ready
                seq += 1
                if first_audio is None:
                    first_audio = round((time.perf_counter() - t_end) * 1000)
                    timings["first_audio_ms"] = first_audio
                await say({"type": "audio", "seq": seq, "mime": mime, "text": text,
                           "engine": config.TTS_ENGINE, "tts_ms": tts_ms,
                           "b64": base64.b64encode(audio).decode()})

        postman_task = asyncio.create_task(postman())

        async def stop_postman():
            try:
                await outbox.put(None)
                await postman_task
            finally:
                head_done.set()
                for e in await asyncio.gather(*jobs, return_exceptions=True):
                    if isinstance(e, Exception):
                        log.warning("synthesis failed (nobody was listening): %s", e)
                        if clog:
                            clog.fail("tts", e)

        while True:
            piece = await queue.get()
            if isinstance(piece, Exception):
                if clog:
                    clog.fail("llm", piece)
                # report at once, before the synthesis queue drains
                await say({"type": "error", "message": "LLM unavailable: %s: %s"
                           % (type(piece).__name__, piece)})
                await stop_postman()
                return
            done = piece is None
            if not done:
                if not full:
                    timings["llm_first_ms"] = round((time.perf_counter() - t0) * 1000)
                buf += piece
                full += piece
            while not barge["stop"]:
                ready, buf = chunking.pop_speakable(buf, force=done, first=(cuts == 0))
                if not ready:
                    break
                spoken = tts_mod.prepare(ready)
                spoken_all.append(spoken)
                cuts += 1
                job = asyncio.ensure_future(voice_of(spoken, is_first=(cuts == 1)))
                jobs.append(job)
                outbox.put_nowait((spoken, job))
            if done or barge["stop"]:
                break

        # take the LLM time BEFORE waiting for synthesis, or it would include TTS time
        timings["llm_ms"] = round((time.perf_counter() - t0) * 1000)
        used = getattr(state.brain, "last_used", None)
        if used:
            timings["brain"] = used
        await stop_postman()
        timings["total_ms"] = round((time.perf_counter() - t_end) * 1000)
        history.extend([{"role": "user", "content": said},
                        {"role": "assistant", "content": full.strip()}])
        del history[:-2 * config.MAX_TURNS]
        if clog:
            clog.turn(said, full.strip(), " ".join(spoken_all), timings,
                      conf=conf, unsure=unsure)
        await say({"type": "done", "reply": full.strip(), "timings": timings})

    turn = {"task": None}

    async def guarded(coro):
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as e:                                 # noqa: BLE001
            log.exception("turn failed")
            if clog:
                clog.fail("turn", e)
            await say({"type": "error", "message": "internal error: %s" % type(e).__name__})

    async def finish_turn():
        """Make sure the previous answer is over (stop it if it is still being spoken)."""
        t = turn["task"]
        if t is not None and not t.done():
            barge["stop"] = True
            await t
        turn["task"] = None

    async def start_call():
        nonlocal clog, said_parts
        history.clear()
        said_parts = []
        CALL_NO[0] += 1
        if clog:
            clog.finish()
        clog = calllog_mod.CallLog(CALL_NO[0], {
            "llm": config.LLM_PROVIDER, "tts": config.TTS_ENGINE,
            "scenario": config.SCENARIO, "prompt_chars": len(state.system_prompt)})
        g = config.greeting()
        await say({"type": "said", "text": ""})
        try:
            t0 = time.perf_counter()
            key = chosen_voice or default_voice or ""
            if key not in GREET_CACHE:
                GREET_CACHE[key] = await loop.run_in_executor(
                    None, tts_mod.synth, tts_mod.prepare(g), chosen_voice)
            audio, mime = GREET_CACHE[key]
            ms = round((time.perf_counter() - t0) * 1000)
            clog.write("greeting", text=g, tts_ms=ms)
            await say({"type": "audio", "seq": 1, "mime": mime, "text": g,
                       "engine": config.TTS_ENGINE, "tts_ms": ms,
                       "b64": base64.b64encode(audio).decode()})
        except Exception as e:                                 # noqa: BLE001
            log.warning("greeting failed: %s", e)
            await say({"type": "tts_failed", "text": g, "message": str(e)})
        history.append({"role": "assistant", "content": g})
        await say({"type": "done", "reply": g, "timings": {"note": "greeting"}})
        await say({"type": "listen"})

    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                data = msg.data
                if resampler is not None:
                    data = await loop.run_in_executor(None, resampler.feed, data)
                    if not data:
                        continue
                if detector is not None:
                    if await loop.run_in_executor(None, detector.feed, data):
                        if clog:
                            clog.write("turn_end", by="silero")
                        await say({"type": "turn_end"})
                phrases = await loop.run_in_executor(None, stream.feed, data)
                if phrases:
                    said_parts += phrases
                    await say({"type": "partial", "text": " ".join(said_parts)})
            elif msg.type == WSMsgType.TEXT:
                cmd = json.loads(msg.data)
                kind = cmd.get("type")
                if kind == "end":
                    t_end = time.perf_counter()
                    said_parts += await loop.run_in_executor(None, stream.finish)
                    said = " ".join(p.strip() for p in said_parts if p.strip()).strip()
                    said_parts = []
                    if detector is not None:
                        detector.reset()
                    conf = stream.take_confidence()
                    unsure = bool(conf is not None and conf < config.ASR_UNSURE_BELOW)
                    await say({"type": "said", "text": said, "conf": conf, "unsure": unsure})
                    if said:
                        # run the turn as a task: the reader loop must stay free to
                        # receive "barge" (the caller talks over the answer) while it runs
                        await finish_turn()
                        turn["task"] = asyncio.create_task(
                            guarded(run_turn(said, t_end, unsure=unsure, conf=conf)))
                    else:
                        await say({"type": "done", "reply": "",
                                   "timings": {"note": "no speech recognised"}})
                elif kind == "call":
                    await finish_turn()
                    await start_call()
                elif kind == "barge":
                    barge["stop"] = True
                    if clog:
                        clog.write("barge")
                elif kind == "rate":
                    # Sample-rate conversion happens HERE with a proper filter. Doing it in
                    # the browser without one made recognition lose and invent words.
                    hz = int(cmd.get("hz") or 0)
                    resampler = Resampler(hz) if 8000 < hz <= 96000 else None
                    await say({"type": "rate_ok", "hz": hz})
                elif kind == "voice":
                    # only voices from the list: otherwise the page could ask the
                    # server to speak with any voice id on your account
                    wanted = str(cmd.get("id") or "")
                    if any(v["id"] == wanted for v in tts_mod.voices()):
                        chosen_voice = wanted
                        await say({"type": "voice_ok", "id": wanted})
                    else:
                        await say({"type": "error", "message": "unknown voice"})
                elif kind == "reset":
                    await finish_turn()
                    history.clear()
                    said_parts = []
                    stream = await loop.run_in_executor(None, _asr().Stream, state.recognizer)
                    await say({"type": "reset_ok"})
            elif msg.type == WSMsgType.ERROR:
                break
    finally:
        t = turn["task"]
        if t is not None and not t.done():
            barge["stop"] = True
            t.cancel()
        if clog:
            clog.finish()
        if not ws.closed:
            await ws.close()
    return ws


STATE_KEY = web.AppKey("state", State)


def make_app(state=None) -> web.Application:
    app = web.Application(client_max_size=12 * 1024 * 1024)
    app[STATE_KEY] = state or State()
    app.router.add_get("/", index)
    app.router.add_get("/api/health", health)
    app.router.add_get("/api/voices", voices)
    app.router.add_get("/ws", ws_handler)
    return app


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    app = make_app()

    async def on_start(_):
        await app[STATE_KEY].warmup()
        asyncio.create_task(app[STATE_KEY].prewarm_greeting())
        log.info("ready: http://%s:%s/?k=<token>   (token: %s)", config.HOST, config.PORT,
                 "from VOICE_TOKEN" if config._env("VOICE_TOKEN") else str(config.DATA_DIR / "token"))

    app.on_startup.append(on_start)
    web.run_app(app, host=config.HOST, port=config.PORT, print=None, access_log=None)


if __name__ == "__main__":
    main()
