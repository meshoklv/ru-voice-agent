"""The brain. Every provider exposes ``stream(messages)``: a generator of text pieces.

Providers
- ``ollama``  : local models through Ollama's native /api/chat.
- ``openai``  : any OpenAI-compatible chat endpoint (OpenAI, Groq, DeepSeek, OpenRouter,
                vLLM, LM Studio, ...). Set OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL.
- ``echo``    : no network, repeats the last phrase (wiring test).
- ``mock``    : canned replies streamed at a fixed speed (fair A/B timing of the pipeline).

``VOICE_LLM=ollama,openai`` builds a fallback queue (see :class:`Chain`).
"""
import http.client
import json
import logging
import os
import threading
import time
import urllib.parse

from . import config

log = logging.getLogger("voice.llm")


def _conn(url: str, timeout: float):
    u = urllib.parse.urlparse(url)
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    return cls(u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=timeout), u


class _Keepalive:
    """Keeps one HTTP connection open: a new TLS handshake costs a noticeable
    fraction of a second on every turn."""

    def __init__(self, base_url: str, timeout: float):
        self._base, self._timeout = base_url, timeout
        self._c = None
        self._u = None
        self._lock = threading.Lock()

    def drop(self):
        try:
            if self._c:
                self._c.close()
        except Exception:                                      # noqa: BLE001
            pass
        self._c = None

    def warm(self) -> bool:
        with self._lock:
            try:
                if self._c is None:
                    self._c, self._u = _conn(self._base, self._timeout)
                if getattr(self._c, "sock", None) is None:
                    self._c.connect()
                return True
            except Exception:                                  # noqa: BLE001
                self.drop()
                return False

    def request(self, path, payload, headers):
        for attempt in (1, 2):
            try:
                if self._c is None:
                    self._c, self._u = _conn(self._base, self._timeout)
                self._c.request("POST", self._u.path.rstrip("/") + path,
                                body=payload, headers=headers)
                return self._c.getresponse()
            except (http.client.HTTPException, OSError):
                self.drop()
                if attempt == 2:
                    raise


class Ollama:
    name = "ollama"

    def __init__(self):
        self._k = _Keepalive(config.OLLAMA_URL, config.OLLAMA_TIMEOUT_S)
        self._lock = threading.Lock()

    def warm(self):
        return self._k.warm()

    def _set_read_timeout(self, seconds):
        sock = getattr(self._k._c, "sock", None)
        if sock is not None:
            try:
                sock.settimeout(seconds)
            except OSError:
                pass

    def stream(self, messages):
        body = {
            "model": config.OLLAMA_MODEL,
            "think": False,            # otherwise reasoning models think silently first
            "stream": True,
            # 32k window: a role prompt plus 12 remembered turns must not silently
            # push the beginning of the call out of context.
            "options": {"num_ctx": 32768, "temperature": config.LLM_TEMPERATURE,
                        "num_predict": config.LLM_MAX_TOKENS},
            "messages": messages,
        }
        payload = json.dumps(body).encode()
        hdr = {"Content-Type": "application/json"}
        with self._lock:
            info = None
            for attempt in (1, 2):
                got = False
                resp = self._k.request("/api/chat", payload, hdr)
                if resp.status != 200:
                    detail = resp.read()[:200].decode("utf-8", "replace")
                    self._k.drop()
                    raise RuntimeError("Ollama answered %s: %s" % (resp.status, detail))
                # First attempt: short deadline for the first word (a stalled server is
                # not worth waiting for). Second attempt: the full timeout, which covers
                # a cold model load.
                self._set_read_timeout(config.OLLAMA_FIRST_TOKEN_S if attempt == 1
                                       else config.OLLAMA_TIMEOUT_S)
                try:
                    done = False
                    for line in resp:
                        # read the stream to the very end, even after "done": leaving an
                        # unread answer in the connection breaks the next request
                        if not line.strip():
                            continue
                        d = json.loads(line)
                        if not done:
                            piece = d.get("message", {}).get("content", "")
                            if piece:
                                if not got:
                                    self._set_read_timeout(config.OLLAMA_TIMEOUT_S)
                                got = True
                                yield piece
                        if d.get("done"):
                            done = True
                            info = d
                except Exception as e:                         # noqa: BLE001
                    self._k.drop()
                    if not got and attempt == 1 and isinstance(e, TimeoutError):
                        log.warning("no first word in %.0f s, retrying on a fresh connection",
                                    config.OLLAMA_FIRST_TOKEN_S)
                        continue
                    if attempt == 2 or got:
                        raise
                    continue
                if got:
                    return
                log.warning("empty answer (attempt %d), reason: %s", attempt,
                            (info or {}).get("done_reason"))
                self._k.drop()
            raise RuntimeError("model returned an empty answer twice (reason: %s)"
                               % (info or {}).get("done_reason"))


class OpenAICompat:
    """OpenAI-style streaming chat completions."""
    name = "openai"

    def __init__(self):
        self._k = _Keepalive(config.OPENAI_BASE_URL, 60)
        self._lock = threading.Lock()

    def warm(self):
        return self._k.warm()

    def stream(self, messages):
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        body = {"model": config.OPENAI_MODEL, "messages": messages, "stream": True,
                "temperature": config.LLM_TEMPERATURE, "max_tokens": config.LLM_MAX_TOKENS}
        payload = json.dumps(body).encode()
        hdr = {"Content-Type": "application/json",
               "Authorization": "Bearer " + config.OPENAI_API_KEY}
        with self._lock:
            r = self._k.request("/chat/completions", payload, hdr)
            if r.status != 200:
                detail = r.read()[:200].decode("utf-8", "replace")
                self._k.drop()
                raise RuntimeError("LLM endpoint answered %s: %s" % (r.status, detail))
            got, refusal = False, ""
            try:
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        # some gateways refuse with HTTP 200 and a plain JSON body
                        if line and not refusal:
                            refusal = line[:200]
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        continue              # read to the end, do not leave data behind
                    try:
                        d = json.loads(data)
                    except ValueError:
                        continue
                    piece = ((d.get("choices") or [{}])[0].get("delta") or {}).get("content") or ""
                    if piece:
                        got = True
                        yield piece
            except Exception:                                  # noqa: BLE001
                self._k.drop()
                raise
            if not got:
                self._k.drop()
                raise RuntimeError("LLM said nothing: %s" % (refusal or "empty stream"))


class Echo:
    name = "echo"

    def stream(self, messages):
        yield "Вы сказали: " + messages[-1]["content"]


class Mock:
    """Canned replies streamed like a real model (first word delay, then N pieces/s).

    Lets you compare pipeline changes fairly: a cloud LLM answers in a different
    time on every call and hides differences of half a second.
    """
    name = "mock"
    REPLIES = [
        "Здравствуйте! Рады вас слышать. На какое время вас записать?",
        "Понял. Стрижка занимает около сорока минут. Вам удобно в субботу, в двенадцать?",
        "Отлично, записал вас. Если планы изменятся, просто позвоните нам.",
    ]

    def __init__(self):
        self._n = 0

    def stream(self, messages):
        text = self.REPLIES[self._n % len(self.REPLIES)]
        self._n += 1
        tps = float(os.environ.get("MOCK_TPS", "45"))
        time.sleep(float(os.environ.get("MOCK_FIRST_S", "1.2")))
        for i in range(0, len(text), 3):
            if i and tps > 0:
                time.sleep(1.0 / tps)
            yield text[i:i + 3]


class Chain:
    """Fallback queue: if the first brain is silent or fails, the next one answers.

    Switching happens ONLY while nothing has been said yet. If the first brain
    started speaking and broke off mid-sentence, starting over with another one would
    make the caller hear half of one sentence and then a whole different one.
    """
    name = "chain"

    def __init__(self, providers):
        self.providers = providers
        self.last_used = None          # who actually answered, for the call log

    def stream(self, messages):
        first_err = None
        for i, p in enumerate(self.providers):
            got = False
            try:
                for piece in p.stream(messages):
                    if not got:
                        self.last_used = p.name
                    got = True
                    yield piece
                if got:
                    return
                err = RuntimeError("%s returned an empty answer" % p.name)
            except Exception as e:                             # noqa: BLE001
                if got:
                    raise
                err = e
            first_err = first_err or err
            log.warning("brain '%s' failed (%s: %s)%s", p.name, type(err).__name__, err,
                        "" if i == len(self.providers) - 1 else ", trying the next one")
        raise first_err or RuntimeError("no brains in the queue")

    def warm(self):
        """Returns the state of the PRIMARY brain, not 'anyone at all'. Otherwise a
        sleeping primary would be hidden behind a healthy fallback."""
        main = False
        for i, p in enumerate(self.providers):
            ok = bool(p.warm()) if hasattr(p, "warm") else False
            log.info("  brain '%s': %s", p.name, "ready" if ok else "not responding")
            if i == 0:
                main = ok
        return main


PROVIDERS = {"ollama": Ollama, "openai": OpenAICompat, "echo": Echo, "mock": Mock}


def make(name=None):
    names = [n.strip() for n in (name or config.LLM_PROVIDER).split(",") if n.strip()]
    for n in names:
        if n not in PROVIDERS:
            raise RuntimeError("unknown LLM provider: %s (have: %s)" % (n, ", ".join(PROVIDERS)))
    if not names:
        raise RuntimeError("no LLM provider configured")
    if len(names) == 1:
        return PROVIDERS[names[0]]()
    return Chain([PROVIDERS[n]() for n in names])
