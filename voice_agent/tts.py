"""Speech synthesis behind one function: ``synth(text, voice) -> (audio_bytes, mime)``.

Engines (``VOICE_TTS``)
- ``piper``  : local, CPU, no API key. Default. Voices are downloaded by
               ``scripts/download_models.py``.
- ``kokoro`` : local, CPU (kokoro-onnx). Kokoro has no official Russian voices; Russian
               goes through espeak phonemes and the stock voices have an accent.
               Experimental.
- ``fish``   : Fish Audio cloud API with your own reference voice id. Needs FISH_API_KEY.

See the companion benchmark (ru-speech-bench) for how these and ten other engines compare.
"""
import collections
import http.client
import io
import json
import logging
import threading
import wave

from . import config
from .textnorm import prepare  # noqa: F401  (re-exported for the server)

log = logging.getLogger("voice.tts")


# ---------------------------------------------------------------- Piper ------
class Piper:
    name = "piper"
    mime = "audio/wav"

    def __init__(self):
        self._voices = {}
        self._lock = threading.Lock()

    @staticmethod
    def available():
        d = config.MODELS_DIR / "piper"
        return sorted(p.stem for p in d.glob("*.onnx")) if d.is_dir() else []

    def _voice(self, name):
        from piper import PiperVoice
        with self._lock:
            if name not in self._voices:
                path = config.MODELS_DIR / "piper" / (name + ".onnx")
                if not path.exists():
                    raise RuntimeError("Piper voice %s not found in %s "
                                       "(run scripts/download_models.py)" % (name, path.parent))
                self._voices[name] = PiperVoice.load(str(path))
            return self._voices[name]

    def warm(self):
        try:
            self._voice(config.PIPER_VOICE)
            return True
        except Exception as e:                                 # noqa: BLE001
            log.warning("Piper is not ready: %s", e)
            return False

    def synth(self, text, voice=None):
        v = self._voice(voice or config.PIPER_VOICE)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            v.synthesize_wav(text, wf)
        return buf.getvalue()


# --------------------------------------------------------------- Kokoro ------
class Kokoro:
    name = "kokoro"
    mime = "audio/wav"

    def __init__(self):
        self._k = None
        self._lock = threading.Lock()

    def _engine(self):
        with self._lock:
            if self._k is None:
                from kokoro_onnx import Kokoro as K
                d = config.MODELS_DIR / "kokoro"
                self._k = K(str(d / "kokoro-v1.0.onnx"), str(d / "voices-v1.0.bin"))
            return self._k

    def warm(self):
        try:
            self._engine()
            return True
        except Exception as e:                                 # noqa: BLE001
            log.warning("Kokoro is not ready: %s", e)
            return False

    def synth(self, text, voice=None):
        import soundfile as sf
        audio, sr = self._engine().create(text, voice=voice or config.KOKORO_VOICE, lang="ru")
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        return buf.getvalue()


# ----------------------------------------------------------------- Fish ------
class Fish:
    """Fish Audio cloud TTS. One persistent connection (a new TLS handshake per phrase
    is expensive). Streaming works only with ``latency: low``."""
    name = "fish"
    mime = "audio/mpeg"

    def __init__(self):
        if not config.FISH_API_KEY:
            raise RuntimeError("FISH_API_KEY is not set")
        self._conn = None
        self._lock = threading.Lock()

    def _open(self):
        return http.client.HTTPSConnection("api.fish.audio", timeout=config.FISH_TIMEOUT_S)

    def _drop(self):
        try:
            if self._conn:
                self._conn.close()
        except Exception:                                      # noqa: BLE001
            pass
        self._conn = None

    def warm(self):
        with self._lock:
            try:
                self._conn = self._conn or self._open()
                if getattr(self._conn, "sock", None) is None:
                    self._conn.connect()
                return True
            except Exception:                                  # noqa: BLE001
                self._drop()
                return False

    def synth(self, text, voice=None):
        body = {"text": text, "format": "mp3", "latency": "low", "normalize": True}
        vid = voice or config.FISH_VOICE_ID
        if vid:
            body["reference_id"] = vid
        payload = json.dumps(body).encode()
        with self._lock:
            for attempt in (1, 2):
                try:
                    if self._conn is None:
                        self._conn = self._open()
                    self._conn.request("POST", "/v1/tts", body=payload, headers={
                        "Authorization": "Bearer " + config.FISH_API_KEY,
                        "Content-Type": "application/json",
                        "model": config.FISH_MODEL})   # the model goes in a HEADER
                    r = self._conn.getresponse()
                    data = r.read()
                    if r.status != 200:
                        self._drop()
                        raise RuntimeError("Fish answered %s: %s" % (
                            r.status, data[:200].decode("utf-8", "replace")))
                    return data
                except RuntimeError:
                    raise
                except Exception as e:                         # noqa: BLE001
                    self._drop()
                    if attempt == 2:
                        raise RuntimeError("Fish unavailable: %s: %s" % (type(e).__name__, e))


_ENGINES = {"piper": Piper, "kokoro": Kokoro, "fish": Fish}
_engine = None
_engine_lock = threading.Lock()


def make():
    global _engine
    with _engine_lock:
        if _engine is None:
            if config.TTS_ENGINE not in _ENGINES:
                raise RuntimeError("unknown VOICE_TTS=%s (have: %s)"
                                   % (config.TTS_ENGINE, ", ".join(_ENGINES)))
            _engine = _ENGINES[config.TTS_ENGINE]()
        return _engine


# ------------------------------------------------------------- phrase cache --
# Short phrases repeat ("Okay.", the greeting); synthesising them again is wasted time.
_said = collections.OrderedDict()
_said_lock = threading.Lock()
_stats = [0, 0]                                  # hits, misses


def cache_stats():
    return {"hits": _stats[0], "misses": _stats[1], "size": len(_said)}


def _cacheable(text):
    return config.TTS_CACHE_MAX > 0 and len(text) <= config.TTS_CACHE_MAX_CHARS


def _cache_get(key):
    if not _cacheable(key[1]):
        return None
    with _said_lock:
        got = _said.get(key)
        if got is not None:
            _said.move_to_end(key)
            _stats[0] += 1
        else:
            _stats[1] += 1
        return got


def _cache_put(key, value):
    # Never cache empty audio: a one-off empty answer would silence the phrase forever.
    if not _cacheable(key[1]) or not value[0]:
        return
    with _said_lock:
        _said[key] = value
        while len(_said) > config.TTS_CACHE_MAX:
            _said.popitem(last=False)


def synth(text: str, voice: str = None):
    """Returns (audio_bytes, mime). The cache key uses the REAL voice, not the argument."""
    voice = voice or default_voice() or ""
    key = (config.TTS_ENGINE, voice, text)
    got = _cache_get(key)
    if got is not None:
        return got
    e = make()
    out = (e.synth(text, voice or None), e.mime)
    _cache_put(key, out)
    return out


# ------------------------------------------------------------------ voices --
def voices():
    """Voices offered in the page. The first one is the default.

    piper: every .onnx found in models/piper. kokoro: the configured voice.
    fish: entries of voices.json ([{"id","name","note"}]) or FISH_VOICE_ID.
    """
    out = []
    if config.TTS_ENGINE == "piper":
        for n in Piper.available():
            out.append({"id": n, "name": n.replace("ru_RU-", "").replace("-medium", ""),
                        "note": "Piper, local"})
        out.sort(key=lambda v: v["id"] != config.PIPER_VOICE)   # configured voice first
    elif config.TTS_ENGINE == "kokoro":
        out.append({"id": config.KOKORO_VOICE, "name": config.KOKORO_VOICE,
                    "note": "Kokoro, local"})
    else:
        try:
            data = json.loads(config.VOICES_FILE.read_text(encoding="utf-8"))
            for v in (data.get("voices", data) if isinstance(data, dict) else data):
                if v.get("id"):
                    out.append({"id": v["id"], "name": v.get("name") or v["id"],
                                "note": v.get("note", "")})
        except (OSError, ValueError):
            pass
        if not out and config.FISH_VOICE_ID:
            out.append({"id": config.FISH_VOICE_ID, "name": "Default", "note": "FISH_VOICE_ID"})
    return out


def default_voice():
    v = voices()
    return v[0]["id"] if v else None
