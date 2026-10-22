"""End-of-turn detection with Silero VAD (v5 ONNX), local, CPU.

Why on the server and not in the browser: shipping ~12 MB of runtime and model to
a phone over mobile data is worse than streaming audio that arrives here anyway.
Cost is a fraction of a millisecond per chunk.

The loudness threshold in the page stays as a fallback: whoever decides first that
the turn is over, wins.
"""
import threading

import numpy as np
import onnxruntime as ort

from . import config

BLOCK = {8000: 256, 16000: 512}      # samples per model call
# Silero v5 expects each block TOGETHER WITH the tail of the previous one: 64 samples
# at 16 kHz, 32 at 8 kHz. Without the tail it barely sees speech (max probability
# dropped to 0.10 instead of 0.99 in our test).
CONTEXT = {8000: 32, 16000: 64}


class SpeechDetector:
    """One instance per call. Tracks whether someone is speaking or silent."""

    def __init__(self, sess: ort.InferenceSession, rate: int = 8000):
        if rate not in BLOCK:
            raise RuntimeError("Silero VAD supports 8000 or 16000 Hz, not %s" % rate)
        self._sess = sess
        self._rate = rate
        self._block = BLOCK[rate]
        self._ctx = np.zeros(CONTEXT[rate], dtype=np.float32)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._buf = np.zeros(0, dtype=np.float32)
        self.reset()

    def reset(self):
        """Prepares the detector for the next turn without touching the model."""
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._spoke = False
        self.last_prob = 0.0

    def feed(self, pcm16: bytes) -> bool:
        """Takes 16-bit PCM at 8 kHz. True = the turn has just ended."""
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        self._buf = np.concatenate([self._buf, x])
        ended = False
        block_ms = self._block * 1000.0 / self._rate
        while len(self._buf) >= self._block:
            block, self._buf = self._buf[:self._block], self._buf[self._block:]
            inp = np.concatenate([self._ctx, block])
            self._ctx = block[-len(self._ctx):]
            prob, self._state = self._sess.run(
                None, {"input": inp[None, :], "state": self._state,
                       "sr": np.array(self._rate, dtype=np.int64)})
            p = float(prob[0][0])
            self.last_prob = p
            if p >= config.VAD_SPEECH_PROB:
                self._speech_ms += block_ms
                self._silence_ms = 0.0
                if self._speech_ms >= config.VAD_MIN_SPEECH_MS:
                    self._spoke = True
            else:
                self._silence_ms += block_ms
                if self._spoke and self._silence_ms >= config.VAD_SILENCE_MS:
                    ended = True
                    self.reset()
        return ended


_sess = None
_lock = threading.Lock()


def session():
    """One model per process; there can be several calls."""
    global _sess
    with _lock:
        if _sess is None:
            if not config.VAD_MODEL.exists():
                raise RuntimeError("Silero VAD model not found: %s (run scripts/download_models.py)"
                                   % config.VAD_MODEL)
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            _sess = ort.InferenceSession(str(config.VAD_MODEL), sess_options=so)
        return _sess


def make(rate: int = 8000) -> SpeechDetector:
    return SpeechDetector(session(), rate)
