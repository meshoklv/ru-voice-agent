"""Speech recognition: T-one (streaming CTC, telephone 8 kHz), local, CPU only.

The model is downloaded from Hugging Face on first start (~145 MB) and kept in
memory afterwards (about six seconds to load).
"""
import threading

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from pyctcdecode import build_ctcdecoder
from tone.decoder import LABELS, GreedyCTCDecoder
from tone.logprob_splitter import StreamingLogprobSplitter
from tone.onnx_wrapper import StreamingCTCModel
from tone.pipeline import StreamingCTCPipeline

from . import config
from .resample import Resampler  # noqa: F401  (re-exported)


def _model_path() -> str:
    """models/tone/model.onnx if present (scripts/download_models.py), else Hugging Face cache."""
    local = config.MODELS_DIR / "tone" / "model.onnx"
    if local.exists():
        return str(local)
    return hf_hub_download("t-tech/T-one", "model.onnx")


class _BeamDecoder:
    """Beam search with hotwords.

    The stock T-one decoder calls decode() without a vocabulary, so we use the
    inner pyctcdecode decoder directly and pass our own words. The 5.5 GB KenLM
    language model is NOT used: it gave only ~2 points in our measurements.
    """

    def __init__(self, hotwords=None, weight=3.0):
        self._d = build_ctcdecoder(labels=list(LABELS))
        self._hot = [w for w in (hotwords or []) if w and all(
            ch in LABELS for ch in w.lower())]
        self._w = weight
        self.last_conf = None        # how sure the model is about the last phrase

    def forward(self, logprobs):
        # Confidence = mean per-frame probability of the most likely symbol. It is
        # not accuracy, but it shows when the model was guessing, so the agent can
        # ask to repeat instead of answering off-topic.
        try:
            self.last_conf = float(np.exp(logprobs.max(axis=1)).mean())
        except Exception:                                   # noqa: BLE001
            self.last_conf = None
        return self._d.decode(logprobs, beam_width=200,
                              hotwords=self._hot or None,
                              hotword_weight=self._w)


class Recognizer:
    """One instance per process. Calls are serialised with a lock."""

    def __init__(self):
        # silence divider: smaller = phrases are released sooner
        StreamingLogprobSplitter.MIN_SILENCE_DURATION = config.ASR_SILENCE
        so = ort.SessionOptions()
        # one thread: otherwise onnxruntime grabs every core for no gain
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(
            _model_path(), sess_options=so)
        self._decoder = self._make_decoder()
        self._pipe = StreamingCTCPipeline(
            StreamingCTCModel(self._sess), StreamingLogprobSplitter(), self._decoder)
        self._lock = threading.Lock()

    @staticmethod
    def _make_decoder():
        if not config.ASR_BEAM:
            return GreedyCTCDecoder()
        return _BeamDecoder(config.ASR_HOTWORDS, config.ASR_HOTWORD_WEIGHT)

    def transcribe(self, pcm16: bytes) -> str:
        """Takes mono 16-bit PCM at 8 kHz, returns recognised text."""
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.int32)
        if audio.size == 0:
            return ""
        pad, chunk = StreamingCTCPipeline.PADDING, StreamingCTCPipeline.CHUNK_SIZE
        a = np.pad(audio, (pad, pad))
        parts = []
        with self._lock:
            state = None
            for i in range(0, len(a), chunk):
                c = a[i:i + chunk]
                if len(c) < chunk:
                    c = np.pad(c, (0, chunk - len(c)))
                phrases, state = self._pipe.forward(c, state)
                parts += [p.text for p in phrases]
            tail = self._pipe.finalize(state)
            if isinstance(tail, tuple):
                tail = tail[0]
            parts += [p.text for p in tail]
        return " ".join(t.strip() for t in parts if t and t.strip()).strip()


class Stream:
    """Streaming recognition of one call.

    Audio chunks are fed as they arrive and phrases are returned while the person
    is still speaking. The onnxruntime session is shared; each stream has its own state.
    """

    PAD = StreamingCTCPipeline.PADDING
    CHUNK = StreamingCTCPipeline.CHUNK_SIZE      # 2400 samples = 300 ms at 8 kHz

    def __init__(self, recognizer: "Recognizer"):
        self._dec = Recognizer._make_decoder()
        self._pipe = StreamingCTCPipeline(
            StreamingCTCModel(recognizer._sess), StreamingLogprobSplitter(), self._dec)
        self._state = None
        self._buf = np.zeros(0, dtype=np.int32)
        self._started = False
        self.samples = 0
        self.confs = []              # confidence of every returned phrase

    def feed(self, pcm16: bytes):
        """Takes mono 16-bit PCM at 8 kHz, returns finished phrases (maybe none)."""
        a = np.frombuffer(pcm16, dtype=np.int16).astype(np.int32)
        self.samples += a.size
        if not self._started:
            # the model was trained with left padding; without it the first words are lost
            a = np.concatenate([np.zeros(self.PAD, dtype=np.int32), a])
            self._started = True
        self._buf = np.concatenate([self._buf, a])
        out = []
        while len(self._buf) >= self.CHUNK:
            chunk, self._buf = self._buf[:self.CHUNK], self._buf[self.CHUNK:]
            phrases, self._state = self._pipe.forward(chunk, self._state)
            if phrases:
                self._note_conf()
            out += [p.text for p in phrases]
        return out

    def _note_conf(self):
        c = getattr(self._dec, "last_conf", None)
        if c is not None:
            self.confs.append(c)

    def confidence(self):
        """Mean confidence over the phrases of this turn (or None)."""
        return round(sum(self.confs) / len(self.confs), 3) if self.confs else None

    def take_confidence(self):
        """Returns the turn confidence and resets the counter for the next turn."""
        c = self.confidence()
        self.confs = []
        return c

    def finish(self):
        """Flushes the tail and closes the stream, returning the remaining text."""
        tail = np.concatenate([self._buf, np.zeros(self.PAD, dtype=np.int32)])
        out = []
        for i in range(0, len(tail), self.CHUNK):
            chunk = tail[i:i + self.CHUNK]
            if len(chunk) < self.CHUNK:
                chunk = np.pad(chunk, (0, self.CHUNK - len(chunk)))
            phrases, self._state = self._pipe.forward(chunk, self._state)
            if phrases:
                self._note_conf()
            out += [p.text for p in phrases]
        final = self._pipe.finalize(self._state)
        if isinstance(final, tuple):
            final = final[0]
        out += [p.text for p in final]
        if final:
            self._note_conf()
        self._buf = np.zeros(0, dtype=np.int32)
        self._started = False
        self._state = None
        return out
