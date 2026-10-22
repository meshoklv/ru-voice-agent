"""Streaming sample-rate conversion to 8 kHz (low-pass filter, then decimation)."""
import numpy as np


class Resampler:
    """Proper sample-rate conversion: low-pass filter first, then decimation.

    Taking every N-th sample (or linear interpolation) is not acceptable: everything
    above 4 kHz folds back into the audible band and wrecks the recording. With
    recognition this shows up as lost words and invented ones.

    Works as a stream: the tail of the previous chunk is kept to avoid clicks at the seams.
    """

    TARGET = 8000
    CUTOFF = 3800.0        # just below half of the target rate
    TAPS = 65              # odd: linear-phase filter without delay shift

    def __init__(self, src_rate: int):
        self.src = int(src_rate)
        self.ratio = self.src / float(self.TARGET)
        n = np.arange(self.TAPS) - (self.TAPS - 1) / 2.0
        fc = self.CUTOFF / self.src
        h = np.sinc(2 * fc * n) * np.hamming(self.TAPS)
        self._h = (h / h.sum()).astype(np.float32)
        self._tail = np.zeros(self.TAPS - 1, dtype=np.float32)
        self._pos = 0.0        # fractional read position carried between chunks

    def feed(self, pcm16: bytes) -> bytes:
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
        if x.size == 0:
            return b""
        buf = np.concatenate([self._tail, x])
        if buf.size < self.TAPS:
            self._tail = buf
            return b""
        y = np.convolve(buf, self._h, mode="valid")     # same length as x
        self._tail = buf[-(self.TAPS - 1):]
        idx = np.arange(self._pos, y.size, self.ratio)
        if idx.size == 0:
            self._pos -= y.size
            return b""
        lo = np.floor(idx).astype(np.int64)
        fr = (idx - lo).astype(np.float32)
        hi = np.minimum(lo + 1, y.size - 1)
        out = y[lo] * (1 - fr) + y[hi] * fr
        self._pos = idx[-1] + self.ratio - y.size
        return np.clip(out, -32768, 32767).astype(np.int16).tobytes()
