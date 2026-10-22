import numpy as np

from voice_agent.resample import Resampler


def _tone(freq, rate, seconds=1.0):
    t = np.arange(int(rate * seconds)) / rate
    return (np.sin(2 * np.pi * freq * t) * 12000).astype(np.int16)


def test_output_length_matches_ratio():
    out = Resampler(48000).feed(_tone(440, 48000).tobytes())
    assert abs(len(out) // 2 - 8000) < 80


def test_chunked_equals_whole_length():
    pcm = _tone(440, 44100, 2.0)
    r = Resampler(44100)
    total = sum(len(r.feed(pcm[i:i + 4096].tobytes())) for i in range(0, len(pcm), 4096))
    assert abs(total // 2 - 16000) < 160


def test_high_frequencies_are_filtered_not_aliased():
    # 7 kHz at 48 kHz would alias to ~1 kHz at 8 kHz without a low-pass filter.
    out = np.frombuffer(Resampler(48000).feed(_tone(7000, 48000).tobytes()), dtype=np.int16)
    assert np.abs(out[200:]).max() < 1500                 # was 12000 before filtering


def test_in_band_tone_survives():
    out = np.frombuffer(Resampler(48000).feed(_tone(1000, 48000).tobytes()), dtype=np.int16)
    assert np.abs(out[200:]).max() > 9000
