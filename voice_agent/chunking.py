"""Cuts the streaming LLM output into pieces that can be spoken right away.

Waiting for the whole answer is pointless: as soon as a sentence is complete it can go
to synthesis while the model is still writing the rest. The FIRST piece uses a lower
bar (a clause up to a comma is enough) because it defines how long the caller listens
to silence.

We measured the first-chunk thresholds on 205 real replies: making them smaller gave
no gain (the sentence/comma cut usually fires first) but increased the number of first
chunks cut mid-phrase from 1 to 7 of 205. Hence the defaults 18 / 55 / 40.
"""
from . import config

SENT_END = ".!?…"


def pop_speakable(buf: str, force=False, first=False):
    """Returns (ready_text, rest). ``force`` flushes everything (end of answer)."""
    if force:
        return (buf.strip(), "") if buf.strip() else ("", buf)
    for i, ch in enumerate(buf):
        if ch in SENT_END and (i + 1 >= len(buf) or buf[i + 1] in " \n"):
            return buf[:i + 1].strip(), buf[i + 1:]
    if first:
        for i, ch in enumerate(buf):
            if ch in ",;:—–" and i >= config.FIRST_CHUNK_MIN:
                return buf[:i + 1].strip(), buf[i + 1:]
        if (len(buf) > config.FIRST_CHUNK_MAX
                and " " in buf[config.FIRST_CHUNK_CUT:]):
            cut = config.FIRST_CHUNK_CUT + buf[config.FIRST_CHUNK_CUT:].index(" ")
            return buf[:cut].strip(), buf[cut:]
    elif len(buf) > 80 and "," in buf:
        cut = buf.rfind(",")
        return buf[:cut + 1].strip(), buf[cut + 1:]
    return "", buf
