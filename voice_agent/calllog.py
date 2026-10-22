"""Per-call log, so a bad conversation can be analysed afterwards.

One JSONL file per call, one line per event: what was recognised, what the model
answered, what went to synthesis, timings of every step, failures.

The access token is never logged. Audio is never stored, only text. Delete
``data/calls`` whenever you like; set VOICE_CALL_LOG=0 to disable it.
"""
import json
import os
import threading
import time
from datetime import datetime

from . import config


class CallLog:
    def __init__(self, number: int, meta: dict):
        self.enabled = os.environ.get("VOICE_CALL_LOG", "1") not in ("0", "", "off")
        self.number = number
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()
        self._turns, self._pauses, self._fails = 0, [], 0
        self.path = None
        if self.enabled:
            config.LOG_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.path = config.LOG_DIR / ("call-%s-%03d.jsonl" % (stamp, number))
        self.write("call_start", **meta)

    def write(self, kind, **fields):
        if not self.path:
            return
        rec = {"t": datetime.now().strftime("%H:%M:%S"),
               "s": round(time.perf_counter() - self._t0, 2), "kind": kind}
        rec.update(fields)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def turn(self, said, reply, spoken, timings, conf=None, unsure=False):
        self._turns += 1
        pause = (timings or {}).get("first_audio_ms")
        if pause:
            self._pauses.append(pause)
        self.write("turn", n=self._turns, said=said, reply=reply, spoken=spoken,
                   conf=conf, unsure=unsure, timings=timings)

    def fail(self, where, message):
        self._fails += 1
        self.write("fail", where=where, message=str(message)[:300])

    def finish(self):
        p = sorted(self._pauses)
        self.write("call_end", turns=self._turns, fails=self._fails,
                   pause_avg_ms=round(sum(p) / len(p)) if p else None,
                   pause_worst_ms=p[-1] if p else None,
                   duration_s=round(time.perf_counter() - self._t0, 1))
