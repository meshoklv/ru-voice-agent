#!/usr/bin/env python3
"""Test client: "calls" a running server with a WAV file instead of a microphone.

    python scripts/call_wav.py question.wav --token "$(cat data/token)"

The file must be mono 16-bit PCM WAV (any sample rate from 8 to 48 kHz). The script
sends the greeting request, then your audio, then prints what the agent recognised, what it
answered and the timings, and saves the answer audio to ./out/.
"""
import argparse
import asyncio
import base64
import json
import wave
from pathlib import Path

import aiohttp


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("--url", default="http://127.0.0.1:7693")
    ap.add_argument("--token", required=True)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    with wave.open(args.wav, "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit("need mono 16-bit PCM WAV")
        rate = w.getframerate()
        pcm = w.readframes(w.getnframes())

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    n = 0
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(args.url.replace("http", "ws") + "/ws?k=" + args.token) as ws:
            await ws.send_json({"type": "rate", "hz": rate if rate > 8000 else 0})
            await ws.send_json({"type": "call"})
            sent = False
            async for msg in ws:
                d = json.loads(msg.data)
                t = d["type"]
                if t == "audio":
                    n += 1
                    ext = "mp3" if "mpeg" in d["mime"] else "wav"
                    (out / ("reply-%02d.%s" % (n, ext))).write_bytes(base64.b64decode(d["b64"]))
                    print("agent  :", d["text"], "(tts %s ms)" % d.get("tts_ms"))
                elif t == "said" and d.get("text"):
                    print("heard  :", d["text"], "conf", d.get("conf"))
                elif t == "listen" and not sent:
                    sent = True
                    step = 4096 * 2
                    for i in range(0, len(pcm), step):
                        await ws.send_bytes(pcm[i:i + step])
                    await ws.send_json({"type": "end"})
                elif t == "done" and sent and d.get("timings", {}).get("total_ms") is not None:
                    print("timings:", d["timings"])
                    break
                elif t in ("error", "tts_failed"):
                    print("!!", d.get("message"))
                    if t == "error":
                        break


if __name__ == "__main__":
    asyncio.run(main())
