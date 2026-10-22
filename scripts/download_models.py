#!/usr/bin/env python3
"""Downloads the model files into ./models (or $VOICE_MODELS_DIR) from their official sources.

    python scripts/download_models.py              # T-one + Silero VAD + Piper voice (default stack)
    python scripts/download_models.py --kokoro     # also Kokoro (experimental Russian)
    python scripts/download_models.py --piper-voice ru_RU-denis-medium

Only the standard library is used. Files are verified by size and skipped if present.
Licenses of every model are listed in THIRD_PARTY_NOTICES.md: read it before you use the
agent commercially (the Piper Russian voices and Kokoro have their own terms).
"""
import argparse
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = Path(os.environ.get("VOICE_MODELS_DIR", ROOT / "models"))

HF = "https://huggingface.co"
FILES = {
    # name: (url, expected size in bytes or None, relative target)
    "tone": [(HF + "/t-tech/T-one/resolve/main/model.onnx", 144193371, "tone/model.onnx")],
    "vad": [("https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/src/silero_vad/data/silero_vad.onnx",
             2327524, "silero_vad.onnx")],
    "kokoro": [
        ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
         325532387, "kokoro/kokoro-v1.0.onnx"),
        ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
         28214398, "kokoro/voices-v1.0.bin"),
    ],
}


def piper_files(voice: str):
    # ru_RU-denis-medium -> ru/ru_RU/denis/medium/ru_RU-denis-medium.onnx
    lang, name, quality = voice.split("-")
    base = "%s/rhasspy/piper-voices/resolve/main/%s/%s/%s/%s/%s" % (
        HF, lang.split("_")[0], lang, name, quality, voice)
    return [(base + ".onnx", None, "piper/%s.onnx" % voice),
            (base + ".onnx.json", None, "piper/%s.onnx.json" % voice)]


def fetch(url, size, rel):
    dst = MODELS / rel
    if dst.exists() and (size is None or dst.stat().st_size == size):
        print("ok      %s (already there)" % rel)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print("fetch   %s" % url)
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            got += len(b)
            if total:
                print("\r        %3d%%" % (100 * got // total), end="", flush=True)
    print()
    if size is not None and tmp.stat().st_size != size:
        tmp.unlink()
        sys.exit("size mismatch for %s: the upstream file changed, update scripts/download_models.py" % rel)
    tmp.rename(dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kokoro", action="store_true", help="also download Kokoro (experimental for Russian)")
    ap.add_argument("--piper-voice", default=os.environ.get("PIPER_VOICE", "ru_RU-denis-medium"))
    ap.add_argument("--skip-piper", action="store_true", help="when you use Kokoro or a cloud TTS only")
    args = ap.parse_args()

    jobs = FILES["tone"] + FILES["vad"]
    if not args.skip_piper:
        jobs += piper_files(args.piper_voice)
    if args.kokoro:
        jobs += FILES["kokoro"]
    for url, size, rel in jobs:
        fetch(url, size, rel)
    print("\nDone. Models are in %s. See THIRD_PARTY_NOTICES.md for their licenses." % MODELS)


if __name__ == "__main__":
    main()
