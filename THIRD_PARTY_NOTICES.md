# Third-party notices

This repository contains **no third-party model weights**. `scripts/download_models.py` downloads them from the official sources
at install time, and the Docker image installs the Python packages below. Licenses were read from the upstream pages in
October 2026. Not legal advice: check them again before a commercial deployment.

## Models

| Component | Role | License | Commercial use | Source |
|---|---|---|---|---|
| T-one (T-Tech) | speech recognition | Apache-2.0 (code and weights) | yes | https://huggingface.co/t-tech/T-one |
| Silero VAD v5.1.2 | end-of-turn detection | MIT | yes | https://github.com/snakers4/silero-vad |
| Piper voice `ru_RU-denis-medium` (default) | speech synthesis | dataset CC0, **but fine-tuned from the English `lessac` voice whose data is research-only**; the license of the checkpoint is not stated | **unverified** | https://huggingface.co/rhasspy/piper-voices |
| Piper voices `dmitri`, `irina`, `ruslan` (optional) | speech synthesis | dmitri: as denis; irina: "Unknown" on the model card; **ruslan: CC BY-NC-SA 4.0** | dmitri/irina: unverified, **ruslan: NO** | same |
| Kokoro-82M (optional, `--kokoro`) | speech synthesis | Apache-2.0 | yes | https://huggingface.co/hexgrad/Kokoro-82M |
| Fish Audio API (optional) | cloud speech synthesis | vendor terms (not reviewed); the open Fish weights are **CC-BY-NC-SA-4.0** (non-commercial), but this project only calls the API | see vendor terms | https://fish.audio |
| Qwen3-8B (default Ollama model `qwen3:8b`) | LLM | Apache-2.0 | yes | https://huggingface.co/Qwen/Qwen3-8B |

Details, measurements and the license reading dates for every speech engine are in the companion benchmark
[ru-speech-bench](https://github.com/meshoklv/ru-speech-bench).

> [!WARNING]
> **If you need a commercially safe stack, do not rely on the default Piper voice without your own legal check**: all four Russian Piper
> voices descend from a research-only English voice. Use Kokoro (Apache-2.0) or a cloud voice you hold a license for.

## Python packages (installed, not vendored)

| Package | License |
|---|---|
| aiohttp | Apache-2.0 AND MIT |
| numpy | BSD-3-Clause (and others, see the package) |
| onnxruntime | MIT |
| huggingface-hub | Apache-2.0 |
| pyctcdecode | Apache-2.0 |
| tone (T-one runtime) | Apache-2.0 |
| num2words | LGPL |
| piper-tts (OHF-Voice/piper1-gpl) | **GPL-3.0-or-later**. It is installed into the Docker image, so an image containing it is a combined work to be distributed under GPL terms. This repository's own code stays MIT and does not vendor it |
| kokoro-onnx (optional) | MIT |
| Ollama (docker image) | MIT |
