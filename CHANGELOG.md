# Changelog

## v0.1.0 — first public release

- Streaming STT (T-one) with Silero end-of-turn detection, barge-in, sentence-level pipelined TTS
- LLM: Ollama or any OpenAI-compatible endpoint, with a fallback queue
- TTS: Piper (local), Kokoro (local, experimental), Fish Audio (cloud)
- Token-protected, loopback-only server; browser client; per-call text logs
- Docker Compose quick start, model downloader, tests without models
