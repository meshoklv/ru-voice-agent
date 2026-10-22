# Contributing

```bash
make install && make test      # or: pip install -r requirements-dev.txt && pytest
ruff check voice_agent tests scripts
```

- Keep the server loopback-only and token-protected; a PR that opens an unauthenticated endpoint will not be merged.
- New LLM or TTS engines: add a class with `stream()` / `synth()` in `llm.py` / `tts.py`, register it, document the env variables in `.env.example`.
- Tests must run without model files (use fakes, see `tests/test_ws_flow.py`).
- New scenarios go to `scenarios/` and must contain only made-up data.
