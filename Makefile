.PHONY: install models run test lint docker-models docker-up
PY ?= python3
VENV ?= .venv
TONE = https://github.com/voicekit-team/T-one/archive/3c5b6c015038173840e62cea99e10cdb1c759116.tar.gz

install:  ## local virtualenv with everything needed for the default stack
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install -U pip
	$(VENV)/bin/pip install -r requirements.txt
	$(VENV)/bin/pip install --no-deps $(TONE)
	$(VENV)/bin/pip install piper-tts

models:   ## download T-one, Silero VAD and a Piper voice into ./models
	$(VENV)/bin/python scripts/download_models.py

run:      ## start the server on 127.0.0.1:7693
	$(VENV)/bin/python -m voice_agent

test:
	$(VENV)/bin/pip install -q -r requirements-dev.txt
	$(VENV)/bin/python -m pytest -q

lint:
	$(VENV)/bin/ruff check .

docker-models:
	docker compose run --rm app python scripts/download_models.py

docker-up:
	docker compose up -d
