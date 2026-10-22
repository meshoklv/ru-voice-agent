FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/models/hf
WORKDIR /app

COPY requirements.txt .
# T-one is installed without its kenlm dependency (not needed for CTC beam search with
# hotwords, and kenlm has no wheels). Pin: commit of voicekit-team/T-one.
RUN pip install -r requirements.txt \
 && pip install --no-deps "https://github.com/voicekit-team/T-one/archive/3c5b6c015038173840e62cea99e10cdb1c759116.tar.gz" \
 && pip install piper-tts
ARG WITH_KOKORO=0
RUN if [ "$WITH_KOKORO" = "1" ]; then pip install kokoro-onnx soundfile; fi

COPY voice_agent ./voice_agent
COPY scenarios ./scenarios
COPY scripts ./scripts
COPY voices.example.json ./

ENV VOICE_HOST=0.0.0.0 VOICE_PORT=7693
EXPOSE 7693
CMD ["python", "-m", "voice_agent"]
