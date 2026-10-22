# Latency notes

What the caller waits for is the silence between "I stopped talking" and "the agent
started talking". Everything here was measured on real runs of this pipeline; where a
number comes from a different setup than the default stack, it says so.

## Where the pause comes from

Live run, cloud TTS (Fish Audio) and a local Qwen model served by Ollama on an RTX 3090 Ti,
reached through SSH tunnels, three-sentence conversation:

| step | time |
|---|---|
| recognise the tail of the speech (T-one, CPU) | 0.32 s |
| first word from the LLM | 1.53 s |
| synthesise the first sentence (cloud TTS) | 1.39 s |
| **pause heard by the caller** | **3.69 s** (3.5 / 3.2 / 6.2 s in a later three-turn run) |

On top of that the browser waits 0.7 s of silence before it decides the turn is over
(Silero VAD on the server can close it sooner, in noisy rooms it closes it at all).

Journal of real calls (18-19 Sep 2026, same setup): first LLM word 1.19 to 2.39 s,
first audio in the browser 2.66 to 3.98 s; worst case in the same journal was a stalled
GPU box with 17.8 s to the first word.

## What helped, what did not (A/B on one build)

Hardware: 8 vCPU of a shared Intel Xeon E5-2695 v2 (2.4 GHz), 14 GB RAM, **no GPU**.
The LLM was replaced by the `mock` provider (first word after 1.2 s, then 45 pieces per
second) so that runs are comparable; TTS was a local CPU voice (Kokoro-class, about
3.3 s per sentence). Absolute seconds are therefore higher than on a GPU machine; only
compare "before" with "after".

| change | result |
|---|---|
| Overlapped synthesis of neighbouring sentences (`TTS_PARALLEL=2`, 28 replies per setting, interleaved runs) | gap between sentences, median: 4.34 s -> 3.89 s (-0.45 s). First audio and total time moved less than run-to-run noise: **do not read them as a speed-up.** |
| First sentence synthesised alone before the rest starts | without it the median first audio got worse (4.65 s -> 5.06 s): a second job steals cores from the first one |
| Short deadline for the first LLM word (10 s, then one retry with 25 s) | stalled server 30 s: success after 30 s -> after 10.3 s. Cold model load (15 s): success after 15.0 s -> 25.1 s (10 s worse). Trade-off chosen on purpose; `OLLAMA_FIRST_TOKEN=25` restores the old behaviour |
| Cache of short phrases (`TTS_CACHE_MAX`) | greeting, "Okay.", "Hello, can you hear me?": 2.7-9.6 s first time, 0.000 s repeated |
| Smaller first-chunk thresholds (12/42/28 instead of 18/55/40), 205 real replies | gain: zero (median 0 characters, mean 1.4 characters, about 10 ms). First chunks cut mid-phrase went from 1 to 7 of 205. **Rejected.** |

## The recognition side

- T-one runs faster than real time on **one CPU core** (greedy decoding, RTF 0.31 on the
  benchmark machine). See `ru-speech-bench` for GigaAM, T-one and cloud STT side by side.
- T-one has no Latin letters and no digits in its output alphabet. "Excel" arrives as
  Cyrillic letters. The LLM reads it fine; do not use the transcript for exact IDs.
- Hotwords in the beam decoder help with rare words; short common words must not be added
  (they get inserted at random positions).
- Sample-rate conversion in the browser without a low-pass filter makes the recognizer lose
  and invent words. The browser sends the raw rate and the server resamples properly.

## Reproduce

```bash
VOICE_LLM=mock TTS_CACHE_MAX=0 TTS_PARALLEL=1 python -m voice_agent   # then 2, alternate runs
python scripts/call_wav.py question.wav --token "$(cat data/token)"   # prints timings
```
