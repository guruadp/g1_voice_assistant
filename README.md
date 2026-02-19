# G1 Voice Assistant (Step-by-Step Build)

This project is built in small verified steps.

Only Step 0 (audio sink/source routing) is referenced from `run_gemini.sh`.

## Step 0: Audio Routing (Pulse)

List current default sink/source and available devices:

```bash
cd high_level/g1_voice_assistant
python3 misc/step0_audio_route.py
```

Set explicit defaults:

```bash
python3 misc/step0_audio_route.py --sink <sink_name> --source <source_name>
```

## Step 1: Push-to-Talk Mic Capture (WIP)

Current input flow:
- `PyAudio` default input device
- stream PCM chunks in memory
- save final recording as WAV

Run:

```bash
python3 misc/step1_arecord_test.py
```

## Step 2: Text Input -> ChatGPT -> OpenAI TTS

This skips STT for now and validates dialog + speech + gesture hooks.

Run:

```bash
python3 src/main.py
```

Requirements:
- `OPENAI_API_KEY` must be set.
- Speaker route must be valid for `aplay`.

Useful flags:

```bash
python3 src/main.py --no-play
python3 src/main.py --voice alloy
```
