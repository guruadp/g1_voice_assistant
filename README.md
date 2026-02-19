# G1 Voice Assistant

Voice pipeline for G1:
- push-to-talk recording (`arecord`)
- STT (OpenAI)
- chat response (OpenAI)
- TTS playback (`paplay`/`aplay`)
- arm gestures + locomotion commands

## Architecture

Core modules in `src/`:

- `src/main.py`
  - Application entrypoint and orchestration loop.
  - Wires audio, NLU, LLM, motion, and locomotion modules.
- `src/config.py`
  - Centralized constants and defaults (models, prompt, device defaults).
- `src/audio_io.py`
  - Pulse environment setup, sink/source defaults, WAV record/play helpers.
- `src/nlu.py`
  - Text normalization and parsing for action/locomotion intents.
- `src/llm.py`
  - Chat request wrapper and compact spoken reply shaping.
- `src/motion.py`
  - `ArmGestureController` and custom arm motion process control.
- `src/loco.py`
  - Locomotion command execution (`apply_loco_commands`).

Support/diagnostic scripts:

- `misc/step0_audio_route.py`: inspect/set Pulse sink/source.
- `misc/step1_arecord_test.py`: standalone push-to-talk mic capture test.

## Run

```bash
python3 src/main.py
```

Requirements:
- `OPENAI_API_KEY` is set.
- Audio tools available (`arecord`, and either `paplay` or `aplay`).

Useful flags:

```bash
python3 src/main.py --no-play
python3 src/main.py --voice alloy
python3 src/main.py --chat-model gpt-4o-mini --tts-model gpt-4o-mini-tts
```

## Audio Setup Helpers

Inspect current Pulse defaults/devices:

```bash
python3 misc/step0_audio_route.py
```

Set explicit Pulse defaults:

```bash
python3 misc/step0_audio_route.py --sink <sink_name> --source <source_name>
```
