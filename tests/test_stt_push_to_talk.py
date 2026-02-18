#!/usr/bin/env python3
import argparse
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel

DEFAULT_SOURCE = "alsa_input.usb-Anker_PowerConf_A3321-DEV-SN1-01.mono-fallback"
DEFAULT_AUDIO_DEVICE = "plughw:0,0"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_WHISPER_MODEL = "base"
DEFAULT_WHISPER_COMPUTE = "int8"


def set_default_pulse_source(source: str) -> None:
    if not shutil.which("pactl"):
        print("[audio] pactl not found; skipping default source setup")
        return
    proc = subprocess.run(["pactl", "set-default-source", source], capture_output=True, text=True)
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip()
        print(f"[audio] failed to set source '{source}': {msg}")
    else:
        print(f"[audio] default source set: {source}")


def record_wav_push_to_talk(device: str, sample_rate: int, channels: int) -> Path:
    tmp = tempfile.NamedTemporaryFile(prefix="g1_stt_test_", suffix=".wav", delete=False)
    wav_path = Path(tmp.name)
    tmp.close()

    cmd = [
        "arecord",
        "-D",
        device,
        "-f",
        "S16_LE",
        "-r",
        str(sample_rate),
        "-c",
        str(channels),
        str(wav_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Recording... press Enter to stop.")
    input()

    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
    proc.wait(timeout=3)

    return wav_path


def transcribe_wav(stt_model: WhisperModel, wav_path: Path) -> str:
    segments, _ = stt_model.transcribe(str(wav_path), language="en")
    return "".join(segment.text for segment in segments).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="STT-only push-to-talk test")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Pulse source name")
    parser.add_argument("--audio-device", default=DEFAULT_AUDIO_DEVICE, help="ALSA capture device")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--whisper-compute-type", default=DEFAULT_WHISPER_COMPUTE)
    parser.add_argument("--keep-wav", action="store_true", help="Keep recorded WAV files for debugging")
    args = parser.parse_args()

    set_default_pulse_source(args.source)
    stt_model = WhisperModel(args.whisper_model, compute_type=args.whisper_compute_type)

    print("STT test ready")
    print(f"Source: {args.source}")
    print(f"Device: {args.audio_device}")
    print("Press Enter to start recording, Enter again to stop. Type q to quit.")

    while True:
        try:
            cmd = input("> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nExit.")
            break

        if cmd in {"q", "quit", "exit"}:
            print("Exit.")
            break

        wav_path = None
        try:
            wav_path = record_wav_push_to_talk(args.audio_device, args.sample_rate, args.channels)
            text = transcribe_wav(stt_model, wav_path)
            if text:
                print(f"stt> {text}")
            else:
                print("stt> [empty]")

            if args.keep_wav and wav_path.exists():
                print(f"[debug] kept {wav_path}")
                wav_path = None

        except Exception as exc:
            print(f"Error: {exc}")
        finally:
            if wav_path and wav_path.exists():
                wav_path.unlink()


if __name__ == "__main__":
    main()
