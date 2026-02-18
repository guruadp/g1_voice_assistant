import argparse
import queue
import threading
import wave
from datetime import datetime
from pathlib import Path

import pyaudio

FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024


def _stdin_waiter(stop_event: threading.Event) -> None:
    try:
        input()
    except EOFError:
        pass
    stop_event.set()


def record_push_to_talk(
    pa: pyaudio.PyAudio,
    sample_rate: int,
    channels: int,
    device_index: int,
) -> bytes:
    audio_q: queue.Queue[bytes] = queue.Queue()
    stop_event = threading.Event()

    stream = pa.open(
        format=FORMAT,
        channels=channels,
        rate=sample_rate,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK_SIZE,
    )

    print("Recording... press Enter to stop")
    threading.Thread(target=_stdin_waiter, args=(stop_event,), daemon=True).start()

    try:
        while not stop_event.is_set():
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_q.put(data)
    finally:
        stream.stop_stream()
        stream.close()

    chunks = []
    while not audio_q.empty():
        chunks.append(audio_q.get_nowait())
    return b"".join(chunks)


def save_wav(path: Path, pcm_bytes: bytes, sample_rate: int, channels: int, sample_width: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1 mic test (PyAudio capture)")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--device-index", type=int, default=None, help="If omitted, use default input device")
    args = parser.parse_args()

    pa = pyaudio.PyAudio()

    try:
        if args.device_index is None:
            mic_info = pa.get_default_input_device_info()
            device_index = mic_info["index"]
            print(f"Using default input device: index={device_index}, name={mic_info['name']}")
        else:
            device_index = args.device_index
            dev = pa.get_device_info_by_index(device_index)
            print(f"Using selected input device: index={device_index}, name={dev['name']}")

        print("Step 1: Push-to-talk mic capture")
        print("Commands: Enter=start recording, q=quit")

        artifacts_dir = Path("artifacts")
        sample_width = pa.get_sample_size(FORMAT)

        while True:
            try:
                cmd = input("> ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\nExit.")
                break

            if cmd in {"q", "quit", "exit"}:
                print("Exit.")
                break
            if cmd not in {"", "r", "rec", "record"}:
                print("Unknown command. Use Enter to record or q to quit.")
                continue

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_file = artifacts_dir / f"mic_test_{ts}.wav"
            pcm = record_push_to_talk(
                pa=pa,
                sample_rate=args.sample_rate,
                channels=args.channels,
                device_index=device_index,
            )

            if not pcm:
                print("No audio captured.")
                continue

            save_wav(
                path=out_file,
                pcm_bytes=pcm,
                sample_rate=args.sample_rate,
                channels=args.channels,
                sample_width=sample_width,
            )
            duration = len(pcm) / (args.sample_rate * args.channels * sample_width)
            print(f"Saved {out_file} ({duration:.2f}s)")

    finally:
        pa.terminate()


if __name__ == "__main__":
    main()
