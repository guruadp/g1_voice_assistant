import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path


def ensure_local_pulse_env() -> None:
    os.environ.pop("PULSE_SERVER", None)
    os.environ["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"
    os.environ["PULSE_RUNTIME_PATH"] = f"{os.environ['XDG_RUNTIME_DIR']}/pulse"
    print(f"[audio] XDG_RUNTIME_DIR={os.environ['XDG_RUNTIME_DIR']}")
    print(f"[audio] PULSE_RUNTIME_PATH={os.environ['PULSE_RUNTIME_PATH']}")


def set_default_pulse_sink(sink: str) -> None:
    if not shutil.which("pactl"):
        print("[audio] pactl not found; skipping default sink setup")
        return
    proc = subprocess.run(["pactl", "set-default-sink", sink], capture_output=True, text=True)
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip()
        print(f"[audio] failed to set sink '{sink}': {msg}")
        return
    print(f"[audio] default sink set: {sink}")


def set_default_pulse_source(source: str) -> None:
    if not shutil.which("pactl"):
        print("[audio] pactl not found; skipping default source setup")
        return
    proc = subprocess.run(["pactl", "set-default-source", source], capture_output=True, text=True)
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip()
        print(f"[audio] failed to set source '{source}': {msg}")
        return
    print(f"[audio] default source set: {source}")


def play_wav(path: Path, sink: str | None = None) -> None:
    if shutil.which("paplay"):
        cmd = ["paplay"]
        if sink:
            cmd.extend(["--device", sink])
        cmd.append(str(path))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return
        print(f"[playback] paplay failed ({proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}")

    if shutil.which("aplay"):
        cmd = ["aplay", str(path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return
        raise RuntimeError(f"aplay failed ({proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}")

    raise RuntimeError("No playback tool found. Install pulseaudio-utils (paplay) or alsa-utils (aplay).")


def record_wav(device: str, sample_rate: int, channels: int, prompt: str = "Recording... press Enter to stop.") -> Path:
    tmp = tempfile.NamedTemporaryFile(prefix="g1_stt_", suffix=".wav", delete=False)
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
    print(prompt)
    input()

    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
    proc.wait(timeout=3)

    return wav_path
