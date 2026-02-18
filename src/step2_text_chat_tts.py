import argparse
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient, action_map


SYSTEM_PROMPT = "You are a helpful humanoid robot assistant and your name id Eddy. Keep replies concise and natural for speech."
DEFAULT_SINK = "alsa_output.usb-Anker_PowerConf_A3321-DEV-SN1-01.iec958-stereo"
DEFAULT_NET_IF = "enP8p1s0"


class ArmGestureController:
    def __init__(self, net_if: str):
        self.enabled = False
        self._client = None
        try:
            ChannelFactoryInitialize(0, net_if)
            self._client = G1ArmActionClient()
            self._client.SetTimeout(10.0)
            self._client.Init()
            self.enabled = True
            print(f"[motion] SDK ready on net_if={net_if}")
        except Exception as exc:
            print(f"[motion] SDK init failed: {exc}")

    def execute(self, style: str) -> None:
        if not self.enabled:
            print(f"[motion] skipped (SDK not ready): {style}")
            return

        action_id = action_map.get(style)
        release_id = action_map.get("release arm")
        if action_id is None:
            print(f"[motion] unknown action style: {style}")
            return

        self._client.ExecuteAction(action_id)
        time.sleep(0.45)
        if release_id is not None:
            self._client.ExecuteAction(release_id)
            time.sleep(0.35)


def map_gesture(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["hi", "hello"]):
        return "high wave"
    if any(k in t for k in ["thank", "thanks"]):
        return "face wave"
    if any(k in t for k in ["great", "awesome", "nice", "happy"]):
        return "high five"
    if any(k in t for k in ["sorry", "cannot", "can't", "no"]):
        return "reject"
    return "high wave"


def ask_chat(client: OpenAI, model: str, system_prompt: str, user_text: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=0.4,
    )
    return (resp.choices[0].message.content or "").strip()


def sanitize_filename(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
    return text[:40] or "reply"


def _save_audio_response(audio_resp, out_path: Path) -> None:
    if hasattr(audio_resp, "stream_to_file"):
        audio_resp.stream_to_file(str(out_path))
        return
    if hasattr(audio_resp, "write_to_file"):
        audio_resp.write_to_file(str(out_path))
        return
    if hasattr(audio_resp, "content") and audio_resp.content is not None:
        out_path.write_bytes(audio_resp.content)
        return
    raise RuntimeError("Unsupported OpenAI TTS response type.")


def tts_to_wav(client: OpenAI, model: str, voice: str, text: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kwargs = {"model": model, "voice": voice, "input": text}
    try:
        audio_resp = client.audio.speech.create(**kwargs, response_format="wav")
    except TypeError:
        try:
            audio_resp = client.audio.speech.create(**kwargs, format="wav")
        except TypeError:
            audio_resp = client.audio.speech.create(**kwargs)

    _save_audio_response(audio_resp, out_path)
    return out_path


def play_wav(path: Path, sink: str | None) -> None:
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


def start_motion_thread(motion: ArmGestureController, gesture: str) -> threading.Thread:
    def _run_motion() -> None:
        try:
            motion.execute(gesture)
        except Exception as exc:
            print(f"[motion] execution error: {exc}")

    t = threading.Thread(target=_run_motion, daemon=True)
    t.start()
    return t


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: text input -> ChatGPT -> OpenAI TTS -> speaker")
    parser.add_argument("--chat-model", default="gpt-4o-mini")
    parser.add_argument("--tts-model", default="gpt-4o-mini-tts")
    parser.add_argument("--voice", default="alloy")
    parser.add_argument("--sink", default=DEFAULT_SINK, help="Pulse sink name for paplay")
    parser.add_argument("--net-if", default=DEFAULT_NET_IF, help="Network interface for Unitree SDK")
    parser.add_argument("--no-play", action="store_true", help="Generate TTS file but do not play it")
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI()
    motion = ArmGestureController(net_if=args.net_if)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("Step 2: Text Chat + TTS + Motion")
    if args.sink:
        print(f"Playback sink: {args.sink}")
    print("Commands: type your message, or q to quit.")

    while True:
        try:
            user_text = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExit.")
            break

        if user_text.lower() in {"q", "quit", "exit"}:
            print("Exit.")
            break
        if not user_text:
            continue

        try:
            reply = ask_chat(client, args.chat_model, SYSTEM_PROMPT, user_text)
            if not reply:
                print("robot> [empty reply]")
                continue

            print(f"robot> {reply}")
            gesture = map_gesture(reply)
            motion_thread = start_motion_thread(motion, gesture)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = sanitize_filename(reply)
            wav_path = artifacts_dir / f"tts_{ts}_{base}.wav"
            tts_to_wav(client, args.tts_model, args.voice, reply, wav_path)

            if not args.no_play:
                try:
                    play_wav(wav_path, args.sink)
                finally:
                    if wav_path.exists():
                        wav_path.unlink()
                        print(f"[tts] deleted {wav_path}")
            else:
                print(f"[tts] saved {wav_path}")

            motion_thread.join(timeout=0.1)

        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
