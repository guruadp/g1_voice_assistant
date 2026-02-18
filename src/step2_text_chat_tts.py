import argparse
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from openai import OpenAI
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient, action_map


SYSTEM_PROMPT = (
    "You are Eddy, a friendly humanoid robot assistant. Speak naturally like a human in short spoken sentences. "
    "You can physically perform these basic G1 arm actions: shake hand, high five, high wave, face wave, clap, "
    "hug, hands up, heart, right hand up, x-ray, left kiss, right kiss, two-hand kiss, reject. "
    "When the user asks for one of these actions, explicitly confirm you can do it and that you are doing it now. "
    "Do not say you cannot physically do those supported actions."
)
DEFAULT_SINK = "alsa_output.usb-Anker_PowerConf_A3321-DEV-SN1-01.iec958-stereo"
DEFAULT_SOURCE = "alsa_input.usb-Anker_PowerConf_A3321-DEV-SN1-01.mono-fallback"
DEFAULT_NET_IF = "enP8p1s0"
DEFAULT_AUDIO_DEVICE = "plughw:0,0"
DEFAULT_STT_MODEL = "gpt-4o-mini-transcribe"


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


def map_action_by_text(text: str) -> str | None:
    t = text.lower()

    # Prioritized intent mapping aligned with g1_arm_action_nlu_llm_only.py action names.
    if any(k in t for k in ["handshake", "shake hand", "shake hands"]):
        return "shake hand"
    if any(k in t for k in ["high five", "hi five"]):
        return "high five"
    if any(k in t for k in ["face wave"]):
        return "face wave"
    if any(k in t for k in ["high wave", "wave high"]):
        return "high wave"
    if any(k in t for k in ["wave", "hello", "hi"]):
        return "high wave"
    if any(k in t for k in ["thank", "thanks"]):
        return "face wave"
    if any(k in t for k in ["hug"]):
        return "hug"
    if any(k in t for k in ["clap"]):
        return "clap"
    if any(k in t for k in ["hands up", "raise hands", "put your hands up"]):
        return "hands up"
    if any(k in t for k in ["heart", "love gesture"]):
        return "heart"
    if re.search(r"\b(reject|no|can\'t|cannot|refuse)\b", t):
        return "reject"
    if any(k in t for k in ["right hand up"]):
        return "right hand up"
    if any(k in t for k in ["x-ray", "xray"]):
        return "x-ray"
    if any(k in t for k in ["left kiss"]):
        return "left kiss"
    if any(k in t for k in ["right kiss"]):
        return "right kiss"
    if any(k in t for k in ["two-hand kiss", "two hand kiss"]):
        return "two-hand kiss"
    return None


def choose_action(user_text: str, reply_text: str) -> str:
    # User intent has priority over assistant wording.
    action = map_action_by_text(user_text)
    if action:
        return action

    action = map_action_by_text(reply_text)
    if action:
        return action

    return "high wave"




def action_phrase(action: str) -> str:
    phrase_map = {
        "shake hand": "a handshake",
        "high five": "a high five",
        "high wave": "a high wave",
        "face wave": "a face wave",
        "clap": "a clap",
        "hug": "a hug",
        "hands up": "hands up",
        "heart": "a heart gesture",
        "right hand up": "right hand up",
        "x-ray": "an x-ray pose",
        "left kiss": "a left kiss",
        "right kiss": "a right kiss",
        "two-hand kiss": "a two-hand kiss",
        "reject": "a reject gesture",
    }
    return phrase_map.get(action, action)

ACTION_SPEECH_STYLE = {
    "shake hand": "For handshake: invite the user to extend their hand and add a warm social line like 'nice to meet you'.",
    "hug": "For hug: invite the user to come a little closer, in a kind and respectful tone.",
    "high five": "For high five: use an upbeat line asking the user to raise their hand.",
    "high wave": "For high wave: use a cheerful greeting line.",
    "face wave": "For face wave: use a friendly polite line for thanks or hello.",
    "clap": "For clap: use a celebratory line.",
    "hands up": "For hands up: use a playful and safe instruction line.",
    "heart": "For heart: use a warm affectionate line.",
    "reject": "For reject: be polite, calm, and gentle in refusal wording.",
}

def action_style_hint(action: str | None) -> str:
    if not action:
        return ""
    return ACTION_SPEECH_STYLE.get(action, "Use a natural human-like line that matches the action.")


def ask_chat(client: OpenAI, model: str, system_prompt: str, user_text: str, requested_action: str | None = None) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    if requested_action:
        messages.append({
            "role": "system",
            "content": (
                f"The user requested this supported action: {requested_action}. "
                f"In your reply, explicitly confirm you can do it and that you are doing {action_phrase(requested_action)} now. "
                f"Style guidance: {action_style_hint(requested_action)} "
                "Reply in 1-2 short spoken sentences. Sound warm and human, but do not use emoji."
            ),
        })
    messages.append({"role": "user", "content": user_text})

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.4,
    )
    return (resp.choices[0].message.content or "").strip()


def _save_audio_response(audio_resp, out_path: Path) -> None:
    if hasattr(audio_resp, "write_to_file"):
        audio_resp.write_to_file(str(out_path))
        return
    if hasattr(audio_resp, "content") and audio_resp.content is not None:
        out_path.write_bytes(audio_resp.content)
        return
    raise RuntimeError("Unsupported OpenAI TTS response type.")


def tts_to_wav(client: OpenAI, model: str, voice: str, text: str, out_path: Path) -> Path:
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


def start_motion_thread(motion: ArmGestureController, gesture: str) -> threading.Thread:
    def _run_motion() -> None:
        try:
            motion.execute(gesture)
        except Exception as exc:
            print(f"[motion] execution error: {exc}")

    t = threading.Thread(target=_run_motion, daemon=True)
    t.start()
    return t


def record_wav_push_to_talk(device: str, sample_rate: int, channels: int) -> Path:
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
    print("Recording... press Enter to stop.")
    input()

    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
    proc.wait(timeout=3)

    return wav_path


def _extract_transcript(resp) -> str:
    text = getattr(resp, "text", None)
    if text:
        return text.strip()
    if isinstance(resp, dict):
        return str(resp.get("text", "")).strip()
    return ""


def transcribe_wav_openai(client: OpenAI, wav_path: Path, stt_model: str, language: str) -> str:
    lang = (language or "auto").strip().lower()
    kwargs = {"model": stt_model, "file": None}
    if lang != "auto":
        kwargs["language"] = lang

    with wav_path.open("rb") as f:
        try:
            kwargs["file"] = f
            resp = client.audio.transcriptions.create(**kwargs)
            text = _extract_transcript(resp)
            if text:
                return text
        except Exception as exc:
            if stt_model != "whisper-1":
                print(f"[stt] {stt_model} failed, retrying whisper-1: {exc}")
            else:
                raise

    if stt_model == "whisper-1":
        return ""

    with wav_path.open("rb") as f:
        fb_kwargs = {"model": "whisper-1", "file": f}
        if lang != "auto":
            fb_kwargs["language"] = lang
        resp = client.audio.transcriptions.create(**fb_kwargs)
        return _extract_transcript(resp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Push-to-talk STT -> ChatGPT -> OpenAI TTS -> Motion")
    parser.add_argument("--chat-model", default="gpt-4o-mini")
    parser.add_argument("--stt-model", default=DEFAULT_STT_MODEL)
    parser.add_argument("--stt-language", default="auto", help="STT language code (e.g. en, ar) or auto")
    parser.add_argument("--tts-model", default="gpt-4o-mini-tts")
    parser.add_argument("--voice", default="alloy")
    parser.add_argument("--sink", default=DEFAULT_SINK, help="Pulse sink name for paplay")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Pulse source name for microphone")
    parser.add_argument("--net-if", default=DEFAULT_NET_IF, help="Network interface for Unitree SDK")
    parser.add_argument("--audio-device", default=DEFAULT_AUDIO_DEVICE, help="ALSA capture device for arecord")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--no-play", action="store_true", help="Skip speaker playback")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    set_default_pulse_source(args.source)

    client = OpenAI()
    motion = ArmGestureController(net_if=args.net_if)

    print("Step 2: Push-to-talk STT + ChatGPT + TTS + Motion")
    print(f"Playback sink: {args.sink}")
    print(f"Mic source: {args.source}")
    print(f"Mic device: {args.audio_device}")
    print(f"STT model: {args.stt_model}")
    print(f"STT language: {args.stt_language}")
    print("[note] arecord uses ALSA device directly; Pulse source affects pulse clients, not arecord.")
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

        wav_in = None
        wav_tts = None
        try:
            wav_in = record_wav_push_to_talk(
                device=args.audio_device,
                sample_rate=args.sample_rate,
                channels=args.channels,
            )

            user_text = transcribe_wav_openai(client, wav_in, args.stt_model, args.stt_language)
            if not user_text:
                print("user> [empty]")
                continue

            print(f"user> {user_text}")
            requested_action = map_action_by_text(user_text)
            reply = ask_chat(client, args.chat_model, SYSTEM_PROMPT, user_text, requested_action=requested_action)
            if not reply:
                print("robot> [empty reply]")
                continue

            print(f"robot> {reply}")
            action = choose_action(user_text, reply)
            print(f"[motion] selected action: {action}")
            motion_thread = start_motion_thread(motion, action)

            with tempfile.NamedTemporaryFile(prefix="g1_tts_", suffix=".wav", delete=False) as tmp_tts:
                wav_tts = Path(tmp_tts.name)

            tts_to_wav(client, args.tts_model, args.voice, reply, wav_tts)
            if not args.no_play:
                play_wav(wav_tts, args.sink)

            motion_thread.join(timeout=0.1)

        except Exception as exc:
            print(f"Error: {exc}")
        finally:
            if wav_in and wav_in.exists():
                wav_in.unlink()
            if wav_tts and wav_tts.exists():
                wav_tts.unlink()


if __name__ == "__main__":
    main()
