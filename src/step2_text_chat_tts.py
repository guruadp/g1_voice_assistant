import argparse
import os
import random
import re
import signal
import subprocess
import tempfile
import sys
import threading
import time
import wave
from pathlib import Path

from openai import OpenAI
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient, action_map
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from audio_io import (
    ensure_local_pulse_env,
    play_wav,
    record_wav,
    set_default_pulse_sink,
    set_default_pulse_source,
)


SYSTEM_PROMPT = (
    "You are Eddy, a friendly humanoid robot assistant. Speak naturally like a human in short spoken sentences. "
    "You can physically perform these basic G1 arm actions: shake hand, high five, high wave, face wave, clap, "
    "hug, hands up, heart, right hand up, x-ray, left kiss, right kiss, two-hand kiss, reject. "
    "You can also do basic locomotion: walk forward/backward, move left/right, turn left/right, and stop. "
    "When the user explicitly asks for one of these actions, respond naturally and warmly. "
    "When the user asks for supported locomotion, acknowledge and speak as if you are doing it; do not claim you cannot. "
    "For normal greetings like hello/hi, respond socially like a human (for example: hi, nice to see you, how can I help you). "
    "Do not force an action confirmation for simple greetings unless the user asked for an action. "
    "Do not say you cannot physically do those supported actions. "
    "For time-sensitive or real-world facts (weather, live events, recent news), use web search when available. "
    "Keep responses concise by default: 1-2 short sentences, focusing only on key info. "
    "Do not provide long lists, hourly breakdowns, or long background unless the user explicitly asks for details."
)
DEFAULT_SINK = "alsa_output.usb-Anker_PowerConf_A3321-DEV-SN1-01.iec958-stereo"
DEFAULT_SOURCE = "alsa_input.usb-Anker_PowerConf_A3321-DEV-SN1-01.mono-fallback"
DEFAULT_NET_IF = "enP8p1s0"
DEFAULT_AUDIO_DEVICE = "plughw:0,0"
DEFAULT_STT_MODEL = "gpt-4o-mini-transcribe"
CUSTOM_MOTION_RELEASE_SEC = 1.5

EXPLAIN_CUSTOM_ACTIONS = [
    "custom:explain_basic",
    "custom:explain_bent_elbows",
    "custom:explain_open_palms",
]

NEGATION_WORDS = {"not", "dont", "don't", "do not", "never", "no"}
SPLIT_PATTERN = re.compile(r"\b(?:and then|then|after that|afterwards|and)\b|[,;]+")

CUSTOM_MOTION_SCRIPTS = {
    "custom:explain_basic": "arm_explain_motion_basic_sdk.py",
    "custom:explain_bent_elbows": "arm_explain_motion_bent_elbows_sdk.py",
    "custom:explain_open_palms": "arm_explain_motion_open_palms_sdk.py",
}


class ArmGestureController:
    def __init__(self, net_if: str, dry_run: bool = False):
        self.enabled = False
        self.net_if = net_if
        self.dry_run = dry_run
        self._client = None
        if self.dry_run:
            self.enabled = True
            print("[DRY-RUN] Motion controller active (no arm commands will be sent).")
            return
        try:
            ChannelFactoryInitialize(0, net_if)
            self._client = G1ArmActionClient()
            self._client.SetTimeout(10.0)
            self._client.Init()
            self.enabled = True
            print(f"[motion] SDK ready on net_if={net_if}")
        except Exception as exc:
            print(f"[motion] SDK init failed: {exc}")

    def execute(self, style: str, auto_release: bool = True) -> None:
        if self.dry_run:
            print(f"[DRY-RUN] Would execute arm action: {style}")
            if auto_release:
                print("[DRY-RUN] Would release arm")
            return
        if not self.enabled:
            print(f"[motion] skipped (SDK not ready): {style}")
            return

        action_id = action_map.get(style)
        if action_id is None:
            print(f"[motion] unknown action style: {style}")
            return

        self._client.ExecuteAction(action_id)
        time.sleep(0.45)
        if auto_release:
            self.release_arm()

    def release_arm(self) -> None:
        if self.dry_run:
            print("[DRY-RUN] Would release arm")
            return
        if not self.enabled:
            return
        release_id = action_map.get("release arm")
        if release_id is None:
            return
        self._client.ExecuteAction(release_id)
        time.sleep(0.35)


    def launch_custom_process(self, style: str, duration_s: float = 20.0, release_duration_s: float = 0.2):
        if self.dry_run:
            print(f"[DRY-RUN] Would launch custom motion {style} for {duration_s:.2f}s")
            return None
        script_name = CUSTOM_MOTION_SCRIPTS.get(style)
        if not script_name:
            print(f"[motion] unknown custom motion: {style}")
            return None

        script_path = Path(__file__).resolve().parent / script_name
        if not script_path.exists():
            print(f"[motion] custom motion script not found: {script_path}")
            return None

        cmd = [
            sys.executable,
            str(script_path),
            "--net-if",
            self.net_if,
            "--duration",
            str(max(0.2, float(duration_s))),
            "--amp",
            "0.10",
            "--release-duration",
            str(max(0.05, float(release_duration_s))),
            "--no-prompt",
        ]
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop_custom_process(self, proc, timeout_s: float = 2.5) -> bool:
        if proc is None:
            return True
        if proc.poll() is not None:
            return True

        # Ask motion script to finish via its smooth release path.
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=timeout_s)
            return True
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=0.6)
                return False
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1.0)
                return False



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
    if re.search(r"\bkiss\b", t):
        return "two-hand kiss"
    if any(k in t for k in ["two-hand kiss", "two hand kiss"]):
        return "two-hand kiss"
    return None


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("_", " ")
    text = text.replace("donot", "do not")
    text = re.sub(r"[^a-z0-9\-\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_requests(text: str):
    parts = [p.strip() for p in SPLIT_PATTERN.split(text) if p.strip()]
    return parts if parts else [text.strip()]


def has_negation(text: str) -> bool:
    t = normalize(text)
    if not t:
        return False
    return any(n in t for n in NEGATION_WORDS)


def contains_action_hint(text: str) -> bool:
    return map_action_by_text(text) is not None


def contains_loco_hint(text: str) -> bool:
    t = normalize(text)
    if not t:
        return False

    if any(k in t for k in ["stop", "halt", "freeze", "stop moving", "stop walk"]):
        return True

    has_turn_word = any(k in t for k in ["turn", "rotate", "spin", "yaw"])
    has_move_word = any(k in t for k in ["walk", "move", "go", "step"])
    has_direction_word = any(k in t for k in ["forward", "backward", "back", "lateral", "sideways"])

    if has_turn_word or has_move_word or has_direction_word:
        return True

    # Bare "left/right" should not be treated as locomotion unless movement/turn intent exists.
    if ("left" in t or "right" in t) and (has_turn_word or has_move_word or "lateral" in t or "sideways" in t):
        return True

    return False


def parse_duration_seconds(text: str, default_duration: float, seconds_per_step: float) -> float:
    t = normalize(text)

    sec_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|sec|s)\b", t)
    if sec_match:
        return max(0.1, float(sec_match.group(1)))

    step_match = re.search(r"(\d+(?:\.\d+)?)\s*steps?\b", t)
    if step_match:
        steps = float(step_match.group(1))
        return max(0.1, steps * max(0.1, seconds_per_step))

    return max(0.1, default_duration)


def parse_loco_command(
    text: str,
    walk_speed: float,
    lateral_speed: float,
    turn_speed: float,
    default_duration: float,
    seconds_per_step: float,
):
    t = normalize(text)
    if not t:
        return None

    if any(k in t for k in ["stop", "halt", "freeze", "stop moving", "stop walk"]):
        return {"vx": 0.0, "vy": 0.0, "omega": 0.0, "duration": 0.2, "is_stop": 1.0}

    has_turn_word = any(k in t for k in ["turn", "rotate", "spin", "yaw"])
    has_move_word = any(k in t for k in ["walk", "move", "go", "step"])

    vx = 0.0
    vy = 0.0
    omega = 0.0

    if has_turn_word:
        if "left" in t and "right" not in t:
            omega = abs(turn_speed)
        elif "right" in t and "left" not in t:
            omega = -abs(turn_speed)
        elif "counterclockwise" in t or "anticlockwise" in t:
            omega = abs(turn_speed)
        elif "clockwise" in t:
            omega = -abs(turn_speed)
        else:
            omega = abs(turn_speed)

    if has_move_word or any(k in t for k in ["forward", "backward", "back", "lateral", "sideways"]):
        if "forward" in t:
            vx = abs(walk_speed)
        elif "backward" in t or re.search(r"\bback\b", t):
            vx = -abs(walk_speed)

        lateral_intent = any(k in t for k in ["lateral", "sideways", "strafe"])
        if lateral_intent or (("left" in t or "right" in t) and not has_turn_word):
            if "left" in t and "right" not in t:
                vy = abs(lateral_speed)
            elif "right" in t and "left" not in t:
                vy = -abs(lateral_speed)

    if vx == 0.0 and vy == 0.0 and omega == 0.0 and has_move_word:
        if any(k in t for k in ["walk", "go", "step"]):
            vx = abs(walk_speed)

    if vx == 0.0 and vy == 0.0 and omega == 0.0:
        return None

    duration = parse_duration_seconds(t, default_duration, seconds_per_step)
    return {"vx": vx, "vy": vy, "omega": omega, "duration": duration, "is_stop": 0.0}


def apply_loco_commands(text: str, loco_client: LocoClient | None, args, dry_run: bool = False) -> bool:
    executed_any = False
    for chunk in split_requests(text):
        if has_negation(chunk) and (contains_action_hint(chunk) or contains_loco_hint(chunk)):
            print(f"Skipped (negated): {chunk}")
            continue

        loco_cmd = parse_loco_command(
            chunk,
            walk_speed=args.walk_speed,
            lateral_speed=args.lateral_speed,
            turn_speed=args.turn_speed,
            default_duration=args.default_duration,
            seconds_per_step=args.seconds_per_step,
        )
        if loco_cmd is None:
            continue

        vx = float(loco_cmd["vx"])
        vy = float(loco_cmd["vy"])
        omega = float(loco_cmd["omega"])
        duration = float(loco_cmd["duration"])

        if dry_run:
            print(f"[DRY-RUN] Would move: vx={vx:.2f}, vy={vy:.2f}, omega={omega:.2f}, duration={duration:.2f}s")
            executed_any = True
            continue

        if loco_client is None:
            print("[loco] client unavailable; skipping move command")
            continue

        code = loco_client.SetVelocity(vx, vy, omega, duration)
        if code != 0:
            print(f"Loco failed. code={code}, vx={vx}, vy={vy}, omega={omega}, duration={duration}")
            continue

        executed_any = True
        print(f"Moved: vx={vx:.2f}, vy={vy:.2f}, omega={omega:.2f}, duration={duration:.2f}s")
    return executed_any


def is_explicit_action_request(text: str, action: str | None) -> bool:
    if not action:
        return False
    t = text.lower()
    # Greeting alone should stay conversational, not action-forced.
    if action in {"high wave", "face wave"} and re.search(r"^\s*(hi|hello|hey)\b", t):
        if not re.search(r"\b(wave|gesture|do|perform|show|give)\b", t):
            return False

    return bool(
        re.search(r"\b(can you|could you|please|give me|do|perform|show|make|let's)\b", t)
        or "?" in t
    )


def is_explain_request(text: str) -> bool:
    t = text.lower().strip()
    patterns = [
        r"\bexplain\b",
        r"\bcan you explain\b",
        r"\bwhat is\b",
        r"\bhow does\b",
        r"\bhow do\b",
        r"\bwhy\b",
        r"\btell me about\b",
        r"\bhelp me understand\b",
    ]
    return any(re.search(p, t) for p in patterns)


def is_social_intro_request(text: str) -> bool:
    t = text.lower().strip()
    patterns = [
        r"\bwho are you\b",
        r"\bwhat is your name\b",
        r"\byour name\b",
        r"\bintroduce yourself\b",
    ]
    return any(re.search(p, t) for p in patterns)


def choose_action(user_text: str, reply_text: str, explain_index: int) -> tuple[str | None, int]:
    # 1) Direct user intent for supported basic actions.
    action = map_action_by_text(user_text)
    if action:
        return action, explain_index

    # 2) Social introductions should feel like greeting behavior.
    if is_social_intro_request(user_text):
        return "high wave", explain_index

    # 3) Explanatory prompts use custom human-like explaining motion variants.
    if is_explain_request(user_text):
        custom = EXPLAIN_CUSTOM_ACTIONS[explain_index % len(EXPLAIN_CUSTOM_ACTIONS)]
        return custom, explain_index + 1

    # 4) Any other non-basic conversation uses explain-style custom motion.
    custom = EXPLAIN_CUSTOM_ACTIONS[explain_index % len(EXPLAIN_CUSTOM_ACTIONS)]
    return custom, explain_index + 1


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


def ask_chat(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_text: str,
    requested_action: str | None = None,
    requested_locomotion: bool = False,
    history: list[dict] | None = None,
    enable_web_search: bool = False,
) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    if requested_action:
        messages.append({
            "role": "system",
            "content": (
                f"The user requested this supported action: {requested_action}. "
                f"Style guidance: {action_style_hint(requested_action)} "
                "Reply in 1-2 short spoken sentences. Sound warm and human, but do not use emoji. "
                "Do not explicitly narrate the action execution, and avoid lines like 'I am doing a handshake now'."
            ),
        })
    if requested_locomotion:
        messages.append({
            "role": "system",
            "content": (
                "The user asked for locomotion. Confirm naturally that you can do the movement. "
                "Do not say you cannot move. Keep it short and human-like."
            ),
        })
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    if enable_web_search:
        try:
            resp = client.responses.create(
                model=model,
                input=messages,
                tools=[{"type": "web_search_preview"}],
                temperature=0.4,
            )
            text = getattr(resp, "output_text", "") or ""
            if text.strip():
                return text.strip()
        except Exception as exc:
            print(f"[web] search unavailable, fallback to chat: {exc}")

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


def compact_spoken_reply(text: str, max_sentences: int = 2) -> str:
    if not text:
        return ""
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("* ") or line.startswith("- "):
            continue
        lines.append(line)
    cleaned = " ".join(lines)
    cleaned = re.sub(r"\[[^\]]+\]\([^)]+\)", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""

    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    out = " ".join(parts[:max_sentences]).strip()
    return out or cleaned


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


def start_motion_thread(motion: ArmGestureController, gesture: str, auto_release: bool = True) -> threading.Thread:
    def _run_motion() -> None:
        try:
            motion.execute(gesture, auto_release=auto_release)
        except Exception as exc:
            print(f"[motion] execution error: {exc}")

    t = threading.Thread(target=_run_motion, daemon=True)
    t.start()
    return t


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


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate() or 1
    return float(frames) / float(rate)


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
    parser.add_argument("--walk-speed", type=float, default=0.25, help="Forward/backward speed (m/s)")
    parser.add_argument("--lateral-speed", type=float, default=0.20, help="Lateral speed (m/s)")
    parser.add_argument("--turn-speed", type=float, default=0.30, help="Turn speed (rad/s)")
    parser.add_argument("--default-duration", type=float, default=1.5, help="Default move duration when none given (s)")
    parser.add_argument("--seconds-per-step", type=float, default=0.6, help="Duration estimate per spoken step")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and simulate motion/loco without sending robot commands")
    parser.add_argument("--disable-web-search", action="store_true", help="Disable OpenAI web search tool")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    ensure_local_pulse_env()
    set_default_pulse_sink(args.sink)
    set_default_pulse_source(args.source)

    client = OpenAI()
    motion = ArmGestureController(net_if=args.net_if, dry_run=args.dry_run)
    history: list[dict] = []
    max_history_messages = 12
    loco = None
    if not args.dry_run:
        loco = LocoClient()
        loco.SetTimeout(10.0)
        loco.Init()
    else:
        print("[DRY-RUN] Loco client disabled (no locomotion commands will be sent).")
    explain_motion_index = 0

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
            try:
                motion.release_arm()
                if loco is not None:
                    loco.StopMove()
            except Exception:
                pass
            print("Exit.")
            break

        wav_in = None
        wav_tts = None
        hold_action = False
        custom_proc = None
        try:
            wav_in = record_wav(
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
            requested_action_for_speech = requested_action if is_explicit_action_request(user_text, requested_action) else None
            requested_locomotion = contains_loco_hint(user_text) and not has_negation(user_text)
            reply = ask_chat(
                client,
                args.chat_model,
                SYSTEM_PROMPT,
                user_text,
                requested_action=requested_action_for_speech,
                requested_locomotion=requested_locomotion,
                history=history,
                enable_web_search=not args.disable_web_search,
            )
            reply = compact_spoken_reply(reply, max_sentences=2)
            if not reply:
                print("robot> [empty reply]")
                continue

            print(f"robot> {reply}")
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": reply})
            if len(history) > max_history_messages:
                history = history[-max_history_messages:]
            loco_only_request = requested_locomotion and requested_action_for_speech is None and not is_explain_request(user_text)
            if loco_only_request:
                action = None
            else:
                action, explain_motion_index = choose_action(user_text, reply, explain_motion_index)

            if action:
                print(f"[motion] selected action: {action}")

            hold_action = action == "shake hand"
            is_custom_motion = bool(action and action.startswith("custom:"))
            combo_loco_and_arm = bool(requested_locomotion and action and not is_custom_motion and not hold_action)
            motion_thread = None
            loco_thread = None

            with tempfile.NamedTemporaryFile(prefix="g1_tts_", suffix=".wav", delete=False) as tmp_tts:
                wav_tts = Path(tmp_tts.name)

            tts_to_wav(client, args.tts_model, args.voice, reply, wav_tts)
            tts_duration = wav_duration_seconds(wav_tts)

            if is_custom_motion and action:
                # Start custom explain loop and explicitly stop it when TTS ends.
                custom_proc = motion.launch_custom_process(
                    action,
                    duration_s=max(20.0, tts_duration + 5.0),
                    release_duration_s=1.2,
                )
            elif action and not combo_loco_and_arm:
                motion_thread = start_motion_thread(motion, action, auto_release=not hold_action)

            if not args.no_play:
                play_wav(wav_tts, args.sink)

            if combo_loco_and_arm:
                # For mixed command requests (e.g., walk + wave), run both together.
                loco_thread = threading.Thread(
                    target=apply_loco_commands,
                    args=(user_text, loco, args),
                    kwargs={"dry_run": args.dry_run},
                    daemon=True,
                )
                loco_thread.start()
                motion_thread = start_motion_thread(motion, action, auto_release=True)
            elif requested_locomotion:
                apply_loco_commands(user_text, loco, args, dry_run=args.dry_run)

            if is_custom_motion:
                graceful = motion.stop_custom_process(custom_proc)
                custom_proc = None
                if not graceful:
                    # Fallback release if we had to force-stop the custom script.
                    motion.release_arm()

            if hold_action:
                motion.release_arm()
                hold_action = False

            if motion_thread is not None:
                motion_thread.join(timeout=0.3)
            if loco_thread is not None:
                loco_thread.join(timeout=3.0)

        except Exception as exc:
            print(f"Error: {exc}")
        finally:
            if custom_proc is not None:
                try:
                    graceful = motion.stop_custom_process(custom_proc)
                    if not graceful:
                        motion.release_arm()
                except Exception:
                    pass
            if hold_action:
                try:
                    motion.release_arm()
                except Exception:
                    pass
            if wav_in and wav_in.exists():
                wav_in.unlink()
            if wav_tts and wav_tts.exists():
                wav_tts.unlink()


if __name__ == "__main__":
    main()
