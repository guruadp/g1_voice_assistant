import argparse
import os
import random
import tempfile
import threading
import wave
from pathlib import Path

from openai import OpenAI
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from audio_io import (
    ensure_local_pulse_env,
    play_wav,
    record_wav,
    set_default_pulse_sink,
    set_default_pulse_source,
)
from nlu import (
    contains_action_hint,
    contains_loco_hint,
    choose_action,
    has_negation,
    is_explain_request,
    is_explicit_action_request,
    map_action_by_text,
    parse_loco_command,
    split_requests,
)
from llm import ask_chat, compact_spoken_reply
from motion import ArmGestureController, start_motion_thread


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
