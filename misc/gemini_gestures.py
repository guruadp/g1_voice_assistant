#!/usr/bin/env python3
"""
Gemini Live Audio + G1 arm gestures.
This is a new script and does not modify gemini.py/run_gemini.sh.
"""

import os
import asyncio
import base64
import io
import traceback
import argparse
import logging
import random
import threading
import queue
import time
import audioop

import cv2
import pyaudio
import PIL.Image
import mss

from google import genai
from google.genai import types

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient, action_map

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"
DEFAULT_MODE = "none"
DEFAULT_NET_IF = "enP8p1s0"


CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    tools=[{"google_search": {}}],
    media_resolution="MEDIA_RESOLUTION_MEDIUM",
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Schedar")
        )
    ),
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=25600,
        sliding_window=types.SlidingWindow(target_tokens=12800),
    ),
)

pya = pyaudio.PyAudio()


class ArmGestureController:
    def __init__(self, net_if: str):
        self._enabled = False
        self._q: queue.Queue[str] = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._last_gesture_ts = 0.0
        self._cooldown_sec = 1.8
        self._thread = threading.Thread(target=self._worker, daemon=True)

        try:
            ChannelFactoryInitialize(0, net_if)
            self._client = G1ArmActionClient()
            self._client.SetTimeout(10.0)
            self._client.Init()
            self._enabled = True
            self._thread.start()
            log.info("ArmGestureController ready on net_if=%s", net_if)
        except Exception as e:
            self._enabled = False
            self._client = None
            log.error("Arm client init failed. Gestures disabled: %s", e)

    def close(self):
        self._stop.set()
        try:
            self._q.put_nowait("__quit__")
        except queue.Full:
            pass

    def notify_text(self, text: str):
        if not self._enabled:
            return
        style = self._choose_style(text)
        self._enqueue(style)

    def notify_audio_turn(self):
        if not self._enabled:
            return
        style = random.choice(["high wave", "face wave"])
        self._enqueue(style)

    def notify_user_speech(self):
        if not self._enabled:
            return
        self._enqueue("high wave")

    def _enqueue(self, style: str):
        now = time.time()
        if (now - self._last_gesture_ts) < self._cooldown_sec:
            return
        try:
            self._q.put_nowait(style)
            self._last_gesture_ts = now
        except queue.Full:
            pass

    def _choose_style(self, text: str) -> str:
        t = (text or "").lower()
        if any(k in t for k in ["hello", "hi", "welcome"]):
            return "high wave"
        if any(k in t for k in ["thank", "thanks"]):
            return "face wave"
        if any(k in t for k in ["great", "awesome", "good", "nice"]):
            return "high five"
        if any(k in t for k in ["cannot", "can't", "sorry", "no"]):
            return "reject"
        return "high wave"

    def _exec_action(self, style: str):
        action_id = action_map.get(style)
        release_id = action_map.get("release arm")
        if action_id is None:
            log.warning("Unknown action style: %s", style)
            return
        log.info("Gesture action: %s", style)
        self._client.ExecuteAction(action_id)
        time.sleep(0.45)
        if release_id is not None:
            self._client.ExecuteAction(release_id)
            time.sleep(0.35)

    def _worker(self):
        while not self._stop.is_set():
            try:
                style = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if style == "__quit__":
                break
            try:
                self._exec_action(style)
            except Exception as e:
                log.warning("Gesture action failed (%s): %s", style, e)


class AudioLoop:
    def __init__(self, client, gesture_ctrl: ArmGestureController, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode
        self.client = client
        self.gesture_ctrl = gesture_ctrl

        log.info("AudioLoop initialized with video_mode: %s", self.video_mode)

        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.audio_stream = None
        self._last_user_speech_gesture_ts = 0.0
        self._user_speech_gesture_cooldown = 3.0
        self._speech_rms_threshold = 900

    def _get_frame(self, cap):
        ret, frame = cap.read()
        if not ret:
            log.warning("Failed to read frame from camera")
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        return {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(image_io.read()).decode(),
        }

    async def get_frames(self):
        log.info("Starting get_frames task (camera)")
        try:
            cap = await asyncio.to_thread(cv2.VideoCapture, 0)
            if not cap.isOpened():
                log.error("Cannot open camera")
                return
        except Exception as e:
            log.error("Error opening camera: %s", e)
            return

        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                await asyncio.sleep(0.1)
                continue

            await asyncio.sleep(1.0)
            await self.out_queue.put(frame)

    def _get_screen(self):
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                i = sct.grab(monitor)
                image_bytes = mss.tools.to_png(i.rgb, i.size)

                img = PIL.Image.open(io.BytesIO(image_bytes))
                image_io = io.BytesIO()
                img.save(image_io, format="jpeg")
                image_io.seek(0)

                return {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(image_io.read()).decode(),
                }
        except Exception as e:
            log.error("Error grabbing screen: %s", e)
            return None

    async def get_screen(self):
        log.info("Starting get_screen task")
        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                await asyncio.sleep(1.0)
                continue

            await asyncio.sleep(1.0)
            await self.out_queue.put(frame)

    async def send_realtime(self):
        log.info("Starting send_realtime task")
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        log.info("Starting listen_audio task")
        try:
            mic_info = pya.get_default_input_device_info()
            self.audio_stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=SEND_SAMPLE_RATE,
                input=True,
                input_device_index=mic_info["index"],
                frames_per_buffer=CHUNK_SIZE,
            )
        except Exception as e:
            log.error("Failed to open microphone: %s", e)
            return

        kwargs = {"exception_on_overflow": False} if __debug__ else {}
        while True:
            try:
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
                # Trigger a greeting gesture when local user speech is detected.
                rms = audioop.rms(data, 2)
                now = time.time()
                if rms > self._speech_rms_threshold and (now - self._last_user_speech_gesture_ts) > self._user_speech_gesture_cooldown:
                    self._last_user_speech_gesture_ts = now
                    self.gesture_ctrl.notify_user_speech()

                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
            except IOError as e:
                log.warning("Audio input overflow/error: %s", e)

    async def receive_audio(self):
        log.info("Starting receive_audio task")
        while True:
            got_audio_in_turn = False
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    if not got_audio_in_turn:
                        got_audio_in_turn = True
                        self.gesture_ctrl.notify_audio_turn()
                    self.audio_in_queue.put_nowait(data)
                    continue

                if text := response.text:
                    log.info("GEMINI: %s", text)
                    self.gesture_ctrl.notify_text(text)

            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        log.info("Starting play_audio task")
        try:
            stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=RECEIVE_SAMPLE_RATE,
                output=True,
            )
        except Exception as e:
            log.error("Failed to open speakers: %s", e)
            return

        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        log.info("Starting main run loop...")
        try:
            async with self.client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                self.session = session
                log.info("LiveConnect session established")

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                tasks = []
                tasks.append(asyncio.create_task(self.send_realtime()))
                tasks.append(asyncio.create_task(self.listen_audio()))

                if self.video_mode == "camera":
                    tasks.append(asyncio.create_task(self.get_frames()))
                elif self.video_mode == "screen":
                    tasks.append(asyncio.create_task(self.get_screen()))

                tasks.append(asyncio.create_task(self.receive_audio()))
                tasks.append(asyncio.create_task(self.play_audio()))

                try:
                    await asyncio.Event().wait()
                finally:
                    for t in tasks:
                        t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

        except asyncio.CancelledError:
            log.info("Run loop cancelled")
        except Exception as e:
            log.error("Exception in main loop: %s", e)
            if self.audio_stream:
                self.audio_stream.close()
            traceback.print_exc()
        finally:
            self.gesture_ctrl.close()
            pya.terminate()
            log.info("PyAudio terminated")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        choices=["camera", "screen", "none"],
        help="pixels to stream from",
    )
    parser.add_argument(
        "--net-if",
        type=str,
        default=DEFAULT_NET_IF,
        help="Robot network interface for G1 arm action client",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        log.error("GEMINI_API_KEY environment variable not set. Exiting.")
        raise SystemExit(1)

    client = genai.Client(http_options={"api_version": "v1beta"}, api_key=api_key)

    gesture_ctrl = ArmGestureController(net_if=args.net_if)
    loop = AudioLoop(client=client, gesture_ctrl=gesture_ctrl, video_mode=args.mode)
    asyncio.run(loop.run())


if __name__ == "__main__":
    main()
