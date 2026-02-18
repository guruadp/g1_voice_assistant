"""
## Documentation
Quickstart: https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started_LiveAPI.py

## Setup

To install the dependencies for this script, run:

pip install google-genai opencv-python pyaudio pillow mss

"""

import os
import asyncio
import base64
import io
import traceback

import cv2
import pyaudio
import PIL.Image
import mss

import argparse
import logging

from google import genai
from google.genai import types

# --- إعدادات التسجيل (Logging) ---
# هذا أفضل من print() للخدمات، لتتمكن من رؤية المخرجات في journalctl
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"

DEFAULT_MODE = "none"

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.environ.get("GEMINI_API_KEY"),
)


CONFIG = types.LiveConnectConfig(
    response_modalities=[
        "AUDIO",
    ],
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


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode
        log.info(f"AudioLoop initialized with video_mode: {self.video_mode}")

        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.audio_stream = None # تمت إضافة هذا المتغير هنا

    # --- تم تعطيل هذه الدالة ---
    # def send_text(self):
    #     while True:
    #         try:
    #             text = await asyncio.to_thread(input)
    #         except EOFError:
    #             print("لم يتم استلام أي إدخال.")
    #             text = None
    #
    #         if text.lower() == "q":
    #             break
    #         await self.session.send(input=text or ".", end_of_turn=True)

    def _get_frame(self, cap):
        # Read the frameq
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            log.warning("Failed to read frame from camera")
            return None
        # Fix: Convert BGR to RGB color space
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        log.info("Starting get_frames task (camera)")
        try:
            cap = await asyncio.to_thread(cv2.VideoCapture, 0)
            if not cap.isOpened():
                log.error("Cannot open camera")
                return # إنهاء المهمة إذا لم تفتح الكاميرا
        except Exception as e:
            log.error(f"Error opening camera: {e}")
            return

        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                await asyncio.sleep(0.1) # انتظر قليلاً قبل المحاولة مرة أخرى
                continue

            await asyncio.sleep(1.0) # إرسال إطار كل ثانية
            await self.out_queue.put(frame)
            log.info("Sent camera frame")

        # Release the VideoCapture object
        cap.release()
        log.info("Released camera")

    def _get_screen(self):
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1] # [0] هو كل الشاشات، [1] هي الشاشة الأساسية
                i = sct.grab(monitor)

                mime_type = "image/jpeg"
                image_bytes = mss.tools.to_png(i.rgb, i.size)
                img = PIL.Image.open(io.BytesIO(image_bytes))

                image_io = io.BytesIO()
                img.save(image_io, format="jpeg")
                image_io.seek(0)

                image_bytes = image_io.read()
                return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}
        except Exception as e:
            log.error(f"Error grabbing screen: {e}")
            return None

    async def get_screen(self):
        log.info("Starting get_screen task")
        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                log.warning("Failed to get screen frame, retrying...")
                await asyncio.sleep(1.0)
                continue

            await asyncio.sleep(1.0) # إرسال لقطة شاشة كل ثانية
            await self.out_queue.put(frame)
            log.info("Sent screen frame")

    async def send_realtime(self):
        log.info("Starting send_realtime task (audio/video)")
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        log.info("Starting listen_audio task (microphone)")
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
            log.error(f"Failed to open microphone: {e}")
            return # إنهاء المهمة إذا فشل فتح الميكروفون

        log.info("Microphone opened successfully")
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            try:
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
            except IOError as e:
                log.warning(f"Audio input overflow/error: {e}")

    async def receive_audio(self):
        log.info("Starting receive_audio task (from Gemini)")
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    log.info(f"GEMINI: {text}") # استخدام log.info بدلاً من print
                    # print(text, end="") # يمكنك إبقاء هذا السطر إذا كنت تراقب من journalctl -f

            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()
            log.info("Audio input queue cleared (end of turn)")


    async def play_audio(self):
        log.info("Starting play_audio task (speakers)")
        try:
            stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=RECEIVE_SAMPLE_RATE,
                output=True,
            )
        except Exception as e:
            log.error(f"Failed to open output stream (speakers): {e}")
            return # إنهاء المهمة إذا فشل فتح السماعات

        log.info("Speakers opened successfully")
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        log.info("Starting main run loop...")
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                log.info("LiveConnect session established")

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                # --- التعديل هنا ---
                # تم تعطيل مهمة إدخال النص لأنها لا تعمل في الخدمة
                # send_text_task = tg.create_task(self.send_text())
                
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                # --- التعديل هنا ---
                # بدلاً من انتظار مهمة النص، نجعل الخدمة تعمل للأبد
                # باستخدام await على حدث (Event) لن يتم إطلاقه أبداً.
                await asyncio.Event().wait()
                
                # --- تم تعطيل الأسطر التالية ---
                # await send_text_task
                # raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            log.info("Run loop cancelled (e.g., service stopping)")
        except ExceptionGroup as EG:
            log.error("ExceptionGroup caught in main loop")
            if self.audio_stream: # تحقق قبل الإغلاق
                self.audio_stream.close()
                log.info("Audio stream closed due to exception")
            traceback.print_exception(EG)
        finally:
            pya.terminate() # تأكد من إغلاق PyAudio عند الخروج
            log.info("PyAudio terminated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()
    
    # التحقق من وجود مفتاح API قبل البدء
    if not os.environ.get("GEMINI_API_KEY"):
        log.error("GEMINI_API_KEY environment variable not set. Exiting.")
        exit(1)
        
    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())
