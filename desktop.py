import base64
import io
import json
import math
import os
import queue
import re
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
import threading
import time
import wave
from array import array
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from voice_capture import VoiceCapture, StreamingSpeechDetector, FRAME_SAMPLES
import mss
import requests
import sounddevice as sd
import uvicorn
from faster_whisper import WhisperModel
from PIL import Image
from PySide6.QtCore import QObject, QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication, QLabel, QMenu, QPushButton, QWidget

from app import app, OLLAMA_EXE, OLLAMA_MODEL, OLLAMA_MODELS_DIR, OLLAMA_URL

HOST = "127.0.0.1"
PORT = 8000
SERVER = f"http://{HOST}:{PORT}"

# Modelo multilíngue leve para reduzir latência e uso de RAM em CPU.
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_DOWNLOAD_ROOT = os.getenv("WHISPER_DOWNLOAD_ROOT", r"D:\ferramentas\Whisper\models")
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "4"))
SCREEN_INTERVAL_SECONDS = int(os.getenv("SCREEN_INTERVAL_SECONDS", "15"))
POSITION_FILE = Path(__file__).parent / ".mascot_position.json"
REGION_FILE = Path(__file__).parent / ".screen_region.json"

# Voice-activity threshold is adaptive, not a fixed number: every mic has a different
# noise floor (room, gain, hardware), so a hardcoded RMS cutoff either misses normal
# speech (floor too high) or never stops triggering (floor too low). We track the floor
# with an asymmetric filter -- snap DOWN fast toward quiet moments, drift UP slowly --
# so speech (which sits above the floor but might not clear a stale threshold) can't drag
# its own floor upward and chase itself forever; only genuine sustained quiet resets it.
# The neural score is smoothed over a few frames and uses separate onset/release
# thresholds, so a single VAD spike cannot open the mic and a brief dropout cannot
# cut a word. The energy gate still uses MIC_LOUD_MULTIPLIER times the learned floor
# (MIC_MIN_LOUD_RMS is a safety floor), with a stronger gate while the mascote speaks.
# Tune via .env if it is too twitchy or too deaf.
MIC_LOUD_MULTIPLIER = float(os.getenv("MIC_LOUD_MULTIPLIER", "1.6"))
MIC_MIN_LOUD_RMS = float(os.getenv("MIC_MIN_LOUD_RMS", "150"))
MIC_SILENCE_MS = int(os.getenv("MIC_SILENCE_MS", "480"))
MIC_MIN_SPEECH_MS = int(os.getenv("MIC_MIN_SPEECH_MS", "160"))
MIC_INTERRUPT_MS = int(os.getenv("MIC_INTERRUPT_MS", "608"))
MIC_VAD_ONSET_PROBABILITY = float(os.getenv("MIC_VAD_ONSET_PROBABILITY", "0.60"))
MIC_VAD_RELEASE_PROBABILITY = float(os.getenv("MIC_VAD_RELEASE_PROBABILITY", "0.35"))
MIC_VAD_ONSET_FRAMES = int(os.getenv("MIC_VAD_ONSET_FRAMES", "2"))
MIC_VAD_RELEASE_FRAMES = int(os.getenv("MIC_VAD_RELEASE_FRAMES", "3"))
MIC_VAD_SMOOTHING_FRAMES = int(os.getenv("MIC_VAD_SMOOTHING_FRAMES", "3"))

# Retain directory handles: closing them removes the DLL search paths on Windows.
_cuda_dll_handles = []


def configure_whisper_cuda():
    if os.name != "nt" or _cuda_dll_handles:
        return
    folders = [Path(os.getenv("WHISPER_CUDA_PATH", str(Path(OLLAMA_EXE).parent / "lib" / "ollama" / "cuda_v12")))]
    for folder in folders:
        if folder.is_dir():
            os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")
            _cuda_dll_handles.append(os.add_dll_directory(str(folder)))

_ollama_process = None


def _chunk_rms(data: bytes) -> float:
    if not data:
        return 0.0
    samples = array("h", data)
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def _silence_wav_bytes(duration_s: float = 0.3, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(b"\x00\x00" * int(duration_s * rate))
    return buf.getvalue()


def ollama_is_up() -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).ok
    except requests.exceptions.RequestException:
        return False


def ensure_ollama_running():
    global _ollama_process
    if ollama_is_up():
        return
    if not os.path.exists(OLLAMA_EXE):
        return

    env = os.environ.copy()
    env["OLLAMA_MODELS"] = OLLAMA_MODELS_DIR
    _ollama_process = subprocess.Popen(
        [OLLAMA_EXE, "serve"],
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )

    for _ in range(30):
        if ollama_is_up():
            break
        time.sleep(1)


def ensure_model_pulled():
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
        names = [m.get("name") for m in tags.get("models", [])]
        if not any(OLLAMA_MODEL in (name or "") for name in names):
            env = os.environ.copy()
            env["OLLAMA_MODELS"] = OLLAMA_MODELS_DIR
            subprocess.run(
                [OLLAMA_EXE, "pull", OLLAMA_MODEL],
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
    except requests.exceptions.RequestException:
        return


def preload_model():
    try:
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": "", "keep_alive": "30m",
                  "think": False, "options": {"num_ctx": 2048}},
            timeout=60,
        )
    except requests.exceptions.RequestException:
        pass


def run_server():
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


def shutdown_ollama():
    if _ollama_process and _ollama_process.poll() is None:
        _ollama_process.terminate()


def grab_screen_data_url(region: dict) -> str:
    with mss.mss() as sct:
        raw = sct.grab(region)
    image = Image.frombytes("RGB", raw.size, raw.rgb)
    max_width = 1280
    if image.width > max_width:
        ratio = max_width / image.width
        image = image.resize((max_width, round(image.height * ratio)))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{encoded}"


def button_style(small: bool = False) -> str:
    radius = 12 if small else 20
    font_size = 12 if small else 16
    return f"""
        QPushButton {{
            border-radius: {radius}px;
            background: rgba(255,255,255,0.85);
            border: 2px solid rgba(120,120,120,0.5);
            font-size: {font_size}px;
        }}
        QPushButton:checked {{
            border: 2px solid #0f6cbd;
            background: rgba(15,108,189,0.25);
        }}
        QPushButton:hover {{ background: rgba(255,255,255,1); }}
    """


STATE_COLORS = {
    "idle": QColor(120, 120, 120),
    "listening": QColor(214, 69, 55),
    "thinking": QColor(15, 108, 189),
    "speaking": QColor(200, 134, 10),
}


class Controller(QObject):
    show_answer = Signal(str)
    show_status = Signal(str)
    set_state = Signal(str)
    mic_toggle_failed = Signal()

    def __init__(self, mascot: "MascotWidget", bubble: "BubbleWidget"):
        super().__init__()
        self.mascot = mascot
        self.bubble = bubble
        self.busy_lock = threading.Lock()
        self.history: list[dict[str, str]] = []
        self.last_screen_url: str | None = None
        self.mic_stop = threading.Event()
        self.interrupt_event = threading.Event()
        self.speaking_event = threading.Event()
        self.mic_session_lock = threading.Lock()
        self.mic_thread: threading.Thread | None = None
        self.whisper_model = None
        self.whisper_load_lock = threading.Lock()
        self.transcribing_event = threading.Event()
        self.capturing_event = threading.Event()
        self.voice_queue = queue.Queue(maxsize=1)
        self.voice_answer_queue = queue.Queue(maxsize=1)
        self.voice_pending_event = threading.Event()
        self.voice_revision = 0
        self.voice_revision_lock = threading.Lock()
        threading.Thread(target=self._voice_worker, daemon=True).start()
        threading.Thread(target=self._voice_answer_worker, daemon=True).start()
        self.screen_timer: QTimer | None = None
        self.screen_region: dict | None = None
        self.region_picker: "RegionPicker | None" = None

        self.show_answer.connect(self._display)
        self.show_status.connect(self._display)
        self.set_state.connect(self.mascot.set_state)
        self.mic_toggle_failed.connect(lambda: self.mascot.mic_button.setChecked(False))

    def _display(self, text: str):
        self.bubble.show_text(text)
        self.reposition_bubble()

    def reposition_bubble(self):
        mascot_geo = self.mascot.frameGeometry()
        screen = QApplication.primaryScreen().availableGeometry()
        x = mascot_geo.left()
        y = mascot_geo.top() - self.bubble.height() - 8
        x = max(screen.left(), min(x, screen.right() - self.bubble.width()))
        if y < screen.top():
            y = mascot_geo.bottom() + 8
        self.bubble.move(x, y)

    def run_explain(self, question: str, image_url: str | None, silent: bool = False,
                    voice_revision=None, stop_event=None):
        if silent and (self.transcribing_event.is_set() or self.capturing_event.is_set()
                       or self.voice_pending_event.is_set()):
            return
        if not self.busy_lock.acquire(blocking=False):
            if not silent:
                self.show_status.emit("Aguarde a explicação atual terminar...")
            return
        answer = None
        try:
            if voice_revision is not None:
                if not self._voice_is_current(voice_revision, stop_event):
                    return
                self.interrupt_event.clear()
            elif not (self.capturing_event.is_set() or self.voice_pending_event.is_set()):
                self.interrupt_event.clear()
            self.set_state.emit("thinking")
            payload = {"image_data_url": image_url, "question": question, "history": self.history[-12:]}
            response = requests.post(f"{SERVER}/api/explain", json=payload, timeout=150)
            data = response.json()
            if response.status_code != 200:
                raise RuntimeError(data.get("detail", "Erro desconhecido"))
            answer = (data.get("answer") or "").strip()
            if not answer:
                if not silent:
                    self.show_status.emit("Nada para explicar.")
                return
            if self.interrupt_event.is_set() or (voice_revision is not None and
                    not self._voice_is_current(voice_revision, stop_event)):
                # User already started talking again while this was in flight (barge-in
                # happened during "thinking") -- drop the now-stale answer instead of
                # showing/speaking it over whatever he's asking now.
                answer = None
                return
            self.history.append({"role": "user", "text": question})
            self.history.append({"role": "assistant", "text": answer})
            if len(self.history) > 12:
                del self.history[:2]
            self.show_answer.emit(answer)
        except requests.exceptions.RequestException as error:
            answer = None
            self.show_answer.emit(f"Não consegui falar com o professor local: {error}")
        except Exception as error:
            answer = None
            self.show_answer.emit(str(error))
        finally:
            # speak() runs here, still under the lock: releasing the lock earlier let a
            # barge-in spawn a second run_explain/speak while this one was still narrating,
            # opening two concurrent output streams on the same audio device.
            if answer:
                self.speak(answer)
            if not answer and not self.interrupt_event.is_set():
                self.set_state.emit("idle")
            self.busy_lock.release()

    def speak(self, text: str):
        if self.interrupt_event.is_set():
            return
        executor = ThreadPoolExecutor(max_workers=1)
        def fetch_audio(part):
            if self.interrupt_event.is_set():
                return b""
            response = requests.post(f"{SERVER}/api/speak", json={"text": part}, timeout=50)
            response.raise_for_status()
            return response.content
        try:
            from app import strip_markdown_for_speech
            parts = [part.strip() for part in re.split(
                r"(?<=[.!?])\s+|\n+", strip_markdown_for_speech(text)) if part.strip()]
            if not parts:
                return
            future = executor.submit(fetch_audio, parts[0])
            for index in range(len(parts)):
                while not future.done():
                    if self.interrupt_event.wait(0.05):
                        future.cancel()
                        return
                audio = future.result()
                if self.interrupt_event.is_set():
                    return
                # Generate only the next sentence while the current one plays.
                if index + 1 < len(parts):
                    future = executor.submit(fetch_audio, parts[index + 1])
                with wave.open(io.BytesIO(audio), "rb") as wav_file:
                    channels = wav_file.getnchannels()
                    rate = wav_file.getframerate()
                    sampwidth = wav_file.getsampwidth()
                    frames = wav_file.readframes(wav_file.getnframes())
                dtype = {1: "uint8", 2: "int16", 4: "int32"}[sampwidth]
                chunk_bytes = (rate // 10) * sampwidth * channels
                self.set_state.emit("speaking")
                self.speaking_event.set()
                with sd.RawOutputStream(samplerate=rate, channels=channels, dtype=dtype) as stream:
                    for offset in range(0, len(frames), chunk_bytes):
                        if self.interrupt_event.is_set():
                            return
                        stream.write(frames[offset:offset + chunk_bytes])
        except (requests.exceptions.RequestException, wave.Error, ValueError, KeyError) as error:
            self.show_status.emit(f"Não consegui reproduzir a voz: {error}")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            self.speaking_event.clear()
            if not self.interrupt_event.is_set():
                self.set_state.emit("idle")

    def on_screen_toggled(self, checked: bool):
        if checked:
            if self.screen_region is None:
                self.region_picker.start()
                return
            self._start_screen_capture()
        else:
            self._stop_screen_capture()

    def _start_screen_capture(self):
        self.screen_timer = QTimer(self)
        self.screen_timer.timeout.connect(self._trigger_periodic_screen_explain)
        self.screen_timer.start(SCREEN_INTERVAL_SECONDS * 1000)
        threading.Thread(
            target=self.grab_and_explain,
            args=("Explique o inglês visível nesta tela.",),
            kwargs={"silent": False},
            daemon=True,
        ).start()

    def _stop_screen_capture(self):
        if self.screen_timer:
            self.screen_timer.stop()
            self.screen_timer = None
        self.last_screen_url = None

    def request_region_reselect(self):
        self.region_picker.start()

    def on_region_selected(self, left: int, top: int, width: int, height: int):
        self.screen_region = {"left": left, "top": top, "width": width, "height": height}
        save_region(self.screen_region)
        if not self.mascot.screen_button.isChecked():
            self.mascot.screen_button.setChecked(True)

    def on_region_cancelled(self):
        if self.screen_region is None:
            self.mascot.screen_button.setChecked(False)

    def _trigger_periodic_screen_explain(self):
        threading.Thread(
            target=self.grab_and_explain,
            args=("Explique o novo inglês visível nesta tela. Só responda se houver uma frase nova e legível.",),
            kwargs={"silent": True},
            daemon=True,
        ).start()

    def grab_and_explain(self, prompt: str, silent: bool = False):
        try:
            self.last_screen_url = grab_screen_data_url(self.screen_region)
        except Exception as error:
            self.show_answer.emit(f"Não consegui capturar a tela: {error}")
            return
        self.run_explain(prompt, self.last_screen_url, silent=silent)

    def on_mic_toggled(self, checked: bool):
        if checked:
            # Each microphone session owns its stop flag; a rapid off/on cannot revive it.
            self.mic_stop = threading.Event()
            self.mic_thread = threading.Thread(target=self._mic_loop, args=(self.mic_stop,), daemon=True)
            self.mic_thread.start()
        else:
            self.mic_stop.set()
            with self.voice_revision_lock:
                self.voice_revision += 1

    def _voice_is_current(self, revision, stop_event):
        with self.voice_revision_lock:
            return revision == self.voice_revision and not stop_event.is_set()

    def _voice_worker(self):
        # Warm the recognizer without opening the microphone or recording anything.
        try:
            self._load_whisper_model()
        except Exception as error:
            print(f"[voice] Reconhecimento não aquecido: {error}", flush=True)
        while True:
            pcm, revision, stop_event, captured_at = self.voice_queue.get()
            if not self._voice_is_current(revision, stop_event):
                continue
            self.transcribing_event.set()
            try:
                self._transcribe_and_ask(pcm, revision, stop_event, captured_at)
            except Exception as error:
                print(f"[voice] Falha no reconhecimento: {error}", flush=True)
                if self._voice_is_current(revision, stop_event):
                    self.voice_pending_event.clear()
            finally:
                self.transcribing_event.clear()

    @staticmethod
    def _put_latest(target_queue, item):
        try:
            target_queue.put_nowait(item)
        except queue.Full:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                pass
            target_queue.put_nowait(item)

    def _voice_answer_worker(self):
        while True:
            question, revision, stop_event = self.voice_answer_queue.get()
            try:
                while self.busy_lock.locked() and self._voice_is_current(revision, stop_event):
                    stop_event.wait(0.05)
                if self._voice_is_current(revision, stop_event):
                    self.run_explain(question, self.last_screen_url, silent=False,
                                     voice_revision=revision, stop_event=stop_event)
            except Exception as error:
                if self._voice_is_current(revision, stop_event):
                    self.show_status.emit(f"Não consegui responder: {error}")
            finally:
                if self._voice_is_current(revision, stop_event):
                    self.voice_pending_event.clear()
                    self.set_state.emit("listening")

    def _load_whisper_model(self):
        with self.whisper_load_lock:
            return self._load_whisper_model_locked()

    def _load_whisper_model_locked(self):
        if self.whisper_model is not None:
            return self.whisper_model
        try:
            configure_whisper_cuda()
            model = WhisperModel(
                WHISPER_MODEL_SIZE, device="cuda", compute_type="int8_float16", download_root=WHISPER_DOWNLOAD_ROOT
            )
            # CTranslate2 loads its CUDA libraries lazily, so a broken GPU setup (old
            # driver, missing cuBLAS/cuDNN DLL) doesn't fail here -- it fails on the first
            # real transcription instead. Force that now with a throwaway clip so a bad
            # GPU falls back to CPU here, once, instead of failing on every real question.
            list(model.transcribe(io.BytesIO(_silence_wav_bytes()), language="en",
                                 beam_size=1, temperature=0.0, without_timestamps=True)[0])
            self.whisper_model = model
            print(f"[voice] Whisper {WHISPER_MODEL_SIZE}: CUDA int8_float16", flush=True)
        except Exception as error:
            print(f"[voice] CUDA indisponível; usando CPU: {error}", flush=True)
            model = None
            self.whisper_model = WhisperModel(
                WHISPER_MODEL_SIZE, device="cpu", compute_type="int8", download_root=WHISPER_DOWNLOAD_ROOT,
                cpu_threads=WHISPER_CPU_THREADS,
            )
        return self.whisper_model

    def _transcribe_and_ask(self, pcm_bytes: bytes, revision, stop_event, captured_at):
        self.show_status.emit("Processando fala...")
        question = ""
        try:
            import numpy as np
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            # language=None: auto-detect per utterance, needed since the user mixes
            # Portuguese and English in the same sentence while practicing.
            started = time.perf_counter()
            segments, _ = self._load_whisper_model().transcribe(
                samples, language=None, vad_filter=True,
                beam_size=1, best_of=1, temperature=0.0,
                condition_on_previous_text=False, without_timestamps=True,
                vad_parameters={"min_silence_duration_ms": 160, "speech_pad_ms": 96},
            )
            question = " ".join(segment.text.strip() for segment in segments).strip()
            print(f"[voice] transcription_seconds={time.perf_counter() - started:.2f} "
                  f"after_capture_seconds={time.perf_counter() - captured_at:.2f}", flush=True)
        except Exception as error:
            print(f"[mic-debug] transcribe error: {error!r}", flush=True)
            if self._voice_is_current(revision, stop_event):
                self.show_answer.emit(f"Não consegui entender o áudio: {error}")

        if not self._voice_is_current(revision, stop_event):
            return

        if not question:
            self.voice_pending_event.clear()
            if not stop_event.is_set():
                self.set_state.emit("listening")
            return

        self.show_status.emit(f"Você disse: {question}")
        self._put_latest(self.voice_answer_queue, (question, revision, stop_event))

    def _mic_loop(self, stop_event):
        # Each session has its own stop flag and exclusive ownership of the input stream.
        with self.mic_session_lock:
            if not stop_event.is_set():
                self._capture_microphone(stop_event)

    def _capture_microphone(self, stop_event):
        audio_queue = queue.Queue(maxsize=32)
        discontinuity = threading.Event()

        def new_segmenter():
            return VoiceCapture(
                silence_ms=MIC_SILENCE_MS, min_speech_ms=MIC_MIN_SPEECH_MS,
                interrupt_ms=MIC_INTERRUPT_MS, min_rms=MIC_MIN_LOUD_RMS,
                multiplier=MIC_LOUD_MULTIPLIER,
                vad_onset_probability=MIC_VAD_ONSET_PROBABILITY,
                vad_release_probability=MIC_VAD_RELEASE_PROBABILITY,
                vad_onset_frames=MIC_VAD_ONSET_FRAMES,
                vad_release_frames=MIC_VAD_RELEASE_FRAMES,
            )

        def new_detector():
            return StreamingSpeechDetector(smoothing_frames=MIC_VAD_SMOOTHING_FRAMES)

        segmenter = new_segmenter()
        try:
            detector = new_detector()
        except Exception as error:
            print(f"[voice] Detector neural indisponível: {error}", flush=True)
            self.show_answer.emit(f"Não consegui preparar a captura de voz: {error}")
            self.mic_toggle_failed.emit()
            return
        def callback(indata, frames, time_info, status):
            if stop_event.is_set():
                return
            if status:
                discontinuity.set()
            try:
                audio_queue.put_nowait(bytes(indata))
            except queue.Full:
                discontinuity.set()
                self._put_latest(audio_queue, bytes(indata))

        try:
            with sd.RawInputStream(samplerate=16000, blocksize=FRAME_SAMPLES,
                                   dtype="int16", channels=1, callback=callback):
                self.set_state.emit("listening")
                while not stop_event.is_set():
                    try:
                        data = audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if discontinuity.is_set():
                        segmenter = new_segmenter()
                        detector = new_detector()
                        self.capturing_event.clear()
                        self.voice_pending_event.clear()
                        discontinuity.clear()
                    was_capturing = segmenter.capturing
                    started, interrupt, complete = segmenter.feed(
                        data, busy=self.busy_lock.locked(),
                        speaking=self.speaking_event.is_set(),
                        speech_probability=detector.probability(data))
                    if started:
                        with self.voice_revision_lock:
                            self.voice_revision += 1
                            revision = self.voice_revision
                        self.capturing_event.set()
                        self.voice_pending_event.set()
                        if interrupt:
                            self.interrupt_event.set()
                        self.show_status.emit("Ouvindo...")
                        self.set_state.emit("listening")
                    if complete is not None:
                        self.capturing_event.clear()
                        self._put_latest(self.voice_queue,
                                         (complete, revision, stop_event, time.perf_counter()))
                        self.show_status.emit("Entendendo sua frase...")
                    elif was_capturing and not segmenter.capturing:
                        self.capturing_event.clear()
                        self.voice_pending_event.clear()
                        self.show_status.emit("Esperando você falar...")
        except Exception as error:
            if not stop_event.is_set():
                self.show_answer.emit(f"Não consegui usar o microfone: {error}")
                self.mic_toggle_failed.emit()
        finally:
            self.capturing_event.clear()
            if stop_event is self.mic_stop:
                self.voice_pending_event.clear()
                self.set_state.emit("idle")



class MascotWidget(QWidget):
    dragged = Signal()
    select_region_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setFixedSize(120, 150)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._drag_offset: QPoint | None = None
        self._state = "idle"

        self.mic_button = QPushButton("🎤", self)
        self.mic_button.setCheckable(True)
        self.mic_button.setGeometry(8, 96, 40, 40)
        self.mic_button.setStyleSheet(button_style())
        self.mic_button.setToolTip("Ligar/desligar microfone")

        self.screen_button = QPushButton("🖥", self)
        self.screen_button.setCheckable(True)
        self.screen_button.setGeometry(72, 96, 40, 40)
        self.screen_button.setStyleSheet(button_style())
        self.screen_button.setToolTip("Ligar/desligar compartilhamento de tela")

        self.menu_button = QPushButton("⋮", self)
        self.menu_button.setGeometry(46, 4, 24, 24)
        self.menu_button.setStyleSheet(button_style(small=True))
        self.menu_button.setToolTip("Menu")
        self.menu_button.clicked.connect(self._show_menu)

    def set_state(self, state: str):
        self._state = state
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = STATE_COLORS.get(self._state, STATE_COLORS["idle"])

        painter.setBrush(QBrush(QColor(255, 255, 255, 235)))
        painter.setPen(QPen(color, 4))
        painter.drawEllipse(10, 10, 80, 80)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(30, 30, 30)))
        eye_y = 42 if self._state == "listening" else 45
        painter.drawEllipse(32, eye_y, 10, 10)
        painter.drawEllipse(58, eye_y, 10, 10)

        painter.setBrush(QBrush(color))
        if self._state == "speaking":
            painter.drawEllipse(38, 62, 24, 14)
        else:
            painter.drawEllipse(38, 66, 24, 6)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event):
        if self._drag_offset is not None:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            self.dragged.emit()

    def mouseReleaseEvent(self, event):
        self._drag_offset = None

    def _show_menu(self):
        menu = QMenu(self)
        region_action = QAction("Selecionar área da tela", self)
        region_action.triggered.connect(self.select_region_requested.emit)
        menu.addAction(region_action)
        quit_action = QAction("Sair", self)
        quit_action.triggered.connect(QApplication.instance().quit)
        menu.addAction(quit_action)
        menu.exec(self.menu_button.mapToGlobal(QPoint(0, self.menu_button.height())))


class BubbleWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.label = QLabel(self)
        self.label.setWordWrap(True)
        self.label.setFont(QFont("Segoe UI", 10))
        self.label.setStyleSheet("color: white; padding: 10px;")
        self.label.move(0, 0)

    def show_text(self, text: str):
        self.label.setFixedWidth(300)
        self.label.setText(text)
        self.label.adjustSize()
        self.setFixedSize(self.label.width(), self.label.height())
        self.show()
        self.raise_()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(QColor(20, 20, 20, 220)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 10, 10)

    def mousePressEvent(self, event):
        self.hide()


class RegionPicker(QWidget):
    selected = Signal(int, int, int, int)
    cancelled = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._origin = QPoint(0, 0)
        self._start: QPoint | None = None
        self._end: QPoint | None = None

    def start(self):
        virtual_geometry = QRect()
        for screen in QApplication.screens():
            virtual_geometry = virtual_geometry.united(screen.geometry())
        self.setGeometry(virtual_geometry)
        self._origin = virtual_geometry.topLeft()
        self._start = None
        self._end = None
        self.show()
        self.raise_()
        self.activateWindow()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 90))
        if self._start and self._end:
            selection = QRect(self._start, self._end).normalized()
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(selection, Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.setPen(QPen(QColor(15, 108, 189), 2))
            painter.drawRect(selection)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._start = event.position().toPoint()
            self._end = self._start
            self.update()

    def mouseMoveEvent(self, event):
        if self._start is not None:
            self._end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if self._start is None or self._end is None:
            return
        selection = QRect(self._start, self._end).normalized()
        self._start = None
        self._end = None
        self.hide()
        if selection.width() >= 20 and selection.height() >= 20:
            self.selected.emit(
                selection.left() + self._origin.x(),
                selection.top() + self._origin.y(),
                selection.width(),
                selection.height(),
            )
        else:
            self.cancelled.emit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._start = None
            self._end = None
            self.hide()
            self.cancelled.emit()


def load_region() -> dict | None:
    try:
        data = json.loads(REGION_FILE.read_text())
        return {"left": data["left"], "top": data["top"], "width": data["width"], "height": data["height"]}
    except (OSError, ValueError, KeyError):
        return None


def save_region(region: dict):
    try:
        REGION_FILE.write_text(json.dumps(region))
    except OSError:
        pass


def load_position() -> QPoint:
    try:
        data = json.loads(POSITION_FILE.read_text())
        return QPoint(data["x"], data["y"])
    except (OSError, ValueError, KeyError):
        screen = QApplication.primaryScreen().availableGeometry()
        return QPoint(screen.right() - 140, screen.bottom() - 170)


def save_position(point: QPoint):
    try:
        POSITION_FILE.write_text(json.dumps({"x": point.x(), "y": point.y()}))
    except OSError:
        pass


def main():
    ensure_ollama_running()
    ensure_model_pulled()
    threading.Thread(target=preload_model, daemon=True).start()
    threading.Thread(target=run_server, daemon=True).start()

    qt_app = QApplication(sys.argv)
    qt_app.setQuitOnLastWindowClosed(False)

    mascot = MascotWidget()
    bubble = BubbleWidget()
    region_picker = RegionPicker()
    mascot.move(load_position())

    ctl = Controller(mascot, bubble)
    ctl.region_picker = region_picker
    ctl.screen_region = load_region()
    mascot.mic_button.toggled.connect(ctl.on_mic_toggled)
    mascot.screen_button.toggled.connect(ctl.on_screen_toggled)
    mascot.select_region_requested.connect(ctl.request_region_reselect)
    mascot.dragged.connect(ctl.reposition_bubble)
    region_picker.selected.connect(ctl.on_region_selected)
    region_picker.cancelled.connect(ctl.on_region_cancelled)

    def on_quit():
        save_position(mascot.pos())
        ctl.mic_stop.set()
        shutdown_ollama()

    qt_app.aboutToQuit.connect(on_quit)

    mascot.show()
    ctl.show_status.emit("Oi! Clique no microfone ou na tela pra eu começar a te ajudar.")

    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
