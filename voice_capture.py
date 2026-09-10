"""Adaptive microphone segmentation with temporal neural voice detection."""
import math
from collections import deque

import numpy as np

SAMPLE_RATE = 16000
FRAME_MS = 32
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


class StreamingSpeechDetector:
    """Run Silero v6 on a stream and smooth frame-level probability spikes."""
    def __init__(self, smoothing_frames=3):
        from faster_whisper.vad import get_vad_model

        if smoothing_frames < 1:
            raise ValueError("smoothing_frames precisa ser pelo menos 1.")
        self.session = get_vad_model().session
        if {item.name for item in self.session.get_inputs()} != {"input", "h", "c"}:
            raise RuntimeError("Instale faster-whisper 1.2.1 para o detector de fala.")
        self.smoothing_frames = int(smoothing_frames)
        self.probabilities = deque(maxlen=self.smoothing_frames)
        self.raw_probability = 0.0
        self.smoothed_probability = 0.0
        self.reset()

    def reset(self):
        """Clear recurrent state after a dropped/reordered audio block."""
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros_like(self.h)
        self.context = np.zeros(64, dtype=np.float32)
        self.probabilities.clear()
        self.raw_probability = 0.0
        self.smoothed_probability = 0.0

    def probability(self, pcm):
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if len(samples) != FRAME_SAMPLES:
            raise ValueError("O detector espera blocos de 512 amostras.")
        inputs = np.concatenate((self.context, samples)).reshape(1, -1)
        output, self.h, self.c = self.session.run(None, {"input": inputs, "h": self.h, "c": self.c})
        self.context = samples[-64:].copy()
        raw = float(np.clip(output.reshape(-1)[0], 0.0, 1.0))
        self.raw_probability = raw
        self.probabilities.append(raw)
        values = np.fromiter(self.probabilities, dtype=np.float32)
        # A short median is a majority decision: one isolated neural spike is
        # removed, while two consecutive speech frames still pass quickly.
        self.smoothed_probability = float(np.median(values))
        return self.smoothed_probability


class VoiceCapture:
    def __init__(self, silence_ms=480, min_speech_ms=160, interrupt_ms=608,
                 max_speech_ms=15000, min_rms=150, multiplier=1.6,
                 vad_onset_probability=0.60, vad_release_probability=0.35,
                 vad_onset_frames=2, vad_release_frames=3):
        self.silence_frames = math.ceil(silence_ms / FRAME_MS)
        self.min_speech_frames = math.ceil(min_speech_ms / FRAME_MS)
        self.interrupt_frames = math.ceil(interrupt_ms / FRAME_MS)
        self.max_frames = math.ceil(max_speech_ms / FRAME_MS)
        self.min_rms = min_rms
        self.multiplier = multiplier
        self.vad_onset_probability = float(vad_onset_probability)
        self.vad_release_probability = float(vad_release_probability)
        self.vad_onset_frames = max(1, int(vad_onset_frames))
        self.vad_release_frames = max(1, int(vad_release_frames))
        if not 0.0 <= self.vad_release_probability < self.vad_onset_probability <= 1.0:
            raise ValueError("Os limiares do VAD precisam obedecer 0 <= saída < entrada <= 1.")
        self.floor = min_rms
        # Also preserve the beginning of a sentence spoken over the teacher.
        self.preroll = deque(maxlen=self.interrupt_frames + math.ceil(192 / FRAME_MS))
        self.buffer = []
        self.loud_streak = 0
        self.speech_frames = 0
        self.quiet_frames = 0
        self.vad_score = 0.0
        self.vad_active = False
        self.vad_onset_streak = 0
        self.vad_release_streak = 0

    @property
    def capturing(self):
        return bool(self.buffer)

    def _update_vad(self, probability):
        """Apply attack/release smoothing and hysteresis to Silero output."""
        probability = float(probability)
        if not math.isfinite(probability):
            probability = 0.0
        probability = min(1.0, max(0.0, probability))

        # Attack quickly enough for a quiet word, release slowly enough to keep
        # short unvoiced consonants and brief neural-confidence dropouts.
        alpha = 0.75 if probability >= self.vad_score else 0.18
        self.vad_score += alpha * (probability - self.vad_score)
        if self.vad_active:
            if self.vad_score < self.vad_release_probability:
                self.vad_release_streak += 1
                if self.vad_release_streak >= self.vad_release_frames:
                    self.vad_active = False
                    self.vad_release_streak = 0
            else:
                self.vad_release_streak = 0
        # Do not let the slow release accumulator turn alternating spikes into
        # speech. Entry needs both sustained confidence and a high current frame;
        # the slower score is reserved for keeping an already active utterance.
        elif (self.vad_score >= self.vad_onset_probability
              and probability >= self.vad_onset_probability):
            self.vad_onset_streak += 1
            if self.vad_onset_streak >= self.vad_onset_frames:
                self.vad_active = True
                self.vad_onset_streak = 0
        else:
            self.vad_onset_streak = 0
        return self.vad_active

    def feed(self, pcm, busy=False, speaking=False, speech_probability=None):
        """Return (started, interrupt, complete_pcm). No recording beyond this call."""
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        threshold = max(self.floor * self.multiplier, self.min_rms)
        if self.capturing:
            threshold *= 0.75  # Preserve quieter syllables after speech has begun.
        elif speaking:
            threshold *= 1.8  # Require stronger sustained input over speaker playback.
        loud = rms >= threshold
        if speech_probability is not None:
            neural_loud = self._update_vad(speech_probability)
            # Neural VAD distinguishes voice from fans/clicks even in a loud room.
            # Keep an amplitude gate over playback, which VAD also recognizes as speech.
            energy_gate = rms >= (
                threshold if speaking and not self.capturing else self.min_rms * 0.5)
            loud = neural_loud and energy_gate
            # For barge-in, count a strong candidate from its first frame so the
            # configured interrupt time remains meaningful. A brief spike still
            # cannot reach interrupt_frames and never interrupts by itself.
            if not self.capturing and busy and energy_gate:
                loud = neural_loud or speech_probability >= self.vad_onset_probability
        self.loud_streak = self.loud_streak + 1 if loud else 0
        if not self.capturing and not busy and not loud:
            alpha = 0.15 if rms < self.floor else 0.05 if speech_probability is not None else 0.003
            self.floor += alpha * (rms - self.floor)
        started = interrupt = False
        if not self.capturing:
            self.preroll.append(pcm)
            # The neural gate already requires persistence for normal capture;
            # the energy-only fallback keeps the previous two-frame debounce.
            needed = self.interrupt_frames if busy else 1 if speech_probability is not None else 2
            if self.loud_streak < needed:
                return False, False, None
            self.buffer = list(self.preroll)
            self.preroll.clear()
            self.speech_frames = self.loud_streak
            self.quiet_frames = 0
            started, interrupt = True, busy
        else:
            self.buffer.append(pcm)
            if loud:
                self.speech_frames += 1
                self.quiet_frames = 0
            else:
                self.quiet_frames += 1
        complete = None
        if self.quiet_frames >= self.silence_frames or len(self.buffer) >= self.max_frames:
            if self.speech_frames >= self.min_speech_frames:
                # Retain 96 ms of final consonants, not all the endpoint waiting time.
                trim = max(0, self.quiet_frames - 3)
                frames = self.buffer[:-trim] if trim else self.buffer
                complete = b"".join(frames)
            self.buffer = []
            self.speech_frames = self.quiet_frames = self.loud_streak = 0
        return started, interrupt, complete
