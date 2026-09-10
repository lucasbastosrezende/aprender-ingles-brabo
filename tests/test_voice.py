import io
import unittest
import wave

import numpy as np

from voice_capture import VoiceCapture, StreamingSpeechDetector, FRAME_SAMPLES, FRAME_MS
from voice_style import style_voice


def block(level):
    return np.full(FRAME_SAMPLES, level, dtype=np.int16).tobytes()


def wav(samples, rate=22050):
    result = io.BytesIO()
    with wave.open(result, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(samples.astype(np.int16).tobytes())
    return result.getvalue()


class CaptureTests(unittest.TestCase):
    def test_onset_and_quiet_consonants_are_retained(self):
        capture = VoiceCapture()
        for _ in range(10):
            capture.feed(block(0))
        for _ in range(8):
            capture.feed(block(1000))
        output = None
        for index in range(15):
            _, _, output = capture.feed(block(50))
            if index < 14:
                self.assertIsNone(output)
        samples = np.frombuffer(output, dtype=np.int16)
        self.assertEqual(np.count_nonzero(samples == 1000), 8 * FRAME_SAMPLES)
        self.assertEqual(np.count_nonzero(samples == 50), 3 * FRAME_SAMPLES)
        self.assertEqual(15 * FRAME_MS, 480)

    def test_click_does_not_become_a_question(self):
        capture = VoiceCapture()
        complete = []
        for level in [0] * 8 + [2000] * 2 + [0] * 20:
            result = capture.feed(block(level))[2]
            if result is not None:
                complete.append(result)
        self.assertEqual(complete, [])

    def test_barge_in_preserves_audio_before_interrupt(self):
        capture = VoiceCapture()
        for _ in range(8):
            capture.feed(block(0), busy=True, speaking=True)
        for index in range(capture.interrupt_frames):
            started, interrupt, _ = capture.feed(block(2000), busy=True, speaking=True)
            self.assertEqual(interrupt, index == capture.interrupt_frames - 1)
        for _ in range(15):
            _, _, audio = capture.feed(block(0))
        self.assertEqual(np.count_nonzero(np.frombuffer(audio, dtype=np.int16) == 2000),
                         capture.interrupt_frames * FRAME_SAMPLES)

    def test_continuous_noise_cannot_grow_buffer_forever(self):
        capture = VoiceCapture(max_speech_ms=1000)
        finished = []
        for _ in range(100):
            result = capture.feed(block(1000))[2]
            if result:
                finished.append(result)
        self.assertTrue(finished)
        self.assertLessEqual(max(map(len, finished)), 32 * FRAME_SAMPLES * 2)

    def test_floor_does_not_chase_speech(self):
        capture = VoiceCapture()
        for _ in range(100):
            capture.feed(block(1000))
        self.assertLessEqual(capture.floor, 150)

    def test_loud_room_noise_does_not_start_neural_capture(self):
        capture = VoiceCapture()
        for _ in range(100):
            started, interrupt, audio = capture.feed(block(2800), speech_probability=0.02)
            self.assertFalse(started)
            self.assertIsNone(audio)
        self.assertGreater(capture.floor, 2500)
        # Voice need not be 1.6 times louder than this learned ambient level.
        capture.feed(block(3000), speech_probability=0.9)
        self.assertTrue(capture.feed(block(3000), speech_probability=0.9)[0])

    def test_isolated_neural_spike_does_not_start_capture(self):
        capture = VoiceCapture()
        events = []
        for probability in [0.95, 0.02] * 20:
            events.append(capture.feed(block(1200), speech_probability=probability))
        self.assertFalse(any(started for started, _, _ in events))
        self.assertFalse(capture.capturing)

    def test_neural_hangover_keeps_short_probability_dropout(self):
        capture = VoiceCapture()
        for _ in range(5):
            capture.feed(block(1200), speech_probability=0.9)
        self.assertTrue(capture.capturing)
        for _ in range(2):
            started, interrupt, complete = capture.feed(block(1200), speech_probability=0.05)
            self.assertFalse(interrupt)
            self.assertIsNone(complete)
            self.assertTrue(capture.capturing)

    def test_streaming_detector_median_rejects_one_frame_spike(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        class FakeSession:
            def __init__(self):
                self.outputs = iter([0.95, 0.02, 0.02])

            def get_inputs(self):
                return [SimpleNamespace(name=name) for name in ("input", "h", "c")]

            def run(self, _, inputs):
                probability = next(self.outputs)
                return np.array([[probability]], dtype=np.float32), inputs["h"], inputs["c"]

        fake_model = SimpleNamespace(session=FakeSession())
        with patch("faster_whisper.vad.get_vad_model", return_value=fake_model):
            detector = StreamingSpeechDetector(smoothing_frames=3)
            probabilities = [detector.probability(block(1200)) for _ in range(3)]
        self.assertGreater(probabilities[0], 0.9)
        self.assertLess(probabilities[1], 0.6)
        self.assertLess(probabilities[2], 0.1)

    def test_streaming_detector_rejects_silence_and_noise(self):
        detector = StreamingSpeechDetector()
        rng = np.random.default_rng(42)
        probabilities = []
        for _ in range(40):
            noise = rng.normal(0, 2800, FRAME_SAMPLES).astype(np.int16).tobytes()
            probabilities.append(detector.probability(noise))
        self.assertLess(max(probabilities), 0.5)


class QueueTests(unittest.TestCase):
    def test_only_latest_pending_utterance_is_kept(self):
        import queue
        from desktop import Controller
        pending = queue.Queue(maxsize=1)
        Controller._put_latest(pending, "old")
        Controller._put_latest(pending, "new")
        self.assertEqual(pending.get_nowait(), "new")
        self.assertTrue(pending.empty())

    def test_finished_transcription_is_discarded_after_mic_off(self):
        import threading
        import time
        from types import SimpleNamespace
        from unittest.mock import Mock
        from desktop import Controller
        ctl = Mock()
        stop = threading.Event()
        def transcribe(*args, **kwargs):
            stop.set()
            return iter([SimpleNamespace(text="Do not send this")]), None
        ctl._load_whisper_model.return_value.transcribe = transcribe
        ctl._voice_is_current.side_effect = lambda revision, event: not event.is_set()
        Controller._transcribe_and_ask(ctl, block(1000), 1, stop, time.perf_counter())
        ctl._put_latest.assert_not_called()

    def test_new_question_replaces_stale_transcription(self):
        import threading
        import time
        from types import SimpleNamespace
        from unittest.mock import Mock
        from desktop import Controller
        ctl = Mock()
        ctl.voice_revision = 1
        def transcribe(*args, **kwargs):
            ctl.voice_revision = 2
            return iter([SimpleNamespace(text="Old question")]), None
        ctl._load_whisper_model.return_value.transcribe = transcribe
        ctl._voice_is_current.side_effect = lambda revision, event: revision == ctl.voice_revision
        Controller._transcribe_and_ask(ctl, block(1000), 1, threading.Event(), time.perf_counter())
        ctl._put_latest.assert_not_called()


class StyleTests(unittest.TestCase):
    def test_pitch_rises_without_speeding_up_by_same_amount(self):
        rate = 22050
        samples = np.sin(2 * np.pi * 220 * np.arange(rate * 2) / rate) * 10000
        output = style_voice(wav(samples), "Vamos aprender.", semitones=2)
        with wave.open(io.BytesIO(output)) as reader:
            self.assertEqual(reader.getframerate(), rate)
            data = np.frombuffer(reader.readframes(reader.getnframes()), dtype=np.int16)
        peak = np.argmax(abs(np.fft.rfft(data))) * rate / len(data)
        self.assertAlmostEqual(peak, 220 * 2 ** (2 / 12), delta=3)
        self.assertAlmostEqual(len(data) / rate, 2 / 1.035, delta=0.08)
        self.assertLess(np.max(abs(data.astype(np.int32))), 32767)

    def test_original_voice_can_be_selected(self):
        data = wav(np.zeros(1000))
        self.assertEqual(style_voice(data, "Oi!", semitones=0), data)


if __name__ == "__main__":
    unittest.main()
