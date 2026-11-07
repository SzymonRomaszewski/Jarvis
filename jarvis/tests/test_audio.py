import os
import queue
import sys
import types
import unittest
from collections import deque
from types import SimpleNamespace as NS
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from helpers import JarvisTestCase  # noqa: E402

RNG = np.random.default_rng(7)


def sine(n, rms):
    return rms * 1.41 * np.sin(np.arange(n) / 6)


class AudioTestCase(JarvisTestCase):
    def setUp(self):
        super().setUp()
        self.clock = [100.0]
        patcher = mock.patch.object(self.j.time, "time", lambda: self.clock[0])
        patcher.start()
        self.addCleanup(patcher.stop)

    def ears(self, speech=(8, 28), voice_rms=900.0, noise_rms=90.0, loud_first=0, **attrs):
        """An Ears whose microphone plays a script: optional loud chime, silence, then speech."""
        outer = self

        class Stream:
            n = 0

            def read(s, frames):
                outer.clock[0] += 0.08
                s.n += 1
                x = RNG.normal(0, noise_rms, frames)
                if s.n <= loud_first:
                    x = x + sine(frames, 4000)
                elif speech and speech[0] <= s.n <= speech[1]:
                    x = x + sine(frames, voice_rms)
                return x.astype(np.int16).reshape(-1, 1), False

        e = self.j.Ears.__new__(self.j.Ears)
        e.np, e.stream, e.noise, e.threshold = np, Stream(), 141.0, 282.0
        e.recent, e.last_peak, e.last_thr = deque([700.0], maxlen=15), 0.0, 0.0
        for k, v in attrs.items():
            setattr(e, k, v)
        return e


class RecordTests(AudioTestCase):
    def test_default_needed_level_is_1700(self):
        self.assertEqual(self.j.MIC_LEVEL, 1700.0)
        e = self.ears(voice_rms=3000)
        self.assertIsNotNone(e.record())
        self.assertEqual(e.last_thr, 1700.0)

    def test_the_level_is_configurable_and_zero_means_automatic(self):
        j = __import__("helpers").load_jarvis({"JARVIS_MIC_LEVEL": "900"})
        self.assertEqual(j.MIC_LEVEL, 900.0)
        self.j.MIC_LEVEL = 0.0
        e = self.ears(voice_rms=240)
        e.recent = deque([700.0], maxlen=15)
        e.record()
        self.assertLess(e.last_thr, 1700.0)

    def test_quiet_voice_is_still_sent_to_whisper(self):
        e = self.ears(voice_rms=300)               # below the 1700 line
        audio = e.record()
        self.assertIsNotNone(audio)                # whisper gets the final say...
        self.assertGreaterEqual(len(audio) / 16000, self.j.NO_SPEECH_TIMEOUT - 0.2)   # ...after the no-speech timeout

    def test_silence_returns_nothing(self):
        self.assertIsNone(self.ears(speech=None).record())

    def test_strict_mode_ignores_borderline_sound(self):
        e = self.ears(voice_rms=300)
        self.assertIsNone(e.record(timeout=2.0, strict=True))

    def test_recording_stops_after_the_silence_window(self):
        e = self.ears(voice_rms=3000, speech=(8, 20))
        audio = e.record()
        seconds = len(audio) / 16000
        self.assertAlmostEqual(seconds, 20 * 0.08 + self.j.SILENCE_SECONDS, delta=0.4)

    def test_the_chime_cannot_fool_the_recorder(self):
        # a loud chime for the first 0.4 s, then 1.2 s of silence, then the user speaks
        e = self.ears(voice_rms=3000, speech=(20, 32), loud_first=5)
        audio = e.record()
        self.assertGreater(len(audio) / 16000, 32 * 0.08 - 0.2)           # speech after the pause was captured

    def test_calibration_sets_the_threshold(self):
        e = self.ears(speech=None)
        e.calibrate()
        self.assertEqual(e.threshold, 1700.0)
        self.j.MIC_LEVEL = 0.0
        e = self.ears(speech=None)
        e.calibrate()
        self.assertAlmostEqual(e.threshold, max(self.j.MIN_SPEECH_RMS, e.noise * 1.7), delta=1)


class StreamTests(AudioTestCase):
    def test_flush_drains_without_stopping_the_microphone(self):
        class S:
            avail, stops, reads = 5000, 0, 0

            @property
            def read_available(s): return s.avail

            def read(s, n): s.avail -= n; s.reads += 1; return (None, False)
            def abort(s): s.stops += 1
            def start(s): s.stops += 1
        e = self.j.Ears.__new__(self.j.Ears)
        e.stream = S()
        e.flush()
        self.assertLessEqual(e.stream.avail, 0)
        self.assertEqual(e.stream.stops, 0)
        self.assertGreaterEqual(e.stream.reads, 4)

    def test_digital_gain_is_applied_and_clipped(self):
        class S:
            def read(s, n): return np.full((n, 1), 400, dtype=np.int16), False
        e = self.j.Ears.__new__(self.j.Ears)
        e.np, e.stream, e.recent = np, S(), deque(maxlen=15)
        self.j.MIC_GAIN = 1.0
        self.assertEqual(e.read()[0], 400)
        self.j.MIC_GAIN = 3.0
        self.assertEqual(e.read()[0], 1200)
        self.j.MIC_GAIN = 200.0
        self.assertEqual(e.read()[0], 32767)

    def test_reads_are_serialised(self):
        self.assertTrue(hasattr(self.j.Ears._lock, "acquire"))


class TranscribeTests(JarvisTestCase):
    def run_whisper(self, text, segments, audio):
        seen = {}
        fake = types.ModuleType("mlx_whisper")

        def transcribe(a, **kw):
            seen["audio"] = a
            return {"text": text, "segments": segments}
        fake.transcribe = transcribe
        return self.j.transcribe(fake, audio), seen

    def test_quiet_speech_is_boosted_loud_is_untouched_silence_stays_quiet(self):
        tone = lambda amp: (amp * np.sin(np.arange(32000) / 5)).astype(np.int16)  # noqa: E731
        _, seen = self.run_whisper("x", [], tone(350))
        self.assertGreater(float(np.percentile(np.abs(seen["audio"]), 99)), 0.2)
        _, seen = self.run_whisper("x", [], tone(20000))                       # already loud: left alone
        self.assertAlmostEqual(float(np.abs(seen["audio"]).max()), 20000 / 32768, places=2)
        _, seen = self.run_whisper("x", [], RNG.normal(0, 3, 32000).astype(np.int16))
        self.assertLess(float(np.abs(seen["audio"]).max()), 0.05)

    def test_hallucination_filters(self):
        audio = np.ones(16000, dtype=np.int16)
        text, _ = self.run_whisper(" Open Safari.", [{"no_speech_prob": 0.7, "avg_logprob": -0.3}], audio)
        self.assertEqual(text, "Open Safari.")                                        # quiet but real speech kept
        text, _ = self.run_whisper(" Thanks for watching!", [{"no_speech_prob": 0.9, "avg_logprob": -1.8}], audio)
        self.assertEqual(text, "")
        self.assertIn("watching", self.j.transcribe.raw)                              # raw text kept for debugging
        text, _ = self.run_whisper(" Thank you.", [{"no_speech_prob": 0.1, "avg_logprob": -0.2}], audio)
        self.assertEqual(text, "")


class ClapTests(AudioTestCase):
    def sequence(self, kinds, typing_idle=None, step=0.08):
        j = self.j
        if typing_idle is not None:
            j._seconds_since_input = typing_idle
        det = j.ClapDetector(np, 100.0)
        hits = 0
        for k in kinds:
            self.clock[0] += step
            x = RNG.normal(0, 80, 1280)
            if k == "c":
                x[400:600] += RNG.normal(0, 1, 200) * 14000 * np.exp(-np.arange(200) / 50)
            elif k == "s":
                x += 5000 * np.sin(np.arange(1280) / 7)
            hits += det.feed(np.clip(x, -32768, 32767).astype(np.int16))
        return hits

    def test_double_clap(self):
        self.assertEqual(self.sequence("cqqqqqc", lambda: 99.0), 1)

    def test_single_clap_far_apart_loud_speech_and_quiet(self):
        self.assertEqual(self.sequence("qqcqqqqqq", lambda: 99.0), 0)
        self.assertEqual(self.sequence("c" + "q" * 40 + "c", lambda: 99.0), 0)
        self.assertEqual(self.sequence("s" * 40, lambda: 99.0), 0)
        self.assertEqual(self.sequence("q" * 40, lambda: 99.0), 0)

    def test_typing_is_not_a_clap(self):
        self.assertEqual(self.sequence("cqqqqqc", lambda: 0.1), 0)

    def test_claps_after_typing_stops_still_work(self):
        # typing until t=100.5, then hands off the keyboard: the claps at ~100.8 s and ~101.3 s count
        self.assertEqual(self.sequence("qqqqqqqqqcqqqqqc", lambda: 0.1 if self.clock[0] < 100.5 else 5.0), 1)
        # ...but a clap during the typing window is discarded, so only one clap is left and nothing triggers
        self.clock[0] = 100.0
        self.assertEqual(self.sequence("qqqqcqqqqqc", lambda: 0.1 if self.clock[0] < 100.5 else 5.0), 0)

    def test_works_when_the_macos_call_is_unavailable(self):
        self.j._cg_lib = False
        self.assertEqual(self.sequence("cqqqqqc"), 1)


class VoiceLoopTests(AudioTestCase):
    """Runs voice_mode against fake hardware to check the control flow."""

    def run_loop(self, reads=40, record_results=(), typed=(), paused=False, click=False, transcripts=("hello",)):
        j = self.j
        outer = self
        outer.flushes, outer.handled, outer.records = 0, [], []
        results, texts = list(record_results), list(transcripts)

        class FakeEars:
            def __init__(s): s.np, s.noise, s.threshold, s.last_peak, s.last_thr, s.n = np, 100.0, 300.0, 0.0, 0.0, 0
            def read(s):
                s.n += 1
                if s.n > reads:
                    raise KeyboardInterrupt
                return np.zeros(1280, dtype=np.int16)
            def flush(s): outer.flushes += 1
            def calibrate(s): pass
            def record(s, timeout=None, strict=False):
                outer.records.append((timeout, strict))
                return results.pop(0) if results else None

        class FakeWake:
            def __init__(s, **k): pass
            def predict(s, c): return {"hey_jarvis": 0.0}
            def reset(s): pass
        owm = types.ModuleType("openwakeword.model")
        owm.Model = FakeWake
        j.Ears, j.make_brain, j.load_whisper = FakeEars, lambda: None, lambda: None
        j.transcribe = lambda w, a: texts.pop(0) if texts else ""
        j.handle = lambda brain, text: outer.handled.append(text) or True
        j.ui_typed = queue.Queue()
        for t in typed:
            j.ui_typed.put(t)
        if click:
            j.ui_trigger.set()
        (j.PAUSED.set if paused else j.PAUSED.clear)()
        with mock.patch.dict(sys.modules, {"openwakeword": types.ModuleType("openwakeword"), "openwakeword.model": owm}), \
                mock.patch.object(j.subprocess, "Popen"):
            try:
                j.voice_mode(use_ui=False)
            except KeyboardInterrupt:
                pass

    def test_idle_listening_never_restarts_the_microphone(self):
        self.run_loop(reads=60)
        self.assertEqual(self.flushes, 1)                       # just the one at startup

    def test_a_click_starts_a_conversation_with_followups_and_no_wake_word(self):
        self.run_loop(click=True, record_results=[np.ones(8000, dtype=np.int16)] * 2, transcripts=("what time is it in london", "and in new york"))
        self.assertEqual(self.handled, ["what time is it in london", "and in new york"])
        self.assertEqual(self.records[0], (None, False))
        self.assertEqual(self.records[1], (self.j.FOLLOWUP_SECONDS, True))        # strict follow-up window
        self.assertEqual(len(self.records), 3)                                    # third follow-up got silence: back to idle

    def test_followups_can_be_disabled(self):
        self.j.FOLLOWUP = False
        self.run_loop(click=True, record_results=[np.ones(8000, dtype=np.int16)], transcripts=("hello",))
        self.assertEqual(self.handled, ["hello"])
        self.assertEqual(len(self.records), 1)

    def test_text_typed_in_the_hud_is_handled_without_the_microphone(self):
        self.run_loop(typed=["open the calendar"])
        self.assertEqual(self.handled, ["open the calendar"])
        self.assertEqual(self.records, [])

    def test_pausing_ignores_wake_words_and_typing(self):
        self.run_loop(paused=True, click=True, typed=["hello"])
        self.assertEqual(self.handled, [])
        self.assertEqual(self.records, [])

    def test_unintelligible_audio_is_dropped(self):
        self.run_loop(click=True, record_results=[np.ones(8000, dtype=np.int16)], transcripts=("",))
        self.assertEqual(self.handled, [])


if __name__ == "__main__":
    unittest.main()
