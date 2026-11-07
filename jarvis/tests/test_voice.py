import glob
import io
import os
import sys
import tempfile
import time
import types
import unittest
import wave
from contextlib import redirect_stdout
from types import SimpleNamespace as NS
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from helpers import JarvisTestCase, load_jarvis, write_file  # noqa: E402


class FakePopen:
    """Records every player process Jarvis starts (afplay / say) instead of making sound."""
    started = []

    def __init__(self, cmd, stdin=None, **kw):
        self.cmd, self.terminated, self.written = cmd, False, b""
        self.stdin = NS(write=self._write, close=lambda: None) if stdin else None
        FakePopen.started.append(self)

    def _write(self, data):
        self.written += data

    def wait(self):
        return 0

    def terminate(self):
        self.terminated = True


class VoiceTestCase(JarvisTestCase):
    def setUp(self):
        super().setUp()
        FakePopen.started = []
        patcher = mock.patch.object(self.j.subprocess, "Popen", FakePopen)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.events = []
        self.j.ui_emit = lambda state=None, **kw: self.events.append((state, kw))
        self.j._resolve_voice = lambda name: "Daniel (English (UK))"
        self.j.speak = type(self.j).__dict__.get("speak") or self.j.speak   # keep the real speak()
        self.j = load_jarvis(self.env)                                       # fresh module with the real speak()
        patcher2 = mock.patch.object(self.j.subprocess, "Popen", FakePopen)
        patcher2.start()
        self.addCleanup(patcher2.stop)
        self.j.ui_emit = lambda state=None, **kw: self.events.append((state, kw))
        self.j._resolve_voice = lambda name: "Daniel (English (UK))"

    def played(self):
        return [p.cmd[0] for p in FakePopen.started]

    def quiet_speak(self, text):
        with redirect_stdout(io.StringIO()):
            self.j.speak(text)


class MacVoiceTests(VoiceTestCase):
    def test_say_uses_the_full_british_voice_name_and_text_via_stdin(self):
        self.quiet_speak("Good evening.")
        p = FakePopen.started[0]
        self.assertEqual(p.cmd[:3], ["say", "-v", "Daniel (English (UK))"])
        self.assertEqual(p.cmd[-2:], ["-f", "-"])
        self.assertEqual(p.written, b"Good evening.")

    def test_voice_resolution_prefers_british_then_premium_then_enhanced(self):
        j = load_jarvis()
        sample = ["Daniel (English (UK)) en_GB    # Hello!", "Daniel (Enhanced)     en_GB    # Hello!", "Daniel (Premium)      en_GB    # Hello!",
                  "Daniel                fr_FR    # Bonjour", "Samantha              en_US    # Hello"]

        def pick(lines, want="Daniel"):
            with mock.patch.object(j.subprocess, "run", return_value=NS(stdout="\n".join(lines) + "\n", returncode=0)):
                return j._resolve_voice(want)
        self.assertEqual(pick(sample), "Daniel (Premium)")
        self.assertEqual(pick([s for s in sample if "Premium" not in s]), "Daniel (Enhanced)")
        self.assertEqual(pick([s for s in sample if "Premium" not in s and "Enhanced" not in s]), "Daniel (English (UK))")
        self.assertEqual(pick(sample, "Daniel (English (UK))"), "Daniel (English (UK))")      # exact names are respected
        self.assertEqual(pick(sample, "Samantha"), "Samantha")
        self.assertIsNone(pick(sample, "Nobody"))


class SpeakTests(VoiceTestCase):
    def test_spoken_text_is_short_but_the_screen_keeps_everything(self):
        long = "Sure, " + "the answer has several parts to it. " * 4 + "Final point here."
        self.quiet_speak(long)
        spoken = FakePopen.started[0].written.decode()
        self.assertLessEqual(len(spoken.split()), self.j.current_persona()[2])
        self.assertTrue(spoken.startswith("The answer"))
        speaking = [kw for state, kw in self.events if state == "speaking"][0]
        self.assertIn("Final point here.", speaking["text"])
        self.assertEqual(self.events[-1][0], "idle")

    def test_markdown_and_ellipses_are_never_read_aloud(self):
        self.quiet_speak("**Done**... opening `Safari`\u2026")
        text = FakePopen.started[0].written.decode()
        self.assertNotIn("*", text)
        self.assertNotIn("`", text)
        self.assertNotIn("...", text)
        self.assertNotIn("\u2026", text)

    def test_first_sound_timing_is_reported_once(self):
        self.j._timing["t0"] = time.time() - 1.5
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.j.speak("One.")
            self.j.speak("Two.")
        self.assertEqual(buf.getvalue().count("first sound"), 1)


class ElevenLabsTests(VoiceTestCase):
    def setUp(self):
        super().setUp()
        self.j.ELEVEN_KEY = "k"

    def fake_requests(self, status):
        seen = {}

        def post(url, params=None, headers=None, json=None, timeout=None):
            seen.update(url=url, headers=headers, json=json)
            return NS(status_code=status, content=b"ID3mp3", text='{"detail":{"code":"paid_plan_required"}}')
        return NS(post=post), seen

    def test_success_plays_the_mp3(self):
        fake, seen = self.fake_requests(200)
        with mock.patch.dict(sys.modules, {"requests": fake}):
            self.quiet_speak("Hello sir")
        self.assertEqual(self.played(), ["afplay"])
        self.assertTrue(seen["url"].endswith("/v1/text-to-speech/wDsJlOXPqcvIUKdLXjDs"))
        self.assertEqual(seen["headers"]["xi-api-key"], "k")
        self.assertEqual(seen["json"]["text"], "Hello sir")

    def test_402_falls_back_to_the_mac_voice_and_stops_retrying(self):
        fake, _ = self.fake_requests(402)
        with mock.patch.dict(sys.modules, {"requests": fake}):
            self.quiet_speak("Second")
            self.quiet_speak("Third")
        self.assertEqual(self.played(), ["say", "say"])
        self.assertFalse(self.j._eleven_on)

    def test_forcing_the_mac_voice(self):
        self.j.TTS_CHOICE = "mac"
        fake, seen = self.fake_requests(200)
        with mock.patch.dict(sys.modules, {"requests": fake}):
            self.quiet_speak("Hello")
        self.assertEqual(self.played(), ["say"])
        self.assertEqual(seen, {})


class PiperTests(VoiceTestCase):
    def setUp(self):
        super().setUp()
        self.model = os.path.join(tempfile.mkdtemp(), "jarvis.onnx")
        write_file(self.model, b"x", binary=True)
        self.j.PIPER_MODEL = self.model

    def test_piper_voice_is_used_when_installed(self):
        self.j._piper_synth = lambda text, path: write_file(path, b"RIFF" + b"0" * 2000, binary=True)
        self.quiet_speak("Hello sir")
        self.assertEqual(self.played(), ["afplay"])

    def test_piper_failure_falls_back_once(self):
        def boom(text, path):
            raise RuntimeError("no piper")
        self.j._piper_synth = boom
        self.quiet_speak("One")
        self.quiet_speak("Two")
        self.assertEqual(self.played(), ["say", "say"])
        self.assertTrue(self.j._piper_failed)

    def test_missing_model_goes_straight_to_the_mac_voice(self):
        self.j.PIPER_MODEL = "/nonexistent.onnx"
        self.quiet_speak("Hello")
        self.assertEqual(self.played(), ["say"])


class KokoroTests(VoiceTestCase):
    def setUp(self):
        super().setUp()
        self.made, self.fail_on = [], None
        outer = self

        class FakeKokoro:
            def __init__(s, model, voices):
                assert model.endswith("kokoro-v1.0.onnx") and voices.endswith("voices-v1.0.bin")

            def create(s, text, voice, speed, lang):
                assert lang == "en-gb" and voice == "bm_george" and speed > 1.0
                if outer.fail_on and outer.fail_on in text:
                    raise RuntimeError("engine exploded")
                outer.made.append(text)
                return np.sin(np.arange(24000) / 20).astype(np.float32) * 0.5, 24000
        self.enterContext(mock.patch.dict(sys.modules, {"kokoro_onnx": NS(Kokoro=FakeKokoro)}))
        os.makedirs(self.j.KOKORO_DIR)
        write_file(self.j.KOKORO_MODEL, b"x", binary=True)
        write_file(self.j.KOKORO_VOICES, b"x", binary=True)

    def test_speaks_sentence_by_sentence_with_valid_audio_and_cleans_up(self):
        before = set(glob.glob(tempfile.gettempdir() + "/*.wav"))
        self.quiet_speak("Right away. Safari is open. Anything else?")
        self.assertEqual(self.made, ["Right away.", "Safari is open.", "Anything else?"])
        self.assertEqual(self.played(), ["afplay"] * 3)
        self.assertEqual(set(glob.glob(tempfile.gettempdir() + "/*.wav")), before)

    def test_wav_files_are_mono_16bit(self):
        self.j._kokoro = None
        path = os.path.join(tempfile.mkdtemp(), "x.wav")
        self.j._kokoro_wav("Hello.", path)
        with wave.open(path) as w:
            self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()), (1, 2, 24000, 24000))

    def test_failure_on_the_first_sentence_falls_back_and_stops_retrying(self):
        self.fail_on = "Right"
        self.quiet_speak("Right away. Done.")
        self.assertEqual(self.played(), ["say"])
        self.assertTrue(self.j._kokoro_failed)
        self.quiet_speak("Again.")
        self.assertEqual(self.played(), ["say", "say"])

    def test_failure_mid_reply_does_not_repeat_what_was_already_spoken(self):
        self.fail_on = "second"
        self.quiet_speak("First one. The second one. Third.")
        self.assertEqual(self.played(), ["afplay"])

    def test_forcing_another_engine(self):
        self.j.TTS_CHOICE = "mac"
        self.quiet_speak("Hello.")
        self.assertEqual((self.made, self.played()), ([], ["say"]))

    def test_installer_downloads_both_files_then_plays_a_sample(self):
        shutil_rmtree = __import__("shutil").rmtree
        shutil_rmtree(self.j.KOKORO_DIR)
        urls = []

        def retrieve(url, dest, hook=None):
            urls.append(url)
            write_file(dest, b"z" * 2_000_000, binary=True)
            if hook:
                hook(1, 2_000_000, 2_000_000)
        with mock.patch("urllib.request.urlretrieve", retrieve), mock.patch.object(self.j.subprocess, "run") as run, \
                redirect_stdout(io.StringIO()):
            self.j.install_kokoro()
        self.assertEqual([u.rsplit("/", 1)[1] for u in urls], ["voices-v1.0.bin", "kokoro-v1.0.onnx"])
        self.assertTrue(all("model-files-v1.0" in u for u in urls))
        self.assertEqual(run.call_args.args[0][0], "afplay")

    def test_installer_explains_a_missing_engine(self):
        with mock.patch.dict(sys.modules, {"kokoro_onnx": None}):
            with self.assertRaises(SystemExit) as cm:
                self.j.install_kokoro()
        self.assertIn("pip install kokoro-onnx", str(cm.exception))


class InterruptTests(VoiceTestCase):
    def test_stop_speaking_terminates_players_and_blocks_new_ones(self):
        proc = self.j._spawn(["afplay", "x.wav"])
        self.j.stop_speaking()
        self.assertTrue(proc.terminated)
        before = len(FakePopen.started)
        self.j._play(["afplay", "y.wav"])
        self.assertEqual(len(FakePopen.started), before)         # nothing new plays after an interrupt
        self.j._STOP.clear()
        self.j._play(["afplay", "y.wav"])
        self.assertEqual(len(FakePopen.started), before + 1)

    def test_a_double_clap_while_talking_stops_the_speech(self):
        j = self.j
        j._seconds_since_input = lambda: 99.0
        clock = [100.0]
        rng = np.random.default_rng(3)
        kinds = list("cqqqqqc") + ["q"] * 400

        class Ears:
            np, noise = np, 100.0
            def read(s):
                clock[0] += 0.08
                x = rng.normal(0, 80, 1280)
                if kinds and kinds.pop(0) == "c":
                    x[400:600] += rng.normal(0, 1, 200) * 14000 * np.exp(-np.arange(200) / 50)
                time.sleep(0.001)
                return np.clip(x, -32768, 32767).astype(np.int16)
        with mock.patch.object(j.time, "time", lambda: clock[0]):
            with redirect_stdout(io.StringIO()), j.BargeIn(Ears()):
                deadline = time.time() + 3
                while not j._STOP.is_set() and time.time() < deadline:
                    time.sleep(0.005)
        self.assertTrue(j._STOP.is_set())

    def test_barge_in_can_be_disabled(self):
        self.j.BARGE_IN = False
        with self.j.BargeIn(NS(read=lambda: self.fail("must not listen"))) as b:
            self.assertIsNone(b._t)


class StreamSpeakerTests(VoiceTestCase):
    def setUp(self):
        super().setUp()
        self.chunks = []
        self.j._speak_chunk = self.chunks.append

    def feed(self, pieces, limit=None):
        sp = self.j.StreamSpeaker(limit=limit)
        for p in pieces:
            sp.feed(p)
        return sp, sp.finish()

    def test_speaks_each_sentence_as_soon_as_it_is_complete(self):
        sp = self.j.StreamSpeaker(limit=0)
        sp.feed("Right away. The cal")
        time.sleep(0.05)
        self.assertEqual(self.chunks, ["Right away."])           # spoken while the rest is still being written
        sp.feed("endar is clear. Anything else?")
        self.assertEqual(sp.finish(), "Right away. The calendar is clear. Anything else?")
        self.assertEqual(self.chunks, ["Right away.", "The calendar is clear.", "Anything else?"])

    def test_decimals_and_abbreviations_do_not_split_sentences(self):
        self.feed(["It is 3.5 degrees. ", "Wrap up."], limit=0)
        self.assertEqual(self.chunks, ["It is 3.5 degrees.", "Wrap up."])

    def test_filler_opener_markdown_and_ellipses_are_cleaned(self):
        self.feed(["Sure, **it's** done... ", "Next."], limit=0)
        self.assertEqual(self.chunks, ["It's done.", "Next."])

    def test_word_budget_keeps_the_rest_on_screen_only(self):
        sp, full = self.feed(["One two three four five. ", "Six seven eight nine ten. ", "Eleven twelve."], limit=8)
        self.assertEqual(self.chunks, ["One two three four five."])
        self.assertIn("Eleven twelve.", full)

    def test_interrupt_swallows_the_rest(self):
        sp = self.j.StreamSpeaker(limit=0)
        self.j._STOP.set()
        sp.feed("Never heard. Nor this.")
        sp.finish()
        self.assertEqual(self.chunks, [])

    def test_the_hud_is_told_when_speech_starts(self):
        self.feed(["Hello there."], limit=0)
        self.assertIn(("speaking", {"text": "Hello there."}), self.events)

    def test_remainder_without_punctuation_is_spoken_at_the_end(self):
        self.feed(["Open now"], limit=0)
        self.assertEqual(self.chunks, ["Open now"])


class HandleTests(VoiceTestCase):
    def setUp(self):
        super().setUp()
        self.chunks, self.said = [], []
        self.j._speak_chunk = self.chunks.append
        self.j.speak = self.said.append

    def run_handle(self, brain, text="hello"):
        with redirect_stdout(io.StringIO()):
            return self.j.handle(brain, text)

    def test_streamed_replies_are_spoken_by_the_stream_not_twice(self):
        def think(text, on_text=None):
            for piece in ("Right ", "away. ", "Safari is open."):
                on_text(piece)
            return "Right away. Safari is open."
        self.assertTrue(self.run_handle(NS(think=think)))
        self.assertEqual(self.chunks, ["Right away.", "Safari is open."])
        self.assertEqual(self.said, [])
        self.assertIn(("idle", {"who": "jarvis", "text": "Right away. Safari is open."}), self.events)

    def test_non_streaming_brains_still_work(self):
        self.run_handle(NS(think=lambda text: "Plain answer."))
        self.assertEqual(self.said, ["Plain answer."])

    def test_empty_reply_says_done(self):
        self.run_handle(NS(think=lambda text, on_text=None: ""))
        self.assertEqual(self.said, ["Done."])

    def test_streaming_off_uses_normal_speech(self):
        self.j.STREAM = False
        self.run_handle(NS(think=lambda text, on_text=None: "Whole reply."))
        self.assertEqual(self.said, ["Whole reply."])
        self.assertEqual(self.chunks, [])

    def test_voice_exit_and_cancel_phrases(self):
        brain = NS(think=lambda text: self.fail("brain must not be called"))
        self.assertFalse(self.run_handle(brain, "Goodbye Jarvis"))
        self.assertEqual(self.said[-1], "Powering down. Goodbye.")
        self.said.clear()
        self.assertTrue(self.run_handle(brain, "never mind"))
        self.assertEqual(self.said, [])

    def test_friendly_spoken_errors(self):
        cases = {"HTTP 401: bad key": "API key was rejected", "HTTP 429: slow": "usage limit", "model 'x' not installed": "ollama pull",
                 "HTTP 404 nope": "model name", "ConnectionError: down": "reach my brain", "boom": "Something went wrong"}
        for error, expected in cases.items():
            def think(text, on_text=None, _e=error):
                raise RuntimeError(_e)
            self.said.clear()
            self.run_handle(NS(think=think))
            self.assertIn(expected, self.said[-1], error)

    def test_a_new_request_clears_an_earlier_interrupt(self):
        self.j._STOP.set()
        self.run_handle(NS(think=lambda text: "ok"))
        self.assertFalse(self.j._STOP.is_set())


if __name__ == "__main__":
    unittest.main()
