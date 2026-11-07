"""End-to-end HUD test in a real headless browser. Skipped automatically if Playwright isn't installed."""
import os
import queue
import socket
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from helpers import load_jarvis  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
    HAVE_PLAYWRIGHT = True
except ImportError:  # pragma: no cover
    HAVE_PLAYWRIGHT = False


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@unittest.skipUnless(HAVE_PLAYWRIGHT, "playwright is not installed")
class HudBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = free_port()
        cls._saved = dict(os.environ)
        cls.j = load_jarvis({"JARVIS_PORT": str(cls.port), "JARVIS_NO_BROWSER": "1"})
        threading.Thread(target=cls.j.start_ui, daemon=True).start()
        time.sleep(0.4)
        cls.pw = sync_playwright().start()
        try:
            cls.browser = cls.pw.chromium.launch()
        except Exception as e:  # noqa: BLE001
            cls.pw.stop()
            raise unittest.SkipTest(f"no browser available: {e}")

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        cls.j.UI_ON = False
        os.environ.clear()
        os.environ.update(cls._saved)

    def setUp(self):
        self.j.ui_trigger.clear()
        self.j.ui_typed = queue.Queue()
        self.page = self.browser.new_page(viewport={"width": 1280, "height": 800})
        self.errors = []
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.page.goto(f"http://127.0.0.1:{self.port}/")
        self.page.wait_for_timeout(300)

    def tearDown(self):
        self.page.close()
        self.assertEqual(self.errors, [])

    def test_typed_request_reaches_jarvis(self):
        self.page.click("#typebox")
        self.page.keyboard.type("what's the weather in London")
        self.page.keyboard.press("Enter")
        self.assertEqual(self.j.ui_typed.get(timeout=2), "what's the weather in London")
        self.assertEqual(self.page.input_value("#typebox"), "")

    def test_space_in_the_box_is_just_a_space_but_space_elsewhere_starts_listening(self):
        self.page.click("#typebox")
        self.page.keyboard.type("a b")
        self.page.wait_for_timeout(300)
        self.assertFalse(self.j.ui_trigger.is_set())
        self.page.click("body", position={"x": 40, "y": 400})
        self.page.keyboard.press("Space")
        for _ in range(20):
            if self.j.ui_trigger.is_set():
                break
            self.page.wait_for_timeout(50)
        self.assertTrue(self.j.ui_trigger.is_set())

    def test_clicking_the_core_starts_listening(self):
        self.page.mouse.click(640, 360)
        for _ in range(20):
            if self.j.ui_trigger.is_set():
                break
            self.page.wait_for_timeout(50)
        self.assertTrue(self.j.ui_trigger.is_set())

    def test_state_changes_and_conversation_appear_live(self):
        self.j.ui_emit("thinking", who="you", text="open safari")
        self.page.wait_for_timeout(400)
        self.assertEqual(self.page.inner_text("#state"), "PROCESSING")
        self.assertIn("open safari", self.page.inner_text("#log"))
        self.j.ui_emit("speaking", who="jarvis", text="Safari is open.")
        self.page.wait_for_timeout(400)
        self.assertEqual(self.page.inner_text("#state"), "SPEAKING")
        self.assertIn("Safari is open.", self.page.inner_text("#log"))
        self.j.ui_emit("idle", info="BRAIN GROQ|TONE CLASSIC")
        self.page.wait_for_timeout(400)
        self.assertEqual(self.page.inner_text("#state"), "STANDING BY")
        self.assertIn("TONE CLASSIC", self.page.inner_text("#info"))


if __name__ == "__main__":
    unittest.main()
