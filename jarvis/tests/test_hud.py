import http.client
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from helpers import ROOT, load_jarvis, read_file, write_file  # noqa: E402


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class HudServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = free_port()
        cls._saved = dict(os.environ)
        cls.j = load_jarvis({"JARVIS_PORT": str(cls.port), "JARVIS_NO_BROWSER": "1"})
        threading.Thread(target=cls.j.start_ui, daemon=True).start()
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", cls.port), 0.2).close()
                break
            except OSError:
                time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.j.UI_ON = False
        os.environ.clear()
        os.environ.update(cls._saved)

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, data

    def auth(self, **extra):
        return {"X-Jarvis-Token": self.j.UI_TOKEN, "Content-Type": "application/json", **extra}

    def setUp(self):
        self.j.ui_trigger.clear()
        while not self.j.ui_typed.empty():
            self.j.ui_typed.get_nowait()

    # ---- the page ----
    def test_page_is_served_with_the_session_token_filled_in(self):
        status, body = self.request("GET", "/")
        page = body.decode()
        self.assertEqual(status, 200)
        self.assertIn("J.A.R.V.I.S", page)
        self.assertIn("typebox", page)
        self.assertIn(self.j.UI_TOKEN, page)
        self.assertNotIn("__JARVIS_TOKEN__", page)

    def test_each_run_has_a_long_random_token(self):
        self.assertGreaterEqual(len(self.j.UI_TOKEN), 20)
        self.assertNotEqual(self.j.UI_TOKEN, load_jarvis().UI_TOKEN)

    def test_unknown_paths_404(self):
        self.assertEqual(self.request("GET", "/nope")[0], 404)
        self.assertEqual(self.request("POST", "/nope", "{}", self.auth())[0], 404)

    # ---- security: other websites must not be able to command Jarvis ----
    def test_requests_for_other_hostnames_are_refused(self):
        # DNS-rebinding: an attacker's domain that resolves to 127.0.0.1 still sends its own Host header
        self.assertEqual(self.request("GET", "/", headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(self.request("GET", "/events", headers={"Host": "evil.example:8765"})[0], 403)

    def test_commands_without_the_token_are_refused(self):
        for path in ("/listen", "/say"):
            self.assertEqual(self.request("POST", path, '{"text": "open terminal"}')[0], 403)
            self.assertEqual(self.request("POST", path, '{"text": "x"}', {"Content-Type": "text/plain"})[0], 403)   # "simple" cross-site post
            self.assertEqual(self.request("POST", path, '{"text": "x"}', {"X-Jarvis-Token": "guess"})[0], 403)
        self.assertFalse(self.j.ui_trigger.is_set())
        self.assertTrue(self.j.ui_typed.empty())

    def test_commands_from_another_origin_are_refused_even_with_the_token(self):
        status, _ = self.request("POST", "/say", '{"text": "x"}', self.auth(Origin="http://evil.example"))
        self.assertEqual(status, 403)
        self.assertTrue(self.j.ui_typed.empty())

    def test_own_origin_with_token_is_accepted(self):
        status, _ = self.request("POST", "/listen", "{}", self.auth(Origin=f"http://127.0.0.1:{self.port}"))
        self.assertEqual(status, 200)
        self.assertTrue(self.j.ui_trigger.is_set())

    # ---- typing into the HUD ----
    def test_typed_text_is_queued(self):
        self.assertEqual(self.request("POST", "/say", json.dumps({"text": "  what's the weather  "}), self.auth())[0], 200)
        self.assertEqual(self.j.ui_typed.get_nowait(), "what's the weather")

    def test_empty_and_malformed_typed_text_is_ignored_and_long_text_is_cut(self):
        self.request("POST", "/say", json.dumps({"text": "   "}), self.auth())
        self.request("POST", "/say", "not json", self.auth())
        self.assertTrue(self.j.ui_typed.empty())
        self.request("POST", "/say", json.dumps({"text": "a" * 900}), self.auth())
        self.assertEqual(len(self.j.ui_typed.get_nowait()), 500)

    # ---- live events ----
    def test_events_stream_state_changes(self):
        sock = socket.create_connection(("127.0.0.1", self.port), 3)
        sock.sendall(f"GET /events HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\n\r\n".encode())
        f = sock.makefile("rb")
        while f.readline().strip():          # skip the response headers
            pass
        first = json.loads(f.readline().decode().split("data: ", 1)[1])
        self.assertIn("state", first)
        self.j.ui_emit("listening", text="Listening...")
        while True:
            line = f.readline().decode()
            if line.startswith("data: "):
                msg = json.loads(line[6:])
                if msg.get("state") == "listening":
                    break
        self.assertEqual(msg["text"], "Listening...")
        sock.close()

    def test_nothing_is_sent_when_the_hud_is_off(self):
        j = load_jarvis()
        j.ui_emit("speaking", text="secret")
        self.assertEqual(j._ui_last, {"state": "boot"})


class HudPageTests(unittest.TestCase):
    def setUp(self):
        self.j = load_jarvis()
        self.html = re.search(r"UI_HTML = r'''(.*?)'''", read_file(os.path.join(ROOT, "jarvis.py")), re.S).group(1)

    def test_typing_in_the_box_does_not_trigger_space_or_f_shortcuts(self):
        self.assertIn("e.stopPropagation()", self.html)

    def test_every_state_has_a_colour(self):
        for state in ("boot", "idle", "listening", "thinking", "speaking", "offline"):
            self.assertRegex(self.html, rf"\b{state}:\s*{{ h:")

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_javascript_is_valid(self):
        js = re.search(r"<script>(.*)</script>", self.html, re.S).group(1)
        path = os.path.join(tempfile.mkdtemp(), "hud.js")
        write_file(path, js)
        result = subprocess.run(["node", "--check", path], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
