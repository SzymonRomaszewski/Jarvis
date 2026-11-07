"""Shared test helpers: load a fresh copy of jarvis.py with an isolated HOME and no API keys."""
import importlib.util
import os
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_file(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def write_file(path, text, binary=False):
    with open(path, "wb" if binary else "w", **({} if binary else {"encoding": "utf-8"})) as f:
        f.write(text)


def load_jarvis(env=None):
    """Import jarvis.py as a new module so every test gets clean state and its own ~/.jarvis folder."""
    home = tempfile.mkdtemp(prefix="jarvis-test-home-")
    os.environ["HOME"] = home
    for key in list(os.environ):
        if key.startswith("JARVIS_") or key.endswith("_API_KEY"):
            del os.environ[key]
    os.environ.update(env or {})
    spec = importlib.util.spec_from_file_location("jarvis_under_test", os.path.join(ROOT, "jarvis.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class JarvisTestCase(unittest.TestCase):
    """Base class: self.j is a freshly imported jarvis module, and speech goes to self.spoken instead of speakers."""
    env = None

    def setUp(self):
        saved = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(saved)))
        self.j = load_jarvis(self.env)
        self.spoken = []
        self.j.speak = lambda text: self.spoken.append(text)


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status=200, payload=None, lines=None, headers=None, text=""):
        self.status_code = status
        self._payload = payload
        self._lines = lines or []
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)
