import io
import os
import stat
import subprocess
import sys
import threading
import types
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace as NS
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
from helpers import ROOT, JarvisTestCase, read_file, write_file  # noqa: E402

SCRIPT = os.path.join(ROOT, "jarvis.py")


def run_cli(*args, env=None):
    home = __import__("tempfile").mkdtemp()
    clean = {k: v for k, v in os.environ.items() if not (k.startswith("JARVIS_") or k.endswith("_API_KEY"))}
    clean.update(HOME=home, **(env or {}))
    return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True, env=clean, timeout=30)


class CommandLineTests(unittest.TestCase):
    def test_help_lists_every_mode(self):
        out = run_cli("--help").stdout
        for flag in ("--text", "--check", "--no-ui", "--mictest", "--claptest", "--voices", "--shortcuts", "--install-voice",
                     "--install-kokoro", "--install-launcher", "--menubar"):
            self.assertIn(flag, out)

    def test_shortcuts_command_creates_defaults_without_any_api_key(self):
        r = run_cli("--shortcuts")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("work mode", r.stdout)
        self.assertIn("shortcuts.json", r.stdout)

    def test_unknown_backend_is_explained(self):
        r = run_cli("--text", env={"JARVIS_BACKEND": "bogus"})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Unknown JARVIS_BACKEND", r.stdout + r.stderr)

    def test_a_backend_without_its_key_says_what_to_set(self):
        r = run_cli("--text", env={"JARVIS_BACKEND": "groq"})
        self.assertIn("GROQ_API_KEY", r.stdout + r.stderr)


class EnvFileTests(JarvisTestCase):
    def test_keys_are_read_from_the_dot_jarvis_env_file_without_overriding_the_shell(self):
        home = os.environ["HOME"]
        os.makedirs(os.path.join(home, ".jarvis"))
        write_file(os.path.join(home, ".jarvis", "env"), '# comment\nexport GROQ_API_KEY="gsk_file"\nOTHER=abc\nSHELL_WINS=file\n')
        os.environ["SHELL_WINS"] = "shell"
        import importlib.util
        spec = importlib.util.spec_from_file_location("j2", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(os.environ["GROQ_API_KEY"], "gsk_file")
        self.assertEqual(os.environ["OTHER"], "abc")
        self.assertEqual(os.environ["SHELL_WINS"], "shell")
        self.assertEqual(module.BACKEND, "groq")        # the key in the file picked the backend


class TextModeTests(JarvisTestCase):
    def test_typed_requests_go_through_the_same_pipeline_and_quit_exits(self):
        asked = []
        self.j.make_brain = lambda: NS(think=lambda text: asked.append(text) or f"Echo: {text}")
        lines = iter(["", "  open the pod bay doors  ", "quit", "never reached"])
        with mock.patch("builtins.input", lambda prompt="": next(lines)), redirect_stdout(io.StringIO()):
            self.j.text_mode()
        self.assertEqual(asked, ["open the pod bay doors"])
        self.assertEqual(self.spoken, ["Echo: open the pod bay doors"])

    def test_goodbye_ends_the_session(self):
        self.j.make_brain = lambda: NS(think=lambda text: self.fail("the brain must not be called"))
        lines = iter(["goodbye jarvis", "never reached"])
        with mock.patch("builtins.input", lambda prompt="": next(lines)), redirect_stdout(io.StringIO()):
            self.j.text_mode()
        self.assertEqual(self.spoken, ["Powering down. Goodbye."])


class LauncherTests(JarvisTestCase):
    def test_launcher_is_an_executable_script_pointing_at_this_install(self):
        os.makedirs(os.path.join(os.environ["HOME"], "Desktop"))
        with redirect_stdout(io.StringIO()):
            self.j.install_launcher()
        path = os.path.join(os.environ["HOME"], "Desktop", "Jarvis.command")
        text = read_file(path)
        self.assertTrue(text.startswith("#!/bin/zsh"))
        self.assertIn(sys.executable, text)
        self.assertIn(os.path.abspath(self.j.__file__), text)
        self.assertTrue(os.stat(path).st_mode & stat.S_IXUSR)

    def test_shortcuts_listing(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.j.shortcuts_cli()
        self.assertIn('"good morning"', out.getvalue())
        self.assertIn("work mode", out.getvalue())


class MenuBarTests(JarvisTestCase):
    def fake_rumps(self):
        apps = []

        class MenuItem:
            def __init__(s, title, callback=None):
                s.title, s.callback = title, callback

        class App:
            def __init__(s, name, title=None, quit_button="Quit"):
                s.title, s.menu, s.quit_button = title, [], quit_button
                apps.append(s)

            def run(s):
                s.ran = True
        return types.SimpleNamespace(App=App, MenuItem=MenuItem, quit_application=mock.Mock()), apps

    def test_explains_how_to_install_the_optional_dependency(self):
        with mock.patch.dict(sys.modules, {"rumps": None}):
            with self.assertRaises(SystemExit) as cm:
                self.j.menubar_mode()
        self.assertIn("pip install rumps", str(cm.exception))

    def test_menu_pauses_resumes_and_quits(self):
        fake, apps = self.fake_rumps()
        started = []
        self.j.voice_mode = lambda use_ui=True: started.append(use_ui)
        with mock.patch.dict(sys.modules, {"rumps": fake}):
            self.j.menubar_mode()
        threading.Event().wait(0.05)
        bar = apps[0]
        self.assertTrue(bar.ran)
        self.assertEqual(started, [True])
        self.assertEqual(os.environ["JARVIS_NO_BROWSER"], "1")
        pause = bar.menu[0]
        self.assertEqual(pause.title, "Pause listening")
        pause.callback(pause)
        self.assertTrue(self.j.PAUSED.is_set())
        self.assertEqual((pause.title, bar.title), ("Resume listening", "J\u00b7"))
        pause.callback(pause)
        self.assertFalse(self.j.PAUSED.is_set())
        self.assertEqual((pause.title, bar.title), ("Pause listening", "J"))
        bar.menu[-1].callback(None)
        fake.quit_application.assert_called_once()
        with mock.patch.object(self.j.subprocess, "Popen") as popen:
            bar.menu[1].callback(None)
        self.assertEqual(popen.call_args.args[0][0], "open")


if __name__ == "__main__":
    unittest.main()
