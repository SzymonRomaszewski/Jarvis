"""The documentation is part of the product: these tests fail if it drifts from the code."""
import json
import os
import re
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from helpers import ROOT, load_jarvis, read_file  # noqa: E402


def text_files():
    skip_dirs = {".git", "__pycache__", "assets", ".venv", "build", "dist"}
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.endswith(".egg-info")]
        for name in files:
            if name.endswith((".py", ".md", ".toml", ".json", ".yml", ".yaml", ".txt", ".example", ".cfg")) or name in ("LICENSE", "Makefile", ".gitignore"):
                yield os.path.join(folder, name)


class DocsInSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = read_file(os.path.join(ROOT, "jarvis.py"))
        cls.readme = read_file(os.path.join(ROOT, "README.md"))
        cls.config_doc = read_file(os.path.join(ROOT, "docs", "configuration.md"))
        cls.j = load_jarvis()

    def test_every_setting_is_documented(self):
        names = set(re.findall(r"(?<![A-Za-z_])(JARVIS_[A-Z_]+[A-Z]|[A-Z]+_API_KEY|OLLAMA_URL)\b", self.source))
        self.assertGreater(len(names), 25)
        missing = sorted(n for n in names if n not in self.config_doc)
        self.assertEqual(missing, [], f"undocumented settings: {missing}")

    def test_every_command_line_flag_is_in_the_readme(self):
        flags = re.findall(r'add_argument\("(--[a-z-]+)"', self.source)
        self.assertGreater(len(flags), 10)
        missing = [f for f in flags if f not in self.readme]
        self.assertEqual(missing, [], f"flags missing from the README: {missing}")

    def test_tool_reference_matches_the_code(self):
        result = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "gen_tools_doc.py"), "--check"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_versions_agree(self):
        pyproject = read_file(os.path.join(ROOT, "pyproject.toml"))
        self.assertIn(f'version = "{self.j.__version__}"', pyproject)
        self.assertIn(f"## [{self.j.__version__}]", read_file(os.path.join(ROOT, "CHANGELOG.md")))

    def test_relative_links_in_markdown_point_at_real_files(self):
        broken = []
        for path in text_files():
            if not path.endswith(".md"):
                continue
            for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", read_file(path)):
                if re.match(r"^(https?:|mailto:)", target):
                    continue
                resolved = os.path.normpath(os.path.join(os.path.dirname(path), target))
                if not os.path.exists(resolved):
                    broken.append(f"{os.path.relpath(path, ROOT)} -> {target}")
        self.assertEqual(broken, [])

    def test_readme_images_exist(self):
        for src in re.findall(r'src="(assets/[^"]+)"', self.readme):
            self.assertTrue(os.path.exists(os.path.join(ROOT, src)), src)


class ExamplesTests(unittest.TestCase):
    def setUp(self):
        self.j = load_jarvis()

    def test_example_shortcuts_only_use_real_tools(self):
        shortcuts = json.loads(read_file(os.path.join(ROOT, "examples", "shortcuts.json")))
        for phrase, spec in shortcuts.items():
            for step in spec.get("do", []):
                if "shell" not in step:
                    self.assertIn(step["tool"], self.j.TOOL_FNS, f"{phrase}: {step}")
            self.assertTrue(spec.get("do") or spec.get("say") or spec.get("ask"), phrase)
        self.assertEqual(self.j.match_shortcut("jarvis end of day please", shortcuts), "end of day")

    def test_example_settings_use_known_keys_and_personalities(self):
        settings = json.loads(read_file(os.path.join(ROOT, "examples", "settings.json")))
        self.assertLessEqual(set(settings), {"personality", "address_as", "location", "custom"})
        self.assertIn(settings["personality"], self.j.PERSONAS)

    def test_example_env_only_sets_real_variables(self):
        source = read_file(os.path.join(ROOT, "jarvis.py"))
        for line in read_file(os.path.join(ROOT, "examples", "env.example")).splitlines():
            m = re.match(r"^#?\s*([A-Z][A-Z_]+)=", line)
            if m:
                self.assertIn(m.group(1), source, line)


class RepoHygieneTests(unittest.TestCase):
    SECRET_PATTERNS = [r"gsk_[A-Za-z0-9]{20,}", r"sk-ant-[A-Za-z0-9_-]{20,}", r"AIza[0-9A-Za-z_-]{30,}", r"xi-api-key:\s*[A-Za-z0-9]{20,}",
                       r"-----BEGIN [A-Z ]*PRIVATE KEY-----"]
    KNOWN_LEAKED = ["gsk_nl8I", "9cd4b73f"]            # keys that were pasted into chats while building this: must never appear

    def test_no_secrets_anywhere(self):
        for path in text_files():
            content = read_file(path)
            for pattern in self.SECRET_PATTERNS:
                self.assertIsNone(re.search(pattern, content), f"possible secret in {path}: {pattern}")
            for leaked in self.KNOWN_LEAKED:
                if not path.endswith("test_docs.py"):
                    self.assertNotIn(leaked, content, path)

    def test_gitignore_protects_keys_and_models(self):
        ignore = read_file(os.path.join(ROOT, ".gitignore"))
        for entry in (".env", ".jarvis/", "*.onnx", "*.bin", "__pycache__/"):
            self.assertIn(entry, ignore)

    def test_licence_and_policies_exist(self):
        for name in ("LICENSE", "SECURITY.md", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "CHANGELOG.md", "requirements.txt"):
            self.assertTrue(os.path.exists(os.path.join(ROOT, name)), name)

    def test_the_script_is_directly_runnable(self):
        self.assertTrue(read_file(os.path.join(ROOT, "jarvis.py")).startswith("#!/usr/bin/env python3"))


if __name__ == "__main__":
    unittest.main()
