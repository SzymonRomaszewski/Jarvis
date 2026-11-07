# Contributing

Thanks for helping make Jarvis better. Bug reports, ideas, docs fixes and pull requests are all welcome.

## Set up

```bash
git clone https://github.com/YOUR-USERNAME/jarvis.git && cd jarvis
python3.12 -m venv .venv && source .venv/bin/activate
pip install numpy requests          # everything the tests need
python -m unittest discover -s tests -v
```

You do not need a Mac, a microphone or any API key to run the tests: every piece of hardware and network access is replaced by a fake. To run the assistant itself you need an Apple Silicon Mac; see the README.

Optional, for the real-browser HUD test and the screenshot script: `pip install playwright pillow && playwright install chromium`.

## Before you open a pull request

```bash
python -m unittest discover -s tests          # all tests pass
python scripts/check_names.py jarvis.py       # no undefined names or unused imports
python scripts/gen_tools_doc.py               # refresh docs/tools.md if you changed a tool
```

A docs test also checks that every setting (`JARVIS_*`) and every command-line flag is documented, so new ones need a line in `docs/configuration.md` or the README.

## Guidelines

- **Add a test with every behaviour change.** Look at the neighbouring tests for the style: fake the hardware, assert on behaviour.
- **Keep I/O in small functions** so tests can replace them (`_osascript`, `_http_json`, `subprocess` calls, the microphone).
- **Fail soft.** One failing turn, tool or voice engine must never stop the assistant. Return a short message the AI can relay.
- **Think about safety for any tool that acts on the Mac.** If it reads outside content, add it to `UNTRUSTED_TOOLS`; if it can change things, consider `confirm()` and `SENSITIVE_TOOLS`. See `docs/architecture.md`.
- **Stay free-first.** New features should work without a paid key, or degrade gracefully.
- **Style:** plain Python 3.12, type hints where they help, comments that explain why, UK English in user-facing text.

## Adding a tool

See "Adding a tool" in [docs/architecture.md](docs/architecture.md). It is a function, one registry line and a test.

## Reporting bugs

Please include the output of `python jarvis.py --check`, your macOS version, which brain you use, and the Terminal lines around the problem. Never paste API keys. For security problems, see [SECURITY.md](SECURITY.md).
