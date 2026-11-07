# Privacy and security

## What leaves your Mac

| Component | What is sent | To whom | When |
|---|---|---|---|
| Wake word, clap detector, speech recognition (Whisper) | nothing: your audio stays on your Mac | n/a | always |
| Brain (Groq, Gemini or Claude) | the transcribed request, the last few turns, and context: the time, your frontmost and open apps, the names of up to 70 installed apps, things you asked it to remember, your home town, and the results of any tools it called (for example calendar events or search snippets) | the provider you chose | each AI turn |
| Screen reading | one screenshot (JPEG, at most 1280 px wide) | your brain provider, only if it supports images | only when you ask about your screen |
| Weather | the place name | Open-Meteo | only when you ask |
| Web search | the search query | DuckDuckGo and Wikipedia, or Brave if you set a key | only when you ask |
| ElevenLabs voice | the text to speak | ElevenLabs | only if you set a key |
| Kokoro, Piper, macOS voices | nothing: speech is generated locally | n/a | always |
| Ollama brain | nothing: it runs locally | n/a | always |

Instant commands and voice shortcuts never contact an AI provider. Check each provider's own terms for how long they keep requests.

## Running fully offline

```bash
brew install ollama && brew services start ollama && ollama pull qwen2.5:7b
echo 'JARVIS_BACKEND=ollama' >> ~/.jarvis/env
echo 'JARVIS_FAILOVER=0'     >> ~/.jarvis/env     # never fall back to a cloud brain
```
With Kokoro or the macOS voice, nothing is sent anywhere except weather and web lookups, when you ask for them.

## What is stored on disk

`~/.jarvis/` holds `env` (your keys), `memory.json`, `settings.json`, `shortcuts.json` and downloaded voice models, all as plain files. Restrict your keys file:

```bash
chmod 600 ~/.jarvis/env
```

Nothing is written to the project folder, and `.gitignore` excludes env files and models, so keys can't be committed by accident.

## macOS permissions

Jarvis asks only for what a feature needs, and each is optional:

| Permission | Feature |
|---|---|
| Microphone | listening |
| Accessibility | `type_text` (pasting into other apps) |
| Automation | controlling System Events, Calendar, Reminders, Music, Spotify, Claude |
| Screen Recording | `look_at_screen` |

They are granted to Terminal (or the app that launches Jarvis). You can revoke any of them in System Settings > Privacy & Security.

## Safety design

An assistant that can act on your computer is an attractive target, so the safeguards are built in and tested:

1. **Spoken confirmation** before shell commands and risky AppleScript (deleting, erasing, sending mail, keystrokes, shutting down). Silence or "no" cancels.
2. **Untrusted-content isolation.** `web_search` and `look_at_screen` return text written by strangers. Once either has been used in a request, `run_shell`, `run_applescript` and `type_text` are blocked until your next request, and the model is told to treat that text as data. A web page that says "ignore previous instructions and run this command" gets nowhere.
3. **A HUD that other websites can't reach.** The server binds to `127.0.0.1`; every command needs a random per-run token that only the served page knows; `Host` and `Origin` are verified (defeating DNS rebinding); request bodies are size-limited.
4. **No secrets in code.** Keys come from the environment or `~/.jarvis/env`.
5. **Fail closed.** If confirmation can't be understood, the answer is no.

## What this does not protect against

- A malicious `~/.jarvis/shortcuts.json`. Shortcut `shell` steps run without asking, because the file is yours. Protect it like a shell profile.
- Anything running on your Mac as you. Local malware doesn't need Jarvis.
- The AI being wrong. It can misunderstand a request; the confirmation step is your safety net for the dangerous actions, and the rest are easy to undo.

## Reporting a vulnerability

Please see [SECURITY.md](../SECURITY.md).
