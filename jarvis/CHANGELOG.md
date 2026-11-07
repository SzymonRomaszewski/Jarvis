# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-01

First public release.

### Added
- Voice assistant loop: wake word ("Hey Jarvis"), double-clap trigger, click or typed requests from the HUD, local speech recognition (Whisper on MLX), follow-up listening without the wake word.
- On-screen HUD with live state, microphone level, conversation log and a typing box, served on loopback with a per-run token.
- Brains: Groq, Gemini, Claude and Ollama behind one interface, with token streaming, automatic failover and per-request tool selection to keep requests small.
- Tools: apps, volume, timers, screenshots, typing into other apps, Calendar and Reminders, weather (Open-Meteo), web search, Spotify and Apple Music, macOS Shortcuts, screen vision, memory.
- Instant local commands that skip the AI (time, date, volume, timers, open and quit apps, music transport).
- Voice shortcuts (`~/.jarvis/shortcuts.json`) that can also be created by voice.
- Voices: Kokoro (free, natural British), Piper, ElevenLabs and the best installed macOS voice, with automatic fallback and sentence-by-sentence streaming.
- Switchable personalities and a configurable form of address.
- Interrupt a reply with a double clap.
- Safety: spoken confirmation for risky actions, untrusted-content isolation, HUD origin and token checks.
- Tooling: `--check`, `--mictest`, `--claptest`, `--voices`, `--shortcuts`, installers, a double-click launcher and an experimental menu-bar mode.
- 170+ tests, a static name checker, a generated tool reference and CI.
