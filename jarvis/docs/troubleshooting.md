# Troubleshooting

Start with the self-test, it checks the brain, voice, microphone, Whisper and the wake word and tells you which failed:

```bash
python jarvis.py --check
```

## Setup

**`command not found: python` or `pip`**
The virtual environment isn't active in this Terminal window. Run `source ~/jarvis-env/bin/activate` (the prompt should start with `(jarvis-env)`), or call it directly: `~/jarvis-env/bin/python jarvis.py`.

**"My API key was rejected" / `HTTP 401`**
Groq keys start with `gsk_`. Create one at [console.groq.com/keys](https://console.groq.com/keys). Check what Jarvis sees without printing the key: `echo ${GROQ_API_KEY:0:4}` should print `gsk_`. Prefer putting it in `~/.jarvis/env` so it works in every window.

**"I've hit the free usage limit"** (`HTTP 429`)
Free tiers have per-minute limits. Jarvis retries and fails over to any other brain you have a key for. Adding a free Gemini key as a backup is the easiest fix.

**It uses Ollama but I never set it up**
With no cloud key set, Jarvis falls back to Ollama. Add a key to `~/.jarvis/env` or install Ollama.

## Hearing you

Run `python jarvis.py --mictest`. It shows a live level meter, tells you whether your voice is loud enough, and runs a real transcription.

**It hears "Hey Jarvis" but then "can't hear" me**
Your voice isn't reaching the required level. Raise Input volume in System Settings > Sound, or lower the bar: `JARVIS_MIC_LEVEL=1000` (or `0` for automatic), or boost the signal: `JARVIS_MIC_GAIN=2`.

**The orange microphone indicator flickers**
Another app is using the microphone (dictation tools such as Wispr Flow, voice modes in other apps, Zoom). Quit it while testing. Click Control Center in the menu bar to see which app has the microphone. If an iPhone is nearby, turn off Continuity Camera. Bluetooth headphones switch to a low-quality mode when the microphone is used; try the built-in speakers.

**Typing makes Jarvis think I clapped**
Claps are ignored within 0.6 s of keyboard or mouse use. Make that stricter with `JARVIS_CLAP_TYPING_GUARD=1.0` or `JARVIS_CLAP_SENS=0.6`, or turn claps off with `JARVIS_CLAPS=0`. Tune it with `python jarvis.py --claptest`.

**The wake word fires by accident or doesn't fire**
Adjust `JARVIS_WAKE`: raise it toward 0.7 for fewer false triggers, lower it toward 0.35 if it misses you.

**Jarvis answers things I say to other people**
Switch off the follow-up window: `JARVIS_FOLLOWUP=0`.

## Voice

**It doesn't sound British**
Run `python jarvis.py --voices` to hear every British voice you have. Install better ones under System Settings > Accessibility > Spoken Content > System Voice > Manage Voices > English (UK). Or install Kokoro: `pip install kokoro-onnx && python jarvis.py --install-kokoro`.

**Kokoro fails**
`pip install -U kokoro-onnx`, then `python jarvis.py --check`. Jarvis falls back to the next voice automatically and prints why.

## Acting on your Mac

**"macOS blocked typing"**
System Settings > Privacy & Security > Accessibility, switch Terminal on, then quit and reopen Terminal.

**Calendar or Reminders say they're blocked** (`-1743`)
System Settings > Privacy & Security > Automation: allow Terminal to control Calendar and Reminders.

**Calendar is slow or misses repeating events**
AppleScript access to Calendar is slow with large calendars. `brew install ical-buddy` and Jarvis uses it automatically.

**"What's on my screen" fails**
Allow Terminal under Privacy & Security > Screen Recording. The brain also has to accept images: Groq with Llama 4 Scout (default), Gemini, Claude, or Ollama with `llava`.

## The HUD

**"Port 8765 is busy"**
Jarvis is already running somewhere. Quit it, or pick another port with `JARVIS_PORT=8766`.

**The HUD window doesn't open**
It opens as an app window in Google Chrome if installed, otherwise your default browser. Or open `http://127.0.0.1:8765/` yourself. Use `--no-ui` to skip it.

## It feels slow

Every turn prints timings:

```
(transcribed in 0.7s)       -> slow? try JARVIS_WHISPER=mlx-community/whisper-small-mlx
(brain answered in 0.9s)    -> slow? try JARVIS_GROQ_MODEL=openai/gpt-oss-20b
(first sound 1.8s after you stopped talking)
```
Also lower `JARVIS_SILENCE` (for example `0.7`) so Jarvis decides you've finished sooner.

## Starting over

Everything Jarvis stores is in `~/.jarvis/`. Delete a file there to reset it (for example `memory.json`), or the whole folder to start fresh.
