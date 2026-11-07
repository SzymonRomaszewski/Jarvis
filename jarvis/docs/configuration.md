# Configuration

Every setting is an environment variable. Put the ones you want in `~/.jarvis/env`, one `NAME=value` per line (an optional leading `export` is fine). Variables already set in your shell win over the file. See [`examples/env.example`](../examples/env.example).

Jarvis also keeps small files in `~/.jarvis/`:

| File | What it is |
|---|---|
| `env` | your keys and settings |
| `memory.json` | things you asked it to remember |
| `settings.json` | personality, what to call you, home location |
| `shortcuts.json` | your voice shortcuts ([example](../examples/shortcuts.json)) |
| `kokoro/` | the Kokoro voice model |
| `voice/` | the optional Piper voice |

## Keys

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | none | free brain ([console.groq.com/keys](https://console.groq.com/keys), starts with `gsk_`) |
| `GEMINI_API_KEY` | none | free brain ([aistudio.google.com/apikey](https://aistudio.google.com/apikey)) |
| `ANTHROPIC_API_KEY` | none | Claude as the brain (paid) |
| `ELEVENLABS_API_KEY` | none | ElevenLabs voice (Voice Library voices need a paid plan) |
| `BRAVE_API_KEY` | none | better web search; without it Jarvis uses DuckDuckGo answers and Wikipedia |

## Brain

| Variable | Default | Purpose |
|---|---|---|
| `JARVIS_BACKEND` | first key found (Claude, Groq, Gemini), else `ollama` | `claude`, `groq`, `gemini` or `ollama` |
| `JARVIS_MODEL` | `claude-sonnet-5-5` | Claude model |
| `JARVIS_GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model |
| `JARVIS_GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model |
| `JARVIS_OLLAMA_MODEL` | `qwen2.5:7b` | Ollama model |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `JARVIS_FAILOVER` | `1` | switch to a backup brain if the main one fails; `0` to disable |
| `JARVIS_STREAM` | `1` | speak the first sentence while the rest is still being written |
| `JARVIS_FASTPATH` | `1` | answer simple commands locally without the AI |
| `JARVIS_PERSONALITY` | saved choice, else `classic` | `classic`, `butler`, `friendly`, `sarcastic`, `minimal` |
| `JARVIS_MAX_WORDS` | by personality (15 to 40) | cap on spoken words (the screen shows everything); `0` disables |
| `JARVIS_UNITS` | `uk` | weather units: `uk` (C, mph), `metric` (C, km/h), `imperial` (F, mph) |
| `JARVIS_VISION` | `1` | allow "what's on my screen"; `0` switches it off |
| `JARVIS_VISION_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq vision model |
| `JARVIS_OLLAMA_VISION_MODEL` | `llava:7b` | Ollama vision model |

## Listening

| Variable | Default | Purpose |
|---|---|---|
| `JARVIS_MIC` | system default | microphone name or number |
| `JARVIS_MIC_LEVEL` | `1700` | fixed loudness your speech must reach; `0` for automatic |
| `JARVIS_MIC_SENS` | `1.0` | above 1 hears quieter voices (automatic mode) |
| `JARVIS_MIC_GAIN` | `1.0` | digital boost for a quiet microphone (try 2 or 3) |
| `JARVIS_SILENCE` | `0.9` | seconds of silence that end your sentence |
| `JARVIS_WAKE` | `0.5` | wake-word sensitivity threshold (lower triggers more easily) |
| `JARVIS_WHISPER` | `mlx-community/whisper-large-v3-turbo` | speech model (`whisper-small-mlx` is faster) |
| `JARVIS_FOLLOWUP` | `1` | keep listening briefly after a reply; `0` to disable |
| `JARVIS_FOLLOWUP_SECONDS` | `4.0` | how long that window lasts |
| `JARVIS_CLAPS` | `1` | double-clap wake; `0` to disable |
| `JARVIS_CLAP_SENS` | `1.0` | above 1 is more sensitive, below 1 stricter |
| `JARVIS_CLAP_TYPING_GUARD` | `0.6` | ignore claps this many seconds after keyboard or mouse use |
| `JARVIS_BARGE_IN` | `1` | a double clap stops Jarvis mid-reply; `0` to disable |

## Voice

| Variable | Default | Purpose |
|---|---|---|
| `JARVIS_TTS` | `auto` | force an engine: `eleven`, `kokoro`, `piper` or `mac` |
| `JARVIS_VOICE` | `Daniel` | macOS voice (best British quality installed is chosen) |
| `JARVIS_RATE` | `185` | macOS voice speed, words per minute |
| `JARVIS_KOKORO_VOICE` | `bm_george` | also `bm_daniel`, `bm_lewis`, `bm_fable`, `bf_emma` |
| `JARVIS_KOKORO_SPEED` | `1.05` | Kokoro speed |
| `JARVIS_PIPER` | `1` | use the Piper voice when installed; `0` to ignore it |
| `JARVIS_PIPER_MODEL` | `~/.jarvis/voice/jarvis.onnx` | path to a Piper model |
| `JARVIS_ELEVEN_VOICE` | `wDsJlOXPqcvIUKdLXjDs` | ElevenLabs voice id |
| `JARVIS_ELEVEN_MODEL` | `eleven_flash_v2_5` | ElevenLabs model |

## HUD

| Variable | Default | Purpose |
|---|---|---|
| `JARVIS_PORT` | `8765` | local port for the HUD (loopback only) |
| `JARVIS_NO_BROWSER` | unset | set to `1` to not open the HUD window automatically |
