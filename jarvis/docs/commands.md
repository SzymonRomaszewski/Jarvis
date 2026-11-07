# Things to say

Start with "Hey Jarvis" (or clap twice), wait for the chime, then speak. After a reply Jarvis listens for a few seconds, so you can follow up without the wake word. You can also click the glowing core, press Space in the HUD, or type into the box.

## Instant (no AI, work offline)

- "What time is it?" / "What's the date?"
- "Volume 40", "Set the volume to thirty", "Turn it up", "Quieter", "Mute", "Unmute"
- "Open Safari", "Launch Chrome", "Open the Spotify app", "Quit Notes" (app names are matched loosely, so slips of speech recognition are forgiven)
- "Set a timer for five minutes", "Ten minute timer", "Timer for an hour and 30 minutes"
- "Pause", "Play", "Next song", "Previous track"
- "Take a screenshot"

## Calendar and reminders

- "What's on my calendar today?" / "...this week?"
- "Add dentist on Monday at 3:30 to my calendar"
- "Put Mum's birthday on the 1st of December" (all-day)
- "What reminders do I have?"
- "Remind me to buy milk tonight at six"
- "Mark the milk reminder as done"

The first time, macOS asks to let Terminal control Calendar and Reminders. Allow it.

## Knowledge

- "What's the weather?" (uses your home location; say "my home is in London" once)
- "Will it rain in Manchester tomorrow?"
- "Who wrote Pride and Prejudice?" / "Look up the population of Iceland"
- "What's on my screen?" / "Summarise this page"

## Working with other apps

- "Ask Claude what the capital of France is" (pastes into the Claude app and presses Enter)
- "Type hello into Notes" (pastes without pressing Enter)
- "Run my Lights Off shortcut" (macOS Shortcuts)
- "Play something by Coldplay" (Apple Music; Spotify opens a search)

## Shortcuts you define

- "Work mode", "Movie mode", "Good morning" (included)
- "Make a shortcut called study mode that opens Notes and Safari and sets the volume to 10"
- "List my shortcuts", "Delete the study mode shortcut"

Or edit `~/.jarvis/shortcuts.json`. Each shortcut is a list of steps, an optional spoken reply, and an optional `ask` that has the AI answer something:

```json
{
  "good morning": {"ask": "In one or two sentences: today's date, my calendar and reminders."},
  "work mode": {
    "do": [{"tool": "open_app", "name": "Safari"}, {"tool": "set_volume", "level": 25}],
    "say": "Work mode."
  },
  "lights off": {"do": [{"tool": "apple_shortcut", "name": "Lights Off"}], "say": "Lights off."},
  "lock screen": {"do": [{"shell": "pmset displaysleepnow"}]}
}
```

A `{"shell": "..."}` step runs without asking, because only you can edit this file. Any tool in [tools.md](tools.md) can be a step.

## Personality, memory and housekeeping

- "Be more sarcastic" / "Switch to butler mode" / "Be minimal" / "Talk like a pirate"
- "Call me boss"
- "Remember I'm vegetarian" / "Forget that"
- "Never mind" cancels. "Goodbye" or "Shut down" quits Jarvis.
- A **double clap** while Jarvis is thinking or talking cancels the reply.
