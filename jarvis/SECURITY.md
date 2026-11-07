# Security policy

## Supported versions

The latest release receives security fixes.

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Use GitHub's private vulnerability reporting ("Security" tab, "Report a vulnerability") or contact the maintainer directly through the email on their GitHub profile.

Include what you found, how to reproduce it, and what you think the impact is. You will get an acknowledgement within a few days. Reports are very welcome in these areas in particular:

- ways a web page, screenshot or other untrusted content could make Jarvis run something it should not (see `UNTRUSTED_TOOLS` and `SENSITIVE_TOOLS`);
- ways to bypass spoken confirmation;
- ways for another website or local process to command the HUD server (token, `Host` and `Origin` checks);
- anywhere a secret could be written to disk, a log or the screen.

The threat model and what is intentionally out of scope are described in [docs/privacy-and-security.md](docs/privacy-and-security.md).

## If you accidentally leak a key

Revoke it at the provider straight away (for Groq: console.groq.com/keys), create a new one, and remove the old one from any history or screenshots.
