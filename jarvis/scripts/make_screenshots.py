#!/usr/bin/env python3
"""Regenerate the README screenshots and demo GIF.

Starts the real HUD server, drives it with scripted events, and captures it with a headless browser.
Needs:  pip install playwright pillow && playwright install chromium
"""
import importlib.util
import os
import socket
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    from PIL import Image
    from playwright.sync_api import sync_playwright

    port = free_port()
    os.environ.update(HOME=tempfile.mkdtemp(), JARVIS_PORT=str(port), JARVIS_NO_BROWSER="1")
    spec = importlib.util.spec_from_file_location("jarvis", os.path.join(ROOT, "jarvis.py"))
    j = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(j)
    threading.Thread(target=j.start_ui, daemon=True).start()
    time.sleep(0.5)
    os.makedirs(OUT, exist_ok=True)

    script = [   # (state, text, who, caption)
        ("idle", None, None, "idle"),
        ("listening", "Listening...", None, "listening"),
        ("thinking", "What's the weather in London?", "you", "thinking"),
        ("speaking", "Light rain, eleven degrees. Take a coat.", "jarvis", "speaking"),
    ]
    frames = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(f"http://127.0.0.1:{port}/")
        page.wait_for_timeout(600)
        j.ui_emit("idle", info="BRAIN GROQ|OPENAI/GPT-OSS-120B|VOICE KOKORO (BM_GEORGE)|TONE CLASSIC")
        for state, text, who, name in script:
            kw = {"text": text} if text else {}
            if who:
                kw["who"] = who
            j.ui_emit(state, **kw)
            if state == "listening":
                for i in range(12):
                    j.ui_emit(level=0.3 + 0.5 * abs(((i * 7) % 10) / 10 - 0.5))
                    page.wait_for_timeout(60)
            page.wait_for_timeout(1400)
            path = os.path.join(OUT, f"hud-{name}.png")
            page.screenshot(path=path)
            print("wrote", path)
        # animated demo: loop through the states
        for state, text, who, _ in script * 2:
            kw = {"text": text} if text else {}
            if who:
                kw["who"] = who
            j.ui_emit(state, **kw)
            for _ in range(7):
                page.wait_for_timeout(110)
                frames.append(Image.open(__import__("io").BytesIO(page.screenshot())).convert("RGB").resize((720, 450), Image.LANCZOS))
        browser.close()

    sheet = Image.new("RGB", (1440, 900))
    for i, (_, _, _, name) in enumerate(script):
        sheet.paste(Image.open(os.path.join(OUT, f"hud-{name}.png")).resize((720, 450), Image.LANCZOS), ((i % 2) * 720, (i // 2) * 450))
    sheet.save(os.path.join(OUT, "hud-states.png"), optimize=True)
    pal = [f.quantize(colors=64, method=Image.Quantize.FASTOCTREE) for f in frames]
    pal[0].save(os.path.join(OUT, "demo.gif"), save_all=True, append_images=pal[1:], duration=110, loop=0, optimize=True)
    print("wrote hud-states.png and demo.gif", os.path.getsize(os.path.join(OUT, "demo.gif")) // 1024, "KB")


if __name__ == "__main__":
    sys.exit(main())
