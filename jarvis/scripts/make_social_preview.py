#!/usr/bin/env python3
"""Create assets/social-preview.png (1280x640), the image GitHub and LinkedIn show when the repo link is shared.

Upload it in GitHub: Settings > General > Social preview. Needs: pip install pillow
Run scripts/make_screenshots.py first (it creates the HUD screenshot this uses).
"""
import os

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def main():
    W, H = 1280, 640
    bg = Image.new("RGB", (W, H), (2, 6, 13))
    hud = Image.open(os.path.join(ASSETS, "hud-speaking.png")).convert("RGB")
    ImageDraw.Draw(hud).rectangle((1000, 705, 1440, 900), fill=(1, 4, 10))        # hide the conversation-log fragment
    core = hud.crop((330, 20, 1110, 735)).resize((620, 568), Image.LANCZOS)      # the rings only, without the UI text
    bg.paste(core, (W - 640, 36))
    # fade the left edge of the HUD into the background
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rectangle((W - 640, 0, W, H), fill=255)
    fade = Image.new("L", (W, H), 0)
    for x in range(W - 640, W - 440):
        ImageDraw.Draw(fade).line((x, 0, x, H), fill=int(255 * (x - (W - 640)) / 200))
    ImageDraw.Draw(fade).rectangle((W - 440, 0, W, H), fill=255)
    base = Image.new("RGB", (W, H), (2, 6, 13))
    base.paste(bg, (0, 0), Image.composite(fade, Image.new("L", (W, H), 0), mask))
    d = ImageDraw.Draw(base)
    d.text((70, 150), "J.A.R.V.I.S", font=font(BOLD, 78), fill=(255, 255, 255))
    d.text((74, 250), "A voice assistant for your Mac", font=font(REGULAR, 34), fill=(120, 200, 255))
    for i, line in enumerate(["Hey Jarvis, or clap twice", "Acts on your Mac: apps, calendar, music",
                              "Free to run. Speech stays on your device."]):
        d.text((74, 340 + i * 44), line, font=font(REGULAR, 24), fill=(160, 190, 215))
    d.line((74, 318, 330, 318), fill=(60, 140, 255), width=3)
    d.text((74, 575), "github.com/YOUR-USERNAME/jarvis", font=font(REGULAR, 22), fill=(90, 125, 160))
    out = os.path.join(ASSETS, "social-preview.png")
    base.save(out, optimize=True)
    print("wrote", out, os.path.getsize(out) // 1024, "KB")


if __name__ == "__main__":
    main()
