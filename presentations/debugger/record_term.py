"""Record a real `difflow debug` session as a typed-terminal GIF.

Runs the CLI with a command script, captures what it prints, and replays it:
commands are typed a character at a time, responses appear as the debugger
printed them. Nothing is mocked: the transcript is the CLI's own output.

    python record_term.py NAME "difflow debug plant.py ..." cmd1 cmd2 ...
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
DEMO = HERE / "demo"
OUT = HERE / "assets"

COLS, ROWS = int(os.environ.get("TERM_COLS", 98)), 26
FONT = "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf"
BOLD = "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Bold.ttf"
SIZE = 15
PAD, BAR = 16, 30

BG, FG, DIM = "#1d1f27", "#d4d6de", "#80869a"
PROMPT, SHELL = "#7aa2f7", "#9ece6a"
BANNER, REASON, BAD, EDIT, CMD = "#e0af68", "#9ece6a", "#f7768e", "#bb9af7", "#ffffff"

font = ImageFont.truetype(FONT, SIZE)
bold = ImageFont.truetype(BOLD, SIZE)
#: glyphs JetBrains Mono lacks, drawn from DejaVu Sans Mono instead
FALLBACK = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", SIZE)
MISSING = set("↳✗✎…")
CW = font.getbbox("M")[2]
LH = int(SIZE * 1.45)
W = PAD * 2 + CW * COLS
H = BAR + PAD * 2 + LH * ROWS


def style(line: str):
    """Segments (text, colour, font) for one output line."""
    if line.startswith("── difflow-dbg ──"):
        return [(line, BANNER, bold)]
    stripped = line.lstrip()
    if stripped.startswith("↳"):
        return [(line, REASON, font)]
    if stripped.startswith("✗") or "nan" in line.split() or line.startswith("difflow debug: stopped"):
        return [(line, BAD, font)]
    if stripped.startswith("✎") or stripped.startswith("next pass:"):
        return [(line, EDIT, font)]
    if stripped.startswith("[") and "]" in stripped[:40]:          # a watch
        return [(line, EDIT, font)]
    if line.startswith(("stream ", "stepping", "continuing", "breakpoint", "conditional",
                        "watching", "all breakpoints", "will stop", "stop-at", "stopping")):
        return [(line, DIM, font)]
    if line.startswith('{"event"'):
        return json_style(line)
    return [(line, FG, font)]


def compact(line: str) -> str:
    """One protocol event without its null and empty fields (to fit a slide).

    Only fields whose value is null, empty or the default request id are
    dropped, and hello's long lists are summarised; every value shown is the
    one the CLI wrote.
    """
    event = json.loads(line)
    keep = {}
    for key, value in event.items():
        if value in (None, [], {}, "") or key == "request_id":
            continue
        if event.get("event") == "hello" and key in ("events", "metrics", "commands",
                                                       "capabilities"):
            keep[key] = "…"
            continue
        if key == "residual" and isinstance(value, float):
            value = float(f"{value:.4g}")
        keep[key] = value
    return json.dumps(keep, ensure_ascii=False, separators=(",", ":")).replace('"…"', "…")


def json_style(line: str):
    try:
        kind = json.loads(line).get("event")
        line = compact(line)
    except Exception:
        kind = None
    colour = {"hello": DIM, "pause": BANNER, "result": FG}.get(kind, FG)
    return [(line, colour, font)]


def clip(segs):
    out, n = [], 0
    for text, colour, f in segs:
        room = COLS - n
        if room <= 0:
            break
        if len(text) > room:
            text = text[: room - 1] + "…"
        out.append((text, colour, f))
        n += len(text)
    return out


class Screen:
    def __init__(self):
        self.lines: list[list] = [[]]
        self.frames: list[tuple[Image.Image, int]] = []
        self.cursor = True

    def newline(self):
        self.lines.append([])

    def add(self, segs):
        self.lines[-1] = clip(self.lines[-1] + segs)

    def render(self, cursor=False) -> Image.Image:
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W, BAR], fill="#2a2d38")
        for i, c in enumerate(("#ff5f57", "#febc2e", "#28c840")):
            d.ellipse([14 + i * 20, 9, 26 + i * 20, 21], fill=c)
        d.text((W // 2, BAR // 2), "difflow debug", fill=DIM, font=font, anchor="mm")
        view = self.lines[-ROWS:]
        for r, segs in enumerate(view):
            x, y = PAD, BAR + PAD + r * LH
            for text, colour, f in segs:
                for ch in text:
                    d.text((x, y), ch, fill=colour, font=FALLBACK if ch in MISSING else f)
                    x += CW
            if cursor and r == len(view) - 1:
                d.rectangle([x + 1, y + 2, x + CW - 1, y + LH - 4], fill=FG)
        return img

    def frame(self, ms, cursor=False):
        self.frames.append((self.render(cursor), ms))


def play(screen: Screen, shell: str, transcript: list[str], speed=1.0, sent=None):
    t = lambda ms: int(ms / speed)
    screen.add([("$ ", SHELL, bold)])
    screen.frame(t(500), cursor=True)
    for ch in shell:
        screen.add([(ch, CMD, font)])
        screen.frame(t(28), cursor=True)
    screen.frame(t(500), cursor=True)
    screen.newline()
    screen.frame(t(900))
    block = 0
    sent = list(sent or [])
    for line in transcript:
        if sent and line.startswith('{"event": "result"'):
            # JSON mode: the client's request, typed, before the reply it gets
            if block:
                screen.frame(t(min(2600, 900 + 110 * block)))
            block = 0
            screen.add([("→ ", PROMPT, bold)])
            screen.frame(t(300), cursor=True)
            for ch in json.dumps(sent.pop(0)):
                screen.add([(ch, CMD, bold)])
                screen.frame(t(30), cursor=True)
            screen.newline()
        if line.startswith("difflow-dbg> "):
            if block:
                screen.frame(t(min(2600, 900 + 110 * block)))
            block = 0
            cmd = line[len("difflow-dbg> "):]
            screen.add([("difflow-dbg> ", PROMPT, bold)])
            screen.frame(t(450), cursor=True)
            for ch in cmd:
                screen.add([(ch, CMD, bold)])
                screen.frame(t(42), cursor=True)
            screen.frame(t(350), cursor=True)
            screen.newline()
            continue
        screen.add(style(line))
        screen.newline()
        block += 1
        screen.frame(t(60))
    screen.frame(t(3500))


def gif(frames, path: Path, width=None):
    """Frames -> GIF through ffmpeg's palettegen (one palette for the whole clip)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        lines = []
        merged = []
        for img, ms in frames:                           # drop repeats, keep their time
            if merged and merged[-1][0].tobytes() == img.tobytes():
                merged[-1] = (merged[-1][0], merged[-1][1] + ms)
            else:
                merged.append((img, ms))
        for i, (img, ms) in enumerate(merged):
            name = tmp / f"f{i:05d}.png"
            img.save(name)
            lines += [f"file '{name}'", f"duration {ms / 1000:.3f}"]
        lines.append(f"file '{tmp / f'f{len(merged) - 1:05d}.png'}'")
        (tmp / "list.txt").write_text("\n".join(lines) + "\n")
        scale = f"scale={width}:-1:flags=lanczos," if width else ""
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", str(tmp / "list.txt"), "-vf",
             f"{scale}split[a][b];[a]palettegen=max_colors=64:stats_mode=full[p];"
             "[b][p]paletteuse=dither=none", "-loop", "0", str(path)],
            check=True)
        merged[-1][0].save(path.with_suffix(".png"))       # the last frame, as a poster


def run(shell: str, commands: list[str], stdin_lines: list[str] | None = None,
        name: str = "session") -> list[str]:
    """Run the CLI for real; return its output lines (the startup banner trimmed)."""
    args = shlex.split(shell)
    script = None
    if commands:
        script = DEMO / f"_{name}.dbg"
        script.write_text("\n".join(commands) + "\n")
        args += ["--script", str(script)]
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONWARNINGS="ignore", PYTHONPATH=str(DEMO))
    proc = subprocess.run(args, cwd=DEMO, env=env, text=True, capture_output=True,
                          input="\n".join(stdin_lines or []) + ("\n" if stdin_lines else ""))
    text = proc.stdout + proc.stderr if "--json" not in shell else proc.stdout + proc.stderr
    return [l for l in text.splitlines() if l.strip() and "Warning" not in l
            and not l.startswith(("  warnings.warn", "  result =", "  streams ="))]


if __name__ == "__main__":
    name, shell, *cmds = sys.argv[1:]
    OUT.mkdir(exist_ok=True)
    lines = run(shell, cmds, name=name)
    (OUT / f"{name}.txt").write_text("\n".join(lines) + "\n")
    s = Screen()
    play(s, shell, lines, sent=cmds if "--json" in shell else None)
    gif(s.frames, OUT / f"{name}.gif")
    size = (OUT / f"{name}.gif").stat().st_size
    print(f"{name}: {len(lines)} lines, {len(s.frames)} frames, {size / 1024:.0f} KB")
