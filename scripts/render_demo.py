"""Run the real CLI on a few requests and draw the output as a terminal screenshot: docs/demo.svg.

    python scripts/render_demo.py [--threads 2]
Needs a trained adapter (models/adapter.pt). The text in the SVG is captured from the CLI, not typed by hand.
"""
import argparse
import subprocess
import sys
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
COMMANDS = [
    ["classify", "Need a lawn in Mysuru for 200 guests on 12 Dec, budget Rs 5 lakh",
     "the DJ was 90 minutes late and the speakers kept cutting out, we want a refund asap",
     "--out", "samples/classified_demo.jsonl"],
    ["batch", "samples/incoming.txt", "--out", "samples/classified.jsonl", "--digest", "samples/digest.md"],
]
MAX_LINES, WRAP = 44, 104


def capture(args: list[str], threads: str | None) -> list[str]:
    cmd = [sys.executable, "-B", "-m", "app.cli", *args] + (["--threads", threads] if threads else [])
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout
    shown = "python -m app.cli " + " ".join(f'"{x}"' if " " in x else x for x in args)
    return ["$ " + shown] + out.splitlines()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads")
    a = ap.parse_args()
    for f in ("samples/classified_demo.jsonl", "samples/classified.jsonl"):
        (ROOT / f).unlink(missing_ok=True)
    lines: list[str] = []
    for cmd in COMMANDS:
        wrapped = []
        for ln in capture(cmd, a.threads):
            wrapped += textwrap.wrap(ln, WRAP, subsequent_indent="      ", drop_whitespace=False) or [""]
        if cmd[0] == "batch":                       # keep the screenshot short: first 4 requests + summary
            cut = next((i for i, ln in enumerate(wrapped) if ln.startswith("[5]")), len(wrapped))
            tail = [ln for ln in wrapped if "requests," in ln or ln.startswith(("saved", "digest"))]
            wrapped = wrapped[:cut] + ["    ..."] + tail
        lines += wrapped + [""]
    lines = lines[:MAX_LINES]
    h = 56 + 19 * len(lines)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{h}" viewBox="0 0 960 {h}" '
           f'font-family="Consolas, Menlo, monospace" font-size="13">',
           f'<rect width="960" height="{h}" rx="10" fill="#14161c"/><rect width="960" height="30" rx="10" fill="#262a35"/>',
           '<circle cx="20" cy="15" r="6" fill="#ff5f56"/><circle cx="40" cy="15" r="6" fill="#ffbd2e"/>'
           '<circle cx="60" cy="15" r="6" fill="#27c93f"/>',
           '<text x="480" y="20" fill="#9aa0b4" text-anchor="middle">event-planning-request-sorter</text>']
    for i, ln in enumerate(lines):
        color = ("#7ee787" if ln.startswith("$") else "#ffa657" if "NEEDS REVIEW" in ln or "[URGENT]" in ln
                 else "#8b93a7" if ln.startswith("    ") else "#e6e9f2")
        svg.append(f'<text x="18" y="{56 + 19 * i}" fill="{color}" xml:space="preserve">{escape(ln)}</text>')
    svg.append("</svg>")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/demo.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(f"wrote docs/demo.svg ({len(lines)} lines)")


if __name__ == "__main__":
    main()
