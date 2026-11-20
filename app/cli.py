"""Command-line entry point: `python -m app.cli --help`. One section per milestone."""
import argparse
import json
import sys
from pathlib import Path

from app import core

DEFAULT_OUT = "outputs/classified.jsonl"


def _add_hello(sub):
    sub.add_parser("hello", help="check that everything is wired up").set_defaults(
        func=lambda a: print("Event Planning Request Sorter is ready. Open README.md for your milestones."))


# ===== Main
def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")      # Windows consoles default to a legacy code page
    except (AttributeError, ValueError, OSError):
        pass
    parser = argparse.ArgumentParser(
        prog="event-planning-request-sorter",
        description="Sort incoming client requests for event planners into request types with a fine-tuned GPT-2.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, add in list(globals().items()):
        if name.startswith("_add_"):
            add(sub)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
