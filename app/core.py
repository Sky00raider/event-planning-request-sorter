"""Event request sorter: GPT-2 + one new token per request type, fine-tuned with a hand-written LoRA.

The file has one section per milestone (search for "===== Milestone"), then the evaluation section.
"""
import csv
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

# ===== Milestone 1: request types and a small dataset
# Each type is one label GPT-2 will learn to name. TEAMS says who in the planning company gets it.
LABELS = [
    "venue_inquiry", "catering_request", "decor_design", "entertainment_booking", "photo_video",
    "guest_logistics", "quote_pricing", "reschedule_cancel", "general_planning", "complaint_feedback", "other",
]
TEAMS = {
    "venue_inquiry": "Venues desk", "catering_request": "Catering desk", "decor_design": "Design studio",
    "entertainment_booking": "Entertainment desk", "photo_video": "Media desk",
    "guest_logistics": "Guest services", "quote_pricing": "Sales and quotes",
    "reschedule_cancel": "Bookings and changes", "general_planning": "Lead planners",
    "complaint_feedback": "Client care", "other": "Front desk (no action)",
}
DATA_PATH = Path("data/requests.jsonl")


def label_token(name: str) -> str:
    return f"<{name}>"


def label_words(name: str) -> str:
    return " " + name.replace("_", " ")


def load_requests(path: str | Path = DATA_PATH, splits: tuple[str, ...] | None = None) -> list[dict]:
    """Rows {"text", "label", "split"} from a .jsonl file, optionally only the given splits."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if splits is None or r["split"] in splits]


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def validate_dataset(rows: list[dict]) -> list[str]:
    """Problems found (empty list = fine): unknown labels, empty text, repeated texts, train/eval overlap."""
    problems, seen = [], {}
    for r in rows:
        if r["label"] not in LABELS:
            problems.append(f"unknown label {r['label']!r}")
        if not r["text"].strip():
            problems.append("empty text")
        key = _norm(r["text"])
        if key in seen:
            problems.append(f"repeated text in {seen[key]} and {r['split']}: {r['text'][:40]!r}")
        seen[key] = r["split"]
    return problems


def class_counts(rows: list[dict]) -> dict[str, int]:
    out = {name: 0 for name in LABELS}
    for r in rows:
        out[r["label"]] += 1
    return out


