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


# ===== Milestone 2: extended tokenizer and model
# "<venue_inquiry>" etc. are added as special tokens, so each request type is exactly ONE token. The embedding
# matrix grows by len(LABELS) rows. GPT-2 ties its output layer to the input embeddings, so a label's row is both
# what the model reads and what the last hidden state is scored against when it predicts the label.
# Each new row starts as the mean embedding of the label's English name (" venue inquiry"), not as noise.
PROMPT_SUFFIX = " "


def load_base(base: str = "gpt2"):
    """Pretrained GPT-2 and its tokenizer (downloaded from Hugging Face the first time)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    return AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32), AutoTokenizer.from_pretrained(base)


def add_label_tokens(model, tokenizer, labels: list[str]) -> list[int]:
    """Pad token = end-of-text, one new token per label, resized embeddings. Returns the label token ids."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    weight = model.get_input_embeddings().weight
    with torch.no_grad():
        start = torch.stack([weight[tokenizer(label_words(n), add_special_tokens=False).input_ids].mean(dim=0)
                             for n in labels])
    tokens = [label_token(n) for n in labels]
    added = tokenizer.add_tokens(tokens, special_tokens=True)
    if added != len(labels):
        raise ValueError(f"expected {len(labels)} new tokens, the tokenizer added {added}")
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    with torch.no_grad():
        model.get_input_embeddings().weight[ids] = start.to(weight.dtype)
    return ids


def train_only_label_rows(model, label_ids: list[int]):
    """Make the embedding matrix trainable but zero the gradient of every row except the label rows."""
    weight = model.get_input_embeddings().weight
    weight.requires_grad_(True)
    keep = torch.zeros(weight.shape[0], 1, dtype=weight.dtype, device=weight.device)
    keep[label_ids] = 1.0
    return weight.register_hook(lambda grad: grad * keep)


class RequestEncoder:
    """Request text -> token ids. Training example = "request text <label>"; at prediction time the
    label is left off and the model must produce it right after the trailing space."""

    def __init__(self, tokenizer, label_ids: list[int], max_len: int = 128):
        self.tokenizer, self.label_ids = tokenizer, list(label_ids)
        self.suffix = tokenizer(PROMPT_SUFFIX, add_special_tokens=False).input_ids
        self.room = max_len - len(self.suffix) - 1

    def prompts(self, texts: list[str]) -> list[list[int]]:
        ids = self.tokenizer([" ".join(t.split()) for t in texts], add_special_tokens=False, verbose=False).input_ids
        return [x[:self.room] + self.suffix for x in ids]

    def training_examples(self, texts: list[str], targets: list[int]) -> list[list[int]]:
        return [p + [self.label_ids[t]] for p, t in zip(self.prompts(texts), targets)]


def pad_batch(seqs: list[list[int]], pad_id: int, device=None):
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids.to(device), mask.to(device)


@torch.no_grad()
def label_logits(model, prompts: list[list[int]], label_ids: list[int], pad_id: int, batch_size: int = 32):
    """Scores of the k label tokens after each prompt, shape (N, k): hidden state of the last real token
    dotted with the k label rows of the tied output layer (no full-vocabulary logits are built)."""
    model.eval()
    device = next(model.parameters()).device
    rows = model.get_output_embeddings().weight[label_ids]
    out = torch.zeros(len(prompts), len(label_ids))
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        ids, mask = pad_batch([prompts[i] for i in idx], pad_id, device)
        hidden = model.base_model(input_ids=ids, attention_mask=mask).last_hidden_state
        last = mask.sum(dim=1) - 1
        out[idx] = (hidden[torch.arange(len(idx), device=device), last] @ rows.T).float().cpu()
    return out


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


