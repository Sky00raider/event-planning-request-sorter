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


# ===== Milestone 3: LoRA by hand, Trainer fine-tuning, a small adapter file
# GPT-2's attention projection c_attn is a Conv1D (weight stored (in, out)). LoRA keeps it frozen and learns
# y = base(x) + (alpha/r) * (x A^T) B^T with A (r x in) small random and B (out x r) zero, so training starts from
# exactly the pretrained model. Only A, B and the label-token rows are trained: a ~600 KB adapter file.
def _is_conv1d(layer: nn.Module) -> bool:
    return type(layer).__name__ == "Conv1D"


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Module, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        n_in, n_out = base.weight.shape if _is_conv1d(base) else (base.in_features, base.out_features)
        self.r, self.scale = r, alpha / r
        w = base.weight
        self.A = nn.Parameter(torch.randn(r, n_in, dtype=w.dtype, device=w.device) * 0.01)
        self.B = nn.Parameter(torch.zeros(n_out, r, dtype=w.dtype, device=w.device))

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T

    @torch.no_grad()
    def merged(self) -> nn.Module:
        """Bake W' = W + scale * B @ A into the base layer (transposed for Conv1D)."""
        delta = self.scale * (self.B @ self.A)
        self.base.weight += delta.T if _is_conv1d(self.base) else delta
        return self.base


def _replace(model: nn.Module, name: str, new: nn.Module) -> None:
    parent, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, child, new)


def inject_lora(model: nn.Module, target: str = "c_attn", r: int = 8, alpha: float = 16.0) -> list[str]:
    for p in model.parameters():
        p.requires_grad_(False)
    names = [n for n, _ in model.named_modules() if n == target or n.endswith("." + target)]
    if not names:
        raise ValueError(f"no layer called {target!r} in the model")
    for n in names:
        _replace(model, n, LoRALinear(model.get_submodule(n), r, alpha))
    return names


def merge_lora(model: nn.Module) -> int:
    names = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    for n in names:
        _replace(model, n, model.get_submodule(n).merged())
    return len(names)


def lora_state(model: nn.Module) -> dict:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n.endswith((".A", ".B"))}


@dataclass
class TaggerConfig:
    labels: list[str] = field(default_factory=lambda: list(LABELS))
    base: str = "gpt2"
    r: int = 8
    alpha: float = 16.0
    target: str = "c_attn"
    max_len: int = 128
    extra: dict = field(default_factory=dict)       # calibrated threshold, training notes


DEFAULT_ADAPTER = Path("models/adapter.pt")


def build(cfg: TaggerConfig, base=None, lora: bool = True):
    """Pretrained base + label tokens + LoRA. `base` is a callable returning (model, tokenizer) (tests pass a tiny one)."""
    model, tokenizer = base() if base else load_base(cfg.base)
    label_ids = add_label_tokens(model, tokenizer, cfg.labels)
    if lora:
        inject_lora(model, cfg.target, cfg.r, cfg.alpha)
    return model, tokenizer, label_ids


def save_adapter(model, label_ids: list[int], cfg: TaggerConfig, path: str | Path = DEFAULT_ADAPTER) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(cfg), "lora": lora_state(model), "label_ids": list(label_ids),
                "label_rows": model.get_input_embeddings().weight[label_ids].detach().cpu().clone()}, path)
    return path.stat().st_size


def load_adapter(path: str | Path = DEFAULT_ADAPTER, base=None):
    """Fresh pretrained base + saved adapter. Returns (model, tokenizer, label_ids, cfg)."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    cfg = TaggerConfig(**saved["config"])
    model, tokenizer, label_ids = build(cfg, base)
    if list(label_ids) != saved["label_ids"]:
        raise ValueError("label token ids differ from the adapter: different base model or labels?")
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, t in saved["lora"].items():
            if name not in params or params[name].shape != t.shape:
                raise ValueError(f"adapter tensor {name} does not fit this model")
            params[name].copy_(t)
        model.get_input_embeddings().weight[label_ids] = saved["label_rows"]
    return model.eval(), tokenizer, label_ids, cfg


@dataclass
class LabelPositionCollator:
    """Pad a batch; `labels` is -100 everywhere except the final position (the label token), so the loss
    only scores "which request type comes next?", never the request text itself."""
    pad_id: int

    def __call__(self, features: list[dict]) -> dict:
        seqs = [f["input_ids"] for f in features]
        ids, mask = pad_batch(seqs, self.pad_id)
        labels = torch.full_like(ids, -100)
        for i, s in enumerate(seqs):
            labels[i, len(s) - 1] = s[-1]
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def accuracy_of(model, encoder: RequestEncoder, rows: list[dict], labels: list[str], pad_id: int) -> float:
    scores = label_logits(model, encoder.prompts([r["text"] for r in rows]), encoder.label_ids, pad_id)
    target = torch.tensor([labels.index(r["label"]) for r in rows])
    return (scores.argmax(dim=1) == target).float().mean().item()


def train(cfg: TaggerConfig, train_rows: list[dict], val_rows: list[dict], base=None, epochs: float = 6,
          lr: float = 1e-3, batch_size: int = 8, seed: int = 0, out_dir: str = "data/trainer") -> dict:
    """Fine-tune with the Hugging Face Trainer. Returns model, tokenizer, label_ids, encoder and a report."""
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(seed)
    model, tokenizer, label_ids = build(cfg, base)
    train_only_label_rows(model, label_ids)
    encoder = RequestEncoder(tokenizer, label_ids, cfg.max_len)
    total = sum(p.numel() for p in model.parameters())
    lora_n = sum(p.numel() for n, p in model.named_parameters() if n.endswith((".A", ".B")))
    trainable = lora_n + len(label_ids) * model.get_input_embeddings().weight.shape[1]

    targets = [cfg.labels.index(r["label"]) for r in train_rows]
    examples = [{"input_ids": x} for x in encoder.training_examples([r["text"] for r in train_rows], targets)]
    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs, learning_rate=lr, per_device_train_batch_size=batch_size,
        weight_decay=0.0, lr_scheduler_type="linear", logging_steps=10, save_strategy="no", report_to="none",
        use_cpu=not torch.cuda.is_available(), seed=seed, remove_unused_columns=False, dataloader_num_workers=0)
    trainer = Trainer(model=model, args=args, train_dataset=examples,
                      data_collator=LabelPositionCollator(tokenizer.pad_token_id))
    t0 = time.time()
    result = trainer.train()
    seconds = time.time() - t0
    report = {"train_seconds": round(seconds, 1), "final_train_loss": round(result.training_loss, 4),
              "total_params": total, "trainable_params": trainable, "epochs": epochs, "lr": lr,
              "batch_size": batch_size, "train_examples": len(train_rows),
              "device": "cuda" if torch.cuda.is_available() else "cpu",
              "loss_log": [{"step": h["step"], "loss": h["loss"]} for h in trainer.state.log_history if "loss" in h]}
    if val_rows:
        report["val_accuracy"] = round(accuracy_of(model, encoder, val_rows, cfg.labels, tokenizer.pad_token_id), 4)
    return {"model": model, "tokenizer": tokenizer, "label_ids": label_ids, "encoder": encoder, "report": report}


# ===== Milestone 4: inference pipeline, confidence triage, request details
def decide(probs: np.ndarray, labels: list[str], threshold: float) -> list[dict]:
    """Best label, confidence, runner-up and a NEEDS REVIEW flag for each row of label probabilities."""
    out = []
    for p in np.asarray(probs):
        order = np.argsort(-p)
        out.append({"label": labels[order[0]], "confidence": round(float(p[order[0]]), 4),
                    "needs_review": bool(p[order[0]] < threshold), "runner_up": labels[order[1]]})
    return out


def calibrate_threshold(confidence: list[float], correct: list[bool], target: float = 0.95,
                        fallback: float = 0.9) -> float:
    """Lowest confidence at which everything auto-accepted (confidence >= t) is at least `target` accurate,
    needing at least 10 accepted requests so one lucky example cannot set the bar."""
    conf, ok = np.asarray(confidence), np.asarray(correct, dtype=float)
    for t in sorted(set(conf.tolist())):
        keep = conf >= t
        if keep.sum() >= 10 and ok[keep].mean() >= target:
            return round(float(t), 4)
    return fallback


MONTHS = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
          r"sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
WEEKDAYS = r"(?:mon|tues?|wed(?:nes)?|thu(?:rs)?|fri|sat(?:ur)?|sun)(?:day)?"
CITIES = ["Bengaluru", "Bangalore", "Mysuru", "Mysore", "Kochi", "Cochin", "Pune", "Delhi", "Goa", "Dubai",
          "Mumbai", "Chennai", "Hyderabad", "Kolkata", "Jaipur", "Udaipur", "Coorg", "Ooty", "Mangaluru",
          "Trivandrum", "Thiruvananthapuram", "Ahmedabad", "Singapore", "London"]
URGENT = re.compile(r"\b(urgent\w*|asap|immediately|right away|today|tonight|tomorrow|this (?:week|weekend)|"
                    r"(?:in|within) (?:\d+|a|two|three|few) days?|last[- ]minute|emergency)\b", re.I)


def _money(match: re.Match) -> float:
    value = float(match.group("num").replace(",", "") or 0)
    unit = (match.group("unit") or "").lower()
    if unit.startswith(("lakh", "lac")) or unit == "l":
        value *= 1e5
    elif unit.startswith("cr"):
        value *= 1e7
    elif unit == "k":
        value *= 1e3
    elif unit == "m":
        value *= 1e6
    return value


def extract_details(text: str) -> dict:
    """Pull out what a coordinator would copy by hand: guests, budget, date, city, urgency (regex only)."""
    out: dict = {}
    m = re.search(r"(\d[\d,]*)\s*\+?\s*(?:guests?|pax|people|persons?|attendees|members|heads|invitees|ppl)\b",
                  text, re.I)
    if m:
        out["guests"] = int(m.group(1).replace(",", ""))
    m = re.search(r"(?P<cur>rs\.?|inr|₹|\$|usd|aed)\s*(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>lakhs?|lacs?|crores?|cr|k|m|l)?\b",
                  text, re.I) or re.search(
        r"(?P<cur>)(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>lakhs?|lacs?|crores?)\b", text, re.I)
    if m:
        cur = m.group("cur") or ""
        out["budget"] = " ".join(m.group(0).split())
        out["budget_value"] = _money(m)
        out["currency"] = "USD" if cur in ("$",) or cur.lower() == "usd" else "AED" if cur.lower() == "aed" else "INR"
    for pat in (r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS}\b(?:,?\s+\d{{4}})?",
                rf"\b{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?:,?\s+\d{{4}})?",
                rf"\b(?:this|next|coming)\s+(?:{WEEKDAYS}|weekend|month)\b", r"\b(?:day after tomorrow|tomorrow|tonight|today)\b"):
        m = re.search(pat, text, re.I)
        if m:
            out["date"] = m.group(0)
            break
    for city in CITIES:
        if re.search(rf"\b{city}\b", text, re.I):
            out["city"] = city
            break
    out["urgent"] = bool(URGENT.search(text))
    return out


def priority_of(label: str, details: dict) -> str:
    """High for anything time-critical or unhappy; low for off-topic; normal otherwise."""
    if details.get("urgent") or label in ("complaint_feedback", "reschedule_cancel"):
        return "high"
    return "low" if label == "other" else "normal"


def make_record(text: str, decision: dict) -> dict:
    details = extract_details(text)
    return {"received": datetime.now(timezone.utc).isoformat(timespec="seconds"), "text": text, **decision,
            "team": TEAMS[decision["label"]], "priority": priority_of(decision["label"], details),
            "details": details}


class Sorter:
    """Loads the adapter, merges it into GPT-2 and classifies requests."""

    def __init__(self, adapter: str | Path = DEFAULT_ADAPTER, base=None, threshold: float | None = None):
        self.model, self.tokenizer, self.label_ids, self.cfg = load_adapter(adapter, base)
        merge_lora(self.model)
        self.labels = self.cfg.labels
        self.threshold = threshold if threshold is not None else self.cfg.extra.get("threshold", 0.9)
        self.encoder = RequestEncoder(self.tokenizer, self.label_ids, self.cfg.max_len)

    def probs(self, texts: list[str]) -> np.ndarray:
        prompts = self.encoder.prompts(texts)
        return softmax(label_logits(self.model, prompts, self.label_ids, self.tokenizer.pad_token_id).numpy())

    def classify(self, texts: list[str]) -> list[dict]:
        decisions = decide(self.probs(texts), self.labels, self.threshold)
        return [make_record(t, d) for t, d in zip(texts, decisions)]


def details_line(details: dict) -> str:
    return ", ".join(f"{k}={details[k]}" for k in ("guests", "budget", "date", "city") if k in details)


def format_record(rec: dict, index: int | None = None) -> str:
    flag = "NEEDS REVIEW" if rec["needs_review"] else "ok"
    head = f"[{index}] " if index is not None else ""
    lines = [f"{head}{rec['label']}  {rec['confidence']:.0%}  {flag}  -> {rec['team']}  (priority: {rec['priority']})",
             f"    \"{rec['text']}\""]
    if rec["needs_review"]:
        lines.append(f"    runner-up: {rec['runner_up']}")
    found = details_line(rec["details"]) + (" [URGENT]" if rec["details"].get("urgent") else "")
    if found.strip():
        lines.append(f"    found: {found.strip()}")
    return "\n".join(lines)


