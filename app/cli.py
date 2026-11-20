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


# ===== Milestone 3: train
def _add_train(sub):
    p = sub.add_parser("train", help="fine-tune GPT-2 on data/requests.jsonl and save the adapter")
    p.add_argument("--base", default="gpt2")
    p.add_argument("--epochs", type=float, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data", default=str(core.DATA_PATH))
    p.add_argument("--corrections", help="a corrections .jsonl from `review`, added to the training data")
    p.add_argument("--out", default=str(core.DEFAULT_ADAPTER))
    p.add_argument("--report", default="results/train_report.json")
    p.add_argument("--target", type=float, default=0.95, help="accuracy the auto-accepted requests should reach")
    p.set_defaults(func=cmd_train)


def cmd_train(a):
    import numpy as np

    rows = core.load_requests(a.data)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    if a.corrections:
        extra = core.load_requests(a.corrections)
        train_rows += extra
        print(f"added {len(extra)} reviewed corrections to the training data")
    cfg = core.TaggerConfig(base=a.base)
    run = core.train(cfg, train_rows, val_rows, epochs=a.epochs, lr=a.lr)
    model, tok, enc = run["model"], run["tokenizer"], run["encoder"]
    scores = core.label_logits(model, enc.prompts([r["text"] for r in val_rows]), enc.label_ids, tok.pad_token_id)
    probs = core.softmax(scores.numpy())
    pred = [cfg.labels[i] for i in probs.argmax(axis=1)]
    ok = [p == r["label"] for p, r in zip(pred, val_rows)]
    cfg.extra = {"threshold": core.calibrate_threshold(probs.max(axis=1).tolist(), ok, a.target),
                 "trained_on": len(train_rows), "epochs": a.epochs}
    size = core.save_adapter(model, enc.label_ids, cfg, a.out)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps({**run["report"], "threshold": cfg.extra["threshold"],
                                          "adapter_bytes": size}, indent=2), encoding="utf-8")
    print(f"val accuracy {np.mean(ok):.3f}, review threshold {cfg.extra['threshold']}, "
          f"adapter {size / 1024:.0f} KB -> {a.out}")


# ===== Milestone 4: classify
def _add_classify(sub):
    p = sub.add_parser("classify", help="classify one or more requests given on the command line")
    p.add_argument("texts", nargs="+")
    _common_args(p)
    p.set_defaults(func=cmd_classify)
    c = sub.add_parser("chat", help="type requests one per line; an empty line quits")
    _common_args(c)
    c.set_defaults(func=cmd_chat)


def _common_args(p):
    p.add_argument("--adapter", default=str(core.DEFAULT_ADAPTER))
    p.add_argument("--out", default=DEFAULT_OUT, help="file to save to: .jsonl or .csv")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--threads", type=int, help="limit CPU threads (handy when the machine is busy)")


def get_sorter(a) -> "core.Sorter":
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    return core.Sorter(a.adapter)


def _show_and_save(a, records):
    for i, r in enumerate(records, 1):
        print(core.format_record(r, i))
    if not a.no_save and records:
        n = core.save_records(records, a.out)
        print(f"saved {n} request(s) -> {a.out}")


def cmd_classify(a):
    _show_and_save(a, get_sorter(a).classify(a.texts))


def cmd_chat(a):
    sorter = get_sorter(a)
    print("Type a client request and press Enter (empty line to quit).")
    while True:
        try:
            text = input("request> ").strip()
        except EOFError:
            break
        if not text:
            break
        _show_and_save(a, sorter.classify([text]))


# ===== Milestone 5: batch, review, digest
def _add_batch(sub):
    b = sub.add_parser("batch", help="classify every request in a .txt, .csv or .jsonl file")
    b.add_argument("file")
    _common_args(b)
    b.add_argument("--digest", help="also write a markdown hand-over sheet to this path")
    b.set_defaults(func=cmd_batch)
    r = sub.add_parser("review", help="confirm or correct the NEEDS REVIEW requests in a saved file")
    r.add_argument("--file", default=DEFAULT_OUT)
    r.add_argument("--corrections", default="data/corrections.jsonl")
    r.set_defaults(func=lambda a: print("review done:", core.review(a.file, a.corrections)))
    d = sub.add_parser("digest", help="turn a saved file into a markdown digest grouped by team")
    d.add_argument("--file", default=DEFAULT_OUT)
    d.add_argument("--out", help="write to this file instead of printing")
    d.set_defaults(func=cmd_digest)


def cmd_batch(a):
    records = get_sorter(a).classify(core.read_requests(a.file))
    _show_and_save(a, records)
    flagged = sum(r["needs_review"] for r in records)
    print(f"{len(records)} requests, {len(records) - flagged} auto-sorted, {flagged} need review")
    if a.digest:
        Path(a.digest).parent.mkdir(parents=True, exist_ok=True)
        Path(a.digest).write_text(core.build_digest(records), encoding="utf-8")
        print(f"digest -> {a.digest}")


def cmd_digest(a):
    text = core.build_digest(core.load_records(a.file))
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"digest -> {a.out}")
    else:
        print(text)


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
