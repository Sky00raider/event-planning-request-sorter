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
