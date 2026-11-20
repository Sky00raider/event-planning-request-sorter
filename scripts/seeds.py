"""Train with several random seeds and report mean and spread, so one lucky or unlucky run does not mislead.

    python scripts/seeds.py [--seeds 0 1 2]      (GPU recommended: about 25 s per seed on a T4)

Writes results/seeds.md and results/seeds.json. The shipped models/adapter.pt is NOT touched.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import core  # noqa: E402


def evaluate(run: dict, cfg: core.TaggerConfig, rows: list[dict]) -> dict:
    enc, tok = run["encoder"], run["tokenizer"]
    texts, gold = [r["text"] for r in rows], [r["label"] for r in rows]
    probs = core.softmax(core.label_logits(run["model"], enc.prompts(texts), enc.label_ids, tok.pad_token_id).numpy())
    pred = [cfg.labels[i] for i in probs.argmax(axis=1)]
    auto = [c >= core.THRESHOLD_FLOOR and not core.too_short(t) for c, t in zip(probs.max(axis=1), texts)]
    right = [p == g for p, g in zip(pred, gold)]
    return {"accuracy": core.accuracy(pred, gold), "macro_f1": core.macro_f1(pred, gold),
            "auto": sum(auto) / len(auto),
            "auto_accuracy": sum(r for r, a in zip(right, auto) if a) / max(sum(auto), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    rows = core.load_requests()
    by = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test", "hard")}
    results = []
    for seed in a.seeds:
        cfg = core.TaggerConfig()
        run = core.train(cfg, by["train"], by["val"], seed=seed, out_dir=f"data/trainer_seed{seed}")
        res = {"seed": seed, "val_accuracy": run["report"]["val_accuracy"],
               "train_seconds": run["report"]["train_seconds"],
               "test": evaluate(run, cfg, by["test"]), "hard": evaluate(run, cfg, by["hard"])}
        results.append(res)
        print(f"seed {seed}: val {res['val_accuracy']:.3f} test {res['test']['accuracy']:.3f} "
              f"hard {res['hard']['accuracy']:.3f}", flush=True)
        del run

    def stat(get) -> str:
        v = np.array([get(r) for r in results]) * 100
        return f"{v.mean():.1f} ± {v.std(ddof=1) if len(v) > 1 else 0:.1f}"

    lines = ["# Seed variation", "",
             f"Same data and settings, {len(results)} training seeds ({', '.join(str(r['seed']) for r in results)}). "
             "Mean ± sample standard deviation, in percent.", "",
             "| Seed | Val acc | Test acc | Hard acc | Test macro-F1 (x100) | Hard macro-F1 (x100) |", "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['seed']} | {100 * r['val_accuracy']:.1f} | {100 * r['test']['accuracy']:.1f} | "
                     f"{100 * r['hard']['accuracy']:.1f} | {100 * r['test']['macro_f1']:.1f} | {100 * r['hard']['macro_f1']:.1f} |")
    lines += [f"| **mean ± sd** | {stat(lambda r: r['val_accuracy'])} | {stat(lambda r: r['test']['accuracy'])} | "
              f"{stat(lambda r: r['hard']['accuracy'])} | {stat(lambda r: r['test']['macro_f1'])} | "
              f"{stat(lambda r: r['hard']['macro_f1'])} |", "",
              f"Review policy (confidence >= {core.THRESHOLD_FLOOR} and >= {core.MIN_WORDS} words), mean over seeds:", "",
              "| Split | Auto-sorted | Accuracy of those |", "|---|---|---|"]
    for split in ("test", "hard"):
        lines.append(f"| {split} | {stat(lambda r: r[split]['auto'])}% | {stat(lambda r: r[split]['auto_accuracy'])}% |")
    Path("results").mkdir(exist_ok=True)
    Path("results/seeds.json").write_text(json.dumps(results, indent=2, default=lambda o: o.item()), encoding="utf-8")
    Path("results/seeds.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
