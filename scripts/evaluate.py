"""Compare the fine-tuned GPT-2 against two baselines and write results/results.md (+ results.json).

    python scripts/evaluate.py [--adapter models/adapter.pt] [--threads 2]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import core  # noqa: E402


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(core.DEFAULT_ADAPTER))
    ap.add_argument("--threads", type=int)
    a = ap.parse_args()
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)

    rows = core.load_requests()
    by = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test", "hard")}
    sorter = core.Sorter(a.adapter)
    nb = core.NaiveBayes().fit(by["train"])
    kw = core.KeywordBaseline()
    out: dict = {"threshold": sorter.threshold, "calibrated_threshold": sorter.calibrated, "splits": {}}

    for split in ("test", "hard"):
        texts, gold = [r["text"] for r in by[split]], [r["label"] for r in by[split]]
        probs = sorter.probs(texts)
        pred = [sorter.labels[i] for i in probs.argmax(axis=1)]
        conf = probs.max(axis=1)
        right = [p == g for p, g in zip(pred, gold)]
        nb_pred, kw_pred = nb.predict(texts), kw.predict(texts)
        auto = [c >= sorter.threshold and not core.too_short(t) for c, t in zip(conf, texts)]
        n_auto = sum(auto)
        out["splits"][split] = {
            "n": len(texts),
            "gpt2": {"accuracy": core.accuracy(pred, gold), "macro_f1": core.macro_f1(pred, gold)},
            "naive_bayes": {"accuracy": core.accuracy(nb_pred, gold), "macro_f1": core.macro_f1(nb_pred, gold)},
            "keywords": {"accuracy": core.accuracy(kw_pred, gold), "macro_f1": core.macro_f1(kw_pred, gold)},
            "per_class_f1": core.per_class_f1(pred, gold),
            "policy": {"auto": n_auto / len(texts), "auto_n": n_auto,
                       "auto_accuracy": sum(r for r, a_ in zip(right, auto) if a_) / n_auto if n_auto else float("nan"),
                       "flagged_short": sum(core.too_short(t) for t in texts)},
            "coverage": core.coverage_table(conf.tolist(), right,
                                            [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, sorter.calibrated]),
            "errors": [{"text": t, "gold": g, "pred": p, "confidence": round(float(c), 3)}
                       for t, g, p, c, ok in zip(texts, gold, pred, conf, right) if not ok],
        }
    Path("results").mkdir(exist_ok=True)
    Path("results/results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = ["# Results", "",
             f"Train {len(by['train'])} requests, validation {len(by['val'])} (used only to set the review "
             f"threshold, {sorter.calibrated}), test {len(by['test'])} (same writing style as train), "
             f"hard {len(by['hard'])} (messy, multi-topic or very short, different wording).", "",
             "## Accuracy against baselines", "",
             "| Model | Test accuracy | Test macro-F1 | Hard accuracy | Hard macro-F1 |", "|---|---|---|---|---|"]
    for key, name in (("keywords", "Keyword rules (ten-minute rules)"), ("naive_bayes", "Naive Bayes (word + bigram)"),
                      ("gpt2", "Fine-tuned GPT-2 + LoRA")):
        t, h = out["splits"]["test"][key], out["splits"]["hard"][key]
        lines.append(f"| {name} | {pct(t['accuracy'])} | {t['macro_f1']:.3f} | {pct(h['accuracy'])} | {h['macro_f1']:.3f} |")
    lines += ["", "## Review threshold: how many requests can be sorted without a human", "",
              "Confidence is the probability GPT-2 gives its chosen request type (softmax over the 11 type tokens). "
              "A request below the threshold is marked NEEDS REVIEW.", "",
              "| Threshold | Test: auto-sorted | Test: accuracy of those | Hard: auto-sorted | Hard: accuracy of those |",
              "|---|---|---|---|---|"]
    for t_row, h_row in zip(out["splits"]["test"]["coverage"], out["splits"]["hard"]["coverage"]):
        mark = " (calibrated)" if t_row["threshold"] == sorter.calibrated else ""
        lines.append(f"| {t_row['threshold']}{mark} | {pct(t_row['auto'])} | {pct(t_row['auto_accuracy'])} | "
                     f"{pct(h_row['auto'])} | {pct(h_row['auto_accuracy'])} |")
    lines += ["", f"## The review policy the CLI uses (confidence >= {sorter.threshold} and at least "
              f"{core.MIN_WORDS} words)", "",
              f"The adapter's calibrated threshold is {sorter.calibrated}; the CLI raises it to a floor of "
              f"{core.THRESHOLD_FLOOR} and always sends messages under {core.MIN_WORDS} words to a human.", "",
              "| Split | Calibrated threshold only: auto-sorted / accuracy | Policy: auto-sorted / accuracy | "
              "Short messages flagged |", "|---|---|---|---|"]
    for split in ("test", "hard"):
        c, p_ = out["splits"][split]["coverage"][-1], out["splits"][split]["policy"]
        lines.append(f"| {split} | {pct(c['auto'])} / {pct(c['auto_accuracy'])} | "
                     f"{pct(p_['auto'])} / {pct(p_['auto_accuracy'])} | {p_['flagged_short']} of "
                     f"{out['splits'][split]['n']} |")
    lines += ["", "## Per-type F1 for GPT-2", "", "| Type | Test F1 | Hard F1 |", "|---|---|---|"]
    for k in core.LABELS:
        lines.append(f"| {k} | {out['splits']['test']['per_class_f1'][k]} | {out['splits']['hard']['per_class_f1'][k]} |")
    lines += ["", "## Hard-set mistakes by GPT-2", "", "| Request | Truth | Predicted | Confidence |", "|---|---|---|---|"]
    for e in out["splits"]["hard"]["errors"]:
        lines.append(f"| {e['text'][:110].replace('|', '/')} | {e['gold']} | {e['pred']} | {e['confidence']} |")
    Path("results/results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
