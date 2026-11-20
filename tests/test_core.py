"""Tests for app/core.py, one group per milestone. Everything runs offline on a tiny random GPT-2 with a tiny
tokenizer built inline (no downloads); the real GPT-2 is only used by `python -m app.cli train`."""
import json

import pytest
import torch

from app import core

# ===== Header: tiny offline model and data
CORPUS = ["looking for a hall in mysuru for 200 guests", "need a dj and a band", "can you cater lunch for us",
          "please send a quote for the package", "photographer for the wedding", "cancel my booking and refund",
          "decor with flowers and balloons", "the food was terrible", "plan our whole event", "hotel rooms and pickup"]


def tiny_base():
    """A fresh 2-layer random GPT-2 plus a small byte-level BPE tokenizer. Same weights every call."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(CORPUS, trainers.BpeTrainer(
        vocab_size=300, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>")
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(fast), n_layer=2, n_head=2, n_embd=32, n_positions=128))
    return model, fast


LABELS3 = ["venue_inquiry", "catering_request", "quote_pricing"]
TOY = {"venue_inquiry": ["looking for a hall in mysuru", "need a banquet hall", "hall for 200 guests"],
       "catering_request": ["can you cater lunch", "need food and lunch", "cater dinner for us"],
       "quote_pricing": ["please send a quote", "quote for the package", "send price quote"]}


def toy_rows(n=4):
    return [{"text": f"{t} {i}", "label": k, "split": "train"} for k, ts in TOY.items() for t in ts for i in range(n)]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    cfg = core.TaggerConfig(labels=LABELS3, base="tiny", r=4, max_len=32)
    out = tmp_path_factory.mktemp("run")
    run = core.train(cfg, toy_rows(), toy_rows(1), base=tiny_base, epochs=25, lr=3e-2, batch_size=8,
                     out_dir=str(out / "trainer"))
    path = out / "adapter.pt"
    core.save_adapter(run["model"], run["label_ids"], cfg, path)
    return run, cfg, path


# ===== Milestone 1: request types and dataset
def test_milestone_1_types_teams_and_tokens():
    assert len(core.LABELS) == 11 and len(set(core.LABELS)) == 11
    assert set(core.TEAMS) == set(core.LABELS)
    assert core.label_token("venue_inquiry") == "<venue_inquiry>"


def test_milestone_1_committed_dataset_is_balanced_and_clean():
    rows = core.load_requests()
    assert core.validate_dataset(rows) == []
    train = core.load_requests(splits=("train",))
    counts = core.class_counts(train)
    assert min(counts.values()) == max(counts.values()) and min(counts.values()) >= 30
    assert {r["split"] for r in rows} == {"train", "val", "test", "hard"}


def test_milestone_1_validation_catches_leaks_and_bad_labels():
    rows = [{"text": "Hall for 100 guests!", "label": "venue_inquiry", "split": "train"},
            {"text": "hall for 100 guests", "label": "venue_inquiry", "split": "test"},
            {"text": "x", "label": "nonsense", "split": "train"}]
    problems = core.validate_dataset(rows)
    assert any("repeated" in p for p in problems) and any("unknown label" in p for p in problems)


# ===== Milestone 2: extended tokenizer and model
def test_milestone_2_each_label_is_one_new_token_and_embeddings_grow():
    model, tok = tiny_base()
    before = model.get_input_embeddings().weight.shape[0]
    ids = core.add_label_tokens(model, tok, LABELS3)
    assert model.get_input_embeddings().weight.shape[0] == before + 3
    assert ids == list(range(before, before + 3))
    assert tok("<venue_inquiry>", add_special_tokens=False).input_ids == [ids[0]]
    assert tok.pad_token == tok.eos_token


def test_milestone_2_new_rows_start_as_the_mean_of_the_label_words():
    model, tok = tiny_base()
    weight = model.get_input_embeddings().weight.detach().clone()
    ids = core.add_label_tokens(model, tok, LABELS3)
    words = tok(core.label_words("catering_request"), add_special_tokens=False).input_ids
    assert torch.allclose(model.get_input_embeddings().weight[ids[1]], weight[words].mean(dim=0), atol=1e-6)


def test_milestone_2_only_label_rows_get_gradient():
    model, tok = tiny_base()
    ids = core.add_label_tokens(model, tok, LABELS3)
    core.train_only_label_rows(model, ids)
    enc = core.RequestEncoder(tok, ids, 32)
    ex = torch.tensor([enc.training_examples(["a hall"], [0])[0]])
    labels = torch.full_like(ex, -100)
    labels[0, -1] = ex[0, -1]                       # score only the label position
    model(input_ids=ex, labels=labels).loss.backward()
    grad = model.get_input_embeddings().weight.grad
    assert grad[ids].abs().sum() > 0 and grad[:ids[0]].abs().sum() == 0


def test_milestone_2_encoder_puts_the_label_last_and_truncates_long_requests():
    model, tok = tiny_base()
    ids = core.add_label_tokens(model, tok, LABELS3)
    enc = core.RequestEncoder(tok, ids, 16)
    ex = enc.training_examples(["hall " * 50, "hall"], [2, 0])
    assert ex[0][-1] == ids[2] and ex[1][-1] == ids[0] and len(ex[0]) <= 16


# ===== Milestone 3: LoRA, Trainer, adapter file
def test_milestone_3_lora_starts_as_the_pretrained_model_and_merges_exactly():
    model, tok = tiny_base()
    x = torch.randint(0, 100, (2, 7))
    model.eval()
    ref = model(x).logits
    core.inject_lora(model, "c_attn", r=4)
    assert torch.allclose(model(x).logits, ref, atol=1e-6)            # B = 0: no change yet
    for m in model.modules():
        if isinstance(m, core.LoRALinear):
            m.B.data.normal_(0, 0.1)
    unmerged = model(x).logits
    assert core.merge_lora(model) == 2
    assert torch.allclose(model(x).logits, unmerged, atol=1e-4)


def test_milestone_3_only_lora_and_label_rows_train():
    model, tok = tiny_base()
    ids = core.add_label_tokens(model, tok, LABELS3)
    core.inject_lora(model, r=4)
    core.train_only_label_rows(model, ids)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all(n.endswith((".A", ".B")) or "wte" in n for n in names) and len(names) == 2 * 2 + 1


def test_milestone_3_trainer_learns_the_toy_set(trained):
    run, cfg, _ = trained
    assert run["report"]["final_train_loss"] < 1.0
    acc = core.accuracy_of(run["model"], run["encoder"], toy_rows(1), cfg.labels, run["tokenizer"].pad_token_id)
    assert acc >= 0.9 and run["report"]["val_accuracy"] >= 0.9


def test_milestone_3_adapter_round_trip_gives_identical_scores(trained):
    run, cfg, path = trained
    model, tok, ids, cfg2 = core.load_adapter(path, tiny_base)
    texts = ["hall for 200 guests", "send me a quote"]
    a = core.label_logits(run["model"], run["encoder"].prompts(texts), run["label_ids"], tok.pad_token_id)
    b = core.label_logits(model, core.RequestEncoder(tok, ids, 32).prompts(texts), ids, tok.pad_token_id)
    assert torch.allclose(a, b, atol=1e-5) and cfg2.labels == LABELS3


# ===== Milestone 4: inference, triage, details
@pytest.fixture(scope="module")
def sorter(trained):
    return core.Sorter(trained[2], base=tiny_base, threshold=0.5)


def test_milestone_4_sorter_labels_requests_with_confidence_team_and_runner_up(sorter):
    recs = sorter.classify(["looking for a hall in mysuru 1", "please send a quote 2"])
    assert [r["label"] for r in recs] == ["venue_inquiry", "quote_pricing"]
    r = recs[0]
    assert 0 < r["confidence"] <= 1 and r["runner_up"] != r["label"] and r["team"] == "Venues desk"


def test_milestone_4_probabilities_sum_to_one_and_threshold_flags_review(sorter):
    p = sorter.probs(["hall", "cater"])
    assert p.shape == (2, 3) and abs(p.sum(axis=1) - 1).max() < 1e-5
    flags = core.decide(p, LABELS3, 1.01)
    assert all(f["needs_review"] for f in flags) and not any(f["needs_review"] for f in core.decide(p, LABELS3, 0.0))


def test_milestone_4_calibration_picks_lowest_safe_threshold():
    conf = [0.99] * 10 + [0.7] * 10 + [0.5] * 10
    right = [True] * 10 + [True] * 9 + [False] + [False] * 10
    assert core.calibrate_threshold(conf, right, target=0.95) == 0.7
    assert core.calibrate_threshold([0.6] * 12, [False] * 12, fallback=0.9) == 0.9


@pytest.mark.parametrize("text,expected", [
    ("Need a lawn in Mysuru for 200 guests on 12 Dec, budget Rs 5 lakh",
     {"guests": 200, "city": "Mysuru", "date": "12 Dec", "budget_value": 500000.0, "currency": "INR"}),
    ("wedding on 15/02/2027 around 1,200 pax, budget $3k", {"guests": 1200, "date": "15/02/2027", "budget_value": 3000.0, "currency": "USD"}),
    ("dj for next saturday in goa", {"date": "next saturday", "city": "goa"}),
    ("may I ask about decor?", {"urgent": False}),
])
def test_milestone_4_details_are_extracted(text, expected):
    d = core.extract_details(text)
    for k, v in expected.items():
        assert d[k] == v or str(d[k]).lower() == str(v).lower()
    assert "date" not in core.extract_details("may I ask about decor?")


def test_milestone_4_urgency_and_priority():
    assert core.extract_details("need a caterer ASAP for tomorrow")["urgent"] is True
    assert core.priority_of("venue_inquiry", {"urgent": True}) == "high"
    assert core.priority_of("complaint_feedback", {"urgent": False}) == "high"
    assert core.priority_of("other", {"urgent": False}) == "low"
    assert core.priority_of("decor_design", {"urgent": False}) == "normal"


def test_milestone_4_cli_classifies_and_prints(monkeypatch, capsys, sorter):
    from app import cli
    monkeypatch.setattr(cli, "get_sorter", lambda a: sorter)
    cli.main(["classify", "looking for a hall in mysuru 1", "--no-save"])
    out = capsys.readouterr().out
    assert "venue_inquiry" in out and "Venues desk" in out and "saved" not in out


# ===== Milestone 5: saving, review loop, digest
def _records(sorter):
    return sorter.classify(["looking for a hall in mysuru 1", "please send a quote 2", "cater lunch 3 urgent"])


def test_milestone_5_save_jsonl_appends_and_reads_back(sorter, tmp_path):
    recs = _records(sorter)
    path = tmp_path / "out" / "c.jsonl"
    assert core.save_records(recs[:2], path) == 2 and core.save_records(recs[2:], path) == 1
    back = core.load_records(path)
    assert [r["text"] for r in back] == [r["text"] for r in recs] and back[0]["details"]["urgent"] is False


def test_milestone_5_save_csv_has_one_header_and_flat_details(sorter, tmp_path):
    path = tmp_path / "c.csv"
    core.save_records(_records(sorter), path)
    core.save_records(_records(sorter), path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("received,text,label") and len(lines) == 7


def test_milestone_5_cli_saves_a_batch_file_and_a_digest(monkeypatch, capsys, sorter, tmp_path):
    from app import cli
    monkeypatch.setattr(cli, "get_sorter", lambda a: sorter)
    src = tmp_path / "in.txt"
    src.write_text("hall for 200 guests 1\n\nsend price quote 2\n", encoding="utf-8")
    out, digest = tmp_path / "o.jsonl", tmp_path / "d.md"
    cli.main(["batch", str(src), "--out", str(out), "--digest", str(digest)])
    assert len(core.load_records(out)) == 2 and "Request digest (2 requests)" in digest.read_text(encoding="utf-8")
    assert "2 requests" in capsys.readouterr().out


def test_milestone_5_read_requests_handles_txt_csv_jsonl(tmp_path):
    (tmp_path / "a.csv").write_text("id,text\n1,hello hall\n2,\n3,send quote\n", encoding="utf-8")
    (tmp_path / "a.jsonl").write_text('{"text": "one"}\n{"text": "two"}\n', encoding="utf-8")
    assert core.read_requests(tmp_path / "a.csv") == ["hello hall", "send quote"]
    assert core.read_requests(tmp_path / "a.jsonl") == ["one", "two"]


def test_milestone_5_review_loop_writes_answers_and_corrections(tmp_path):
    recs = [{"text": "t1", "label": "other", "confidence": 0.4, "needs_review": True, "runner_up": "decor_design",
             "team": core.TEAMS["other"], "priority": "low", "details": {"urgent": False}},
            {"text": "t2", "label": "other", "confidence": 0.3, "needs_review": True, "runner_up": "decor_design",
             "team": core.TEAMS["other"], "priority": "low", "details": {"urgent": False}},
            {"text": "t3", "label": "other", "confidence": 0.99, "needs_review": False, "runner_up": "decor_design",
             "team": core.TEAMS["other"], "priority": "low", "details": {"urgent": False}}]
    f, corr = tmp_path / "c.jsonl", tmp_path / "corr.jsonl"
    core.save_records(recs, f)
    answers = iter(["3", "s"])
    result = core.review(f, corr, ask=lambda _: next(answers), say=lambda *_: None)
    assert result == {"reviewed": 1, "corrected": 1, "left": 1}
    saved = core.load_records(f)
    assert saved[0]["reviewed_label"] == "decor_design" and "reviewed_label" not in saved[1]
    fix = core.load_requests(corr)
    assert fix == [{"text": "t1", "label": "decor_design", "split": "train", "source": "review"}]


def test_milestone_5_digest_groups_by_team_and_lists_urgent_first(sorter):
    recs = _records(sorter)
    text = core.build_digest(recs)
    assert text.index("## High priority") < text.index("## By team") < text.index("## Needs a human")
    assert "cater lunch 3 urgent" in text.split("## By team")[0]


