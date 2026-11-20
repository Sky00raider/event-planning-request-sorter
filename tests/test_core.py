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


