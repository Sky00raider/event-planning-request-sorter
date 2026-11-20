# Event Planning Request Sorter

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/847m3z7jrr/badge.svg)](https://ask.thili.ai/b/847m3z7jrr)

The Event Planning Request Sorter is a tool for event planners to efficiently manage incoming client requests. It leverages a fine-tuned GPT-2 model to automatically categorize requests (e.g., "venue inquiry", "catering request"), enabling quicker task delegation and response prioritization.

## What you are building

**Goal.** You will have built a tool that automatically sorts incoming client requests for event planners, demonstrating the fine-tuning logic from the drill to classify short textual requests and the inference pipeline to make predictions.

**Done when.** You will have a command-line interface where you can input new event requests, see them classified into predefined types by your fine-tuned GPT-2 model, and save these classified requests to a file for review.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Define common event request types and create a small dataset of example requests.**  
      Hint: Define your common event request types and create a small dataset of example requests, similar to how legal clauses were prepared for tagging.
- [ ] **2. Configure GPT-2 with an extended tokenizer and model for custom tokens.**  
      Hint: Configure GPT-2 by extending its tokenizer and model to include your custom request type tokens, following the `lfllm-lflegalgpt` drill.
- [ ] **3. Train the model on the example requests using the Hugging Face Trainer.**  
      Hint: Train your model on the example requests using the Hugging Face `Trainer` to fine-tune GPT-2 for classification.
- [ ] **4. Build an interface (CLI) to input new requests and show their classification.**  
      Hint: Build a command-line interface (CLI) that uses the inference pipeline to accept new requests and display their predicted classification.
- [ ] **5. Allow saving classified requests to a file for review.**  
      Hint: Extend your CLI to save the input request along with its classification to a designated file.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- (none)
