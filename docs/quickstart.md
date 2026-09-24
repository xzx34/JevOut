# Quickstart

JevOut evaluates bounded decisions and constructs short additions that fit the
surrounding context while preserving the task, choices, and correct answer.

## 1. Install

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
git clone https://github.com/xzx34/JevOut.git
cd JevOut
uv sync --frozen
```

## 2. Validate input

The included toy file is synthetic and makes no provider calls:

```bash
uv run context-opt validate \
  --input examples/toy_choices.jsonl \
  --output validated.jsonl
```

Validation checks answer keys and insertion boundaries, then writes normalized
items when `--output` is supplied.

## 3. Evaluate clean decisions

For Jev, expose the credential only through the environment:

```bash
export TYPESAFE_API_KEY="..."
uv run context-opt evaluate \
  --input validated.jsonl \
  --output clean.jsonl \
  --target jev \
  --cache-dir .cache/context-opt
```

The output contains one clean probability distribution per decision unit.
Multi-answer items are expanded into binary membership units internally.

## 4. Optimize context

Start an OpenAI-compatible endpoint for the proposer and checker, then run:

```bash
uv run context-opt optimize \
  --input validated.jsonl \
  --output optimized.jsonl \
  --target jev \
  --proposer-url http://127.0.0.1:8000/v1 \
  --proposer-model your-model \
  --cache-dir .cache/context-opt
```

If the endpoint requires authentication, put its key in an environment variable
and pass the variable name with `--proposer-api-key-env`. The optimizer fixes a
wrong target option from the clean distribution, considers only initially
correct decisions, and records every proposal, checker decision, accepted target
evaluation, and probability margin.

The default adaptive run uses 16 candidates and 4 rounds, so each eligible
decision receives at most 64 accepted target evaluations. Use `--strategy
targeted_one_shot` or `--strategy neutral_one_shot` for the one-shot controls.
