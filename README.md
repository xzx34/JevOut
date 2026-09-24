# JevOut: Natural Context Can Flip Decision Models

This repository contains the reusable implementation accompanying the paper.
It evaluates whether short, answer-preserving context additions can redirect an
initially correct bounded decision to a target option fixed in advance.

The implementation calls this procedure **probability-guided context
optimization**. A successful outcome is a **targeted flip**, and Targeted Flip
Rate (TFR) is measured over decisions the target model initially answers
correctly under a stated target-evaluation budget.

The package provides a strict input contract, deterministic context rendering,
decision-target adapters, clean evaluation, probability-guided context
optimization, the formal one-shot controls, and cross-model transfer evaluation.

This code release does not include the manuscript source, experimental data,
paper figures, result artifacts, model weights, or proposer fine-tuning pipeline.

## Install

```bash
uv sync --frozen
```

See the [quickstart](docs/quickstart.md), [input format](docs/input_format.md),
and [target adapter guide](docs/target_adapters.md) for a complete run.

Validate JSONL without calling a model:

```bash
uv run context-opt validate --input examples/toy_choices.jsonl
```

Jev credentials are read from `TYPESAFE_API_KEY`. Never commit a populated
`.env` file.

## Core API

```python
from context_optimization import OptimizationConfig, optimize_context

result = optimize_context(
    item,
    target,
    proposer,
    checker,
    config=OptimizationConfig(particles=16, rounds=4),
)
```

The target option is fixed from the clean distribution before optimization.
Items that the target does not initially answer correctly are ineligible and
remain outside the TFR denominator.

With an OpenAI-compatible proposer service, the corresponding CLI shape is:

```bash
uv run context-opt optimize \
  --input decisions.jsonl \
  --output optimized.jsonl \
  --target jev \
  --proposer-url http://127.0.0.1:8000/v1 \
  --proposer-model your-model
```

The default adaptive configuration permits at most 64 accepted target
evaluations per eligible decision (16 candidates over 4 rounds). Rejected or
duplicate proposals do not consume that target-evaluation budget.

Cross-model evaluation is available through `transfer_context` and
`targeted_transfer_rate`. It keeps the source-selected decision unit and target
option fixed, then evaluates the frozen context on a destination target.

## Development

```bash
uv sync --frozen --extra dev
uv run pytest
uv run ruff check .
uv build --wheel
```

Licensed under Apache-2.0. Please report security issues using GitHub's private
security-advisory workflow; see [SECURITY.md](SECURITY.md).
