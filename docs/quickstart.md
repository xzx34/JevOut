# Quickstart

## Install and run offline

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/xzx34/JevOut.git
cd JevOut
uv sync --frozen
uv run context-opt demo --output-dir outputs/demo
```

The deterministic synthetic demo needs no model service or credentials. It
writes original items, clean predictions, optimized contexts, independent
contexts, and repeated predictions as JSONL. Its probabilities illustrate the
interfaces; empirical results are in [results.md](results.md).

## Prepare input

```bash
uv run context-opt validate \
  --input examples/toy_choices.jsonl --output outputs/validated.jsonl
```

Validation checks answer keys and insertion boundaries. Each item needs a
unique identifier, source, question, choices, and gold keys; see the
[input format](input_format.md). Multi-answer items expand into binary
membership units, and construction selects at most one eligible unit per item.

## Connect models

Set `TYPESAFE_API_KEY` in the environment, or in an untracked local `.env` file.
The Jev adapter defaults to `jev-1.13.0`; `TYPESAFE_MODEL` and
`TYPESAFE_BASE_URL` can select another supported revision or endpoint.

Start an OpenAI-compatible service for the proposer and checker. Its
`--proposer-model` value must match a model served by that endpoint. A separate
`--checker-model` is optional; by default the same model performs both roles in
separate requests. If authentication is required, pass an environment-variable
name with `--proposer-api-key-env`.

For an HTTP target, replace `--target jev` in the following commands with
`--target http --target-id my-model --target-url http://127.0.0.1:9000`.
The [adapter guide](target_adapters.md) specifies that service's request and
response format and includes a Python callback example.

## Run a small optimization

First evaluate the original decisions:

```bash
uv run context-opt evaluate \
  --input outputs/validated.jsonl --output outputs/clean.jsonl \
  --target jev --cache-dir .cache/context-opt
```

Then try four candidates over two iterations:

```bash
uv run context-opt optimize \
  --input outputs/validated.jsonl --output outputs/optimized-small.jsonl \
  --target jev \
  --proposer-url http://127.0.0.1:8000/v1 --proposer-model your-model \
  --particles 4 --rounds 2 --cache-dir .cache/context-opt
```

This permits at most eight accepted target evaluations per eligible item,
plus eligibility evaluation. The default `--particles 16 --rounds 4` permits
64. The JSON summary reports eligible decisions, flips, TFR, proposal attempts,
and accepted calls. Items answered incorrectly on the original input are
reported as skipped. Acceptance rejections remain unsuccessful outcomes within
the eligible population. API failures raise errors instead of becoming model
failures; completed construction records are saved after each item.

## One-shot controls

Use the same `optimize` command with `--strategy targeted_one_shot` or
`--strategy neutral_one_shot`, and `--proposal-retries 0` to match the paper's
single-proposal controls. The neutral proposer does not receive the wrong target;
both controls are evaluated against a target fixed from the clean distribution.
The shared CLI retry default is one, as in the adaptive optimizer.

## Independent-root baseline

For a matched comparison, use a saved optimization file to retain its original
unit, clean distribution, and wrong target:

```bash
uv run context-opt optimize \
  --input decisions.jsonl --output outputs/independent.jsonl \
  --strategy independent_root --reference outputs/optimized.jsonl \
  --target jev \
  --proposer-url http://127.0.0.1:8000/v1 --proposer-model your-model \
  --target-budget 64 --proposal-budget 128 --batch-size 16 \
  --cache-dir .cache/independent
```

The original input must contain every item in the reference file. Extra input
items are skipped, so you can reuse the full original dataset while keeping
exactly the reference's eligible population. Every proposal starts from the original input, contains one
addition, and sees neither previous proposals nor target feedback. The control
refills rejected slots while the call and proposal budgets permit. It checks
the stopping threshold after each block of 16 slots. The complete protocol is
in [experiment settings](paper_protocol.md).

Omit `--reference` to select eligible units and targets from new clean
evaluations. Use a separate cache directory for independent runs.

## Retest a saved context

```bash
uv run context-opt retest \
  --input decisions.jsonl --contexts outputs/optimized.jsonl \
  --output outputs/repeatability.jsonl --target jev \
  --repetitions 10 --minimum-hits 8
```

The command retests successful records in the supplied context file. Supply a
preselected subset of records to audit a sample. It selects the recorded
successful context with the highest target probability before making any new
calls, then evaluates that exact context and the original input ten times each.
It has no local cache option, and the Python API unwraps `CachedTarget` wrappers.
The response revision must match discovery. Each case records all predictions,
target hits, clean gold hits, and both confirmation criteria; see
[output format](output_format.md). There is no fallback to a different context
after observing repeat outcomes.

## Interpret the outputs

TFR is successful eligible decisions divided by all eligible decisions under
the specified discovery budget. A wrong choice other than the fixed target
does not count. A stopping probability of 0.7 is separate from success: any
selection of the fixed target counts as a flip. Repeated evaluations assess
an already discovered context and do not replace discovery TFR.
