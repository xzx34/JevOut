"""Run a minimal in-process target: uv run python examples/callable_target.py."""

import json

from context_optimization import CallableTarget, DecisionItem, DecisionResult, evaluate
from context_optimization.io import content_hash, read_jsonl


def score(unit, source):
    # Replace this synthetic distribution with probabilities from your model.
    # Return a probability for every declared branch, summing to one.
    keys = list(unit.branches)
    probabilities = {key: 1.0 / len(keys) for key in keys}
    return DecisionResult(
        unit_id=unit.unit_id,
        target_id="custom-example",
        selected=keys[0],
        probabilities=probabilities,
        latency_ms=0,
        model_revision="synthetic-example-v1",
        request_hash=content_hash({"unit": unit.unit_id, "source": source}),
    )


if __name__ == "__main__":
    items = read_jsonl("examples/toy_choices.jsonl", DecisionItem)
    records = evaluate(items, CallableTarget("custom-example", score))
    print(json.dumps([record.model_dump(mode="json") for record in records], indent=2))
