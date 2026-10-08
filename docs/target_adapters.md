# Target adapters

A decision target maps an evaluation unit and its source text to a normalized
probability distribution. The package includes Jev, generic HTTP, cached, and
callable adapters.

## Jev

`JevTarget` reads `TYPESAFE_API_KEY`. `TYPESAFE_BASE_URL` and `TYPESAFE_MODEL`
may override the default endpoint and model revision.

```python
from context_optimization import JevTarget

target = JevTarget()
```

Credentials are read at runtime and are never serialized into results.

## Generic HTTP target

`HttpDecisionTarget` sends `POST <base_url>/evaluate` with:

```json
{"unit": {"...": "EvaluationUnit fields"}, "source": "effective source text"}
```

The response must include `selected`, `probabilities`, and `model_revision`.
Probabilities must be normalized, and `selected` must be an argmax. The adapter
adds unit, target, latency, and request-hash fields.

```python
from context_optimization import HttpDecisionTarget

target = HttpDecisionTarget("local-model", "http://127.0.0.1:9000")
```

## Python target

Use `CallableTarget` for an in-process model or custom integration. The callback
receives an `EvaluationUnit` and the effective source text and returns a
`DecisionResult`.

```python
from context_optimization import CallableTarget

target = CallableTarget("my-target", evaluate_unit)
```

Wrap any adapter with `CachedTarget(target, cache_dir)` to cache results by the
complete target, unit, and source payload. Cache directories are selected by
the caller and should not be committed.

Run the complete synthetic callback example with:

```bash
uv run python examples/callable_target.py
```

Replace its uniform distribution with your model's probabilities to connect an
in-process target. Return one probability for every branch in `unit.branches`,
and choose a branch attaining the maximum. The `source` argument is the actual
text to score, including additions when supplied; do not always score
`unit.source`. Use a stable `target_id` and report the checkpoint or service
revision in every result.

For a local scoring server, `HttpDecisionTarget` sends the same effective
source and unit to your `/evaluate` endpoint. The response can be a
`DecisionResult` JSON object; the client fills the request hash, unit identifier,
target identifier, and latency. The server must provide the selected branch,
complete probability distribution, and model revision.

`retest_context` bypasses nested `CachedTarget` wrappers and checks the model
revision against discovery. Custom adapters used for repeatability should
perform a new provider/model invocation on each call.

For another production service, implement the `DecisionTarget` protocol:
`target_id`, `evaluate(unit, source=None)`, and `evaluate_many(units)`.
