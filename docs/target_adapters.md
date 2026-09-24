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

For another production service, implement the `DecisionTarget` protocol:
`target_id`, `evaluate(unit, source=None)`, and `evaluate_many(units)`.
