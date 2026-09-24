import json

import httpx

from context_optimization.io import content_hash
from context_optimization.schema import DecisionResult, EvaluationUnit
from context_optimization.targets import CachedTarget, CallableTarget, JevTarget


def test_jev_payload_and_probability_normalization(monkeypatch) -> None:
    monkeypatch.setenv("TEST_TYPESAFE_KEY", "not-a-secret")

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["state"] == "Source text."
        assert payload["questions"]["decision_0"]["type"] == "choice"
        return httpx.Response(
            200,
            json={
                "model": "jev-test",
                "answers": {
                    "decision_0": {
                        "choice": "A",
                        "confidence": 0.6,
                        "probabilities": {"A": 6, "B": 4},
                    }
                },
                "usage": {"input_tokens": 10},
            },
        )

    target = JevTarget(
        base_url="https://example.test/v1/systemone",
        model="jev-test",
        api_key_env="TEST_TYPESAFE_KEY",
        transport=httpx.MockTransport(handler),
    )
    unit = EvaluationUnit(
        unit_id="x",
        item_id="x",
        primitive="choice",
        source="Source text.",
        instructions="Choose.",
        branches={"A": "Alpha", "B": "Beta"},
        gold_branch="A",
        group_id="x",
    )
    result = target.evaluate(unit)
    assert result.probabilities == {"A": 0.6, "B": 0.4}
    assert result.selected == "A"
    assert result.input_tokens == 10


def test_cached_target_reuses_identical_evaluation(tmp_path) -> None:
    calls = 0

    def callback(unit, source):
        nonlocal calls
        calls += 1
        return DecisionResult(
            unit_id=unit.unit_id,
            target_id="fake",
            selected="A",
            probabilities={"A": 0.8, "B": 0.2},
            latency_ms=0,
            model_revision="fake-v1",
            request_hash=content_hash(source),
        )

    unit = EvaluationUnit(
        unit_id="x",
        item_id="x",
        primitive="choice",
        source="Source text.",
        instructions="Choose.",
        branches={"A": "Alpha", "B": "Beta"},
        gold_branch="A",
        group_id="x",
    )
    target = CachedTarget(CallableTarget("fake", callback), tmp_path)
    assert target.evaluate(unit) == target.evaluate(unit)
    assert calls == 1
