from context_optimization.evaluate import evaluate
from context_optimization.io import content_hash
from context_optimization.schema import ChoiceOption, DecisionItem, DecisionResult
from context_optimization.targets import CallableTarget


def test_evaluate_marks_initial_correctness() -> None:
    item = DecisionItem(
        item_id="x",
        source="A short source.",
        question="Which option?",
        choices=[ChoiceOption(key="A", text="Alpha"), ChoiceOption(key="B", text="Beta")],
        gold_keys=["A"],
    )

    def callback(unit, source):
        return DecisionResult(
            unit_id=unit.unit_id,
            target_id="fake",
            selected="A",
            probabilities={"A": 0.8, "B": 0.2},
            latency_ms=0,
            model_revision="fake-v1",
            request_hash=content_hash(source),
        )

    records = evaluate([item], CallableTarget("fake", callback))
    assert len(records) == 1
    assert records[0].initially_correct
