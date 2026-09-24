from test_optimize import AcceptingChecker, FakeProposer

from context_optimization.io import content_hash
from context_optimization.optimize import optimize_context
from context_optimization.schema import ChoiceOption, DecisionItem, DecisionResult
from context_optimization.targets import CallableTarget
from context_optimization.transfer import targeted_transfer_rate, transfer_context


def item() -> DecisionItem:
    return DecisionItem(
        item_id="transfer",
        source="Original source.",
        question="Which option?",
        choices=[ChoiceOption(key="A", text="Alpha"), ChoiceOption(key="B", text="Beta")],
        gold_keys=["A"],
    )


def source_target() -> CallableTarget:
    def callback(unit, source):
        probability = 0.8 if "contextual signal" in source else 0.2
        probabilities = {"A": 1.0 - probability, "B": probability}
        return DecisionResult(
            unit_id=unit.unit_id,
            target_id="source",
            selected=max(probabilities, key=probabilities.get),
            probabilities=probabilities,
            latency_ms=0,
            model_revision="source-v1",
            request_hash=content_hash(source),
        )

    return CallableTarget("source", callback)


def destination_target(*, clean_correct: bool = True) -> CallableTarget:
    def callback(unit, source):
        if source == unit.source:
            probability = 0.3 if clean_correct else 0.7
        else:
            probability = 0.75
        probabilities = {"A": 1.0 - probability, "B": probability}
        return DecisionResult(
            unit_id=unit.unit_id,
            target_id="destination",
            selected=max(probabilities, key=probabilities.get),
            probabilities=probabilities,
            latency_ms=0,
            model_revision="destination-v1",
            request_hash=content_hash(source),
        )

    return CallableTarget("destination", callback)


def test_frozen_context_transfers_to_matched_destination() -> None:
    source = optimize_context(
        item(),
        source_target(),
        FakeProposer(),
        AcceptingChecker(),
    )
    transferred = transfer_context(source, item(), destination_target())
    assert transferred.matched_clean_correct
    assert transferred.source_targeted_flip
    assert transferred.targeted_transfer
    assert targeted_transfer_rate([transferred]) == 1.0


def test_initially_wrong_destination_is_outside_transfer_denominator() -> None:
    source = optimize_context(
        item(),
        source_target(),
        FakeProposer(),
        AcceptingChecker(),
    )
    transferred = transfer_context(
        source,
        item(),
        destination_target(clean_correct=False),
    )
    assert not transferred.matched_clean_correct
    assert transferred.result is None
