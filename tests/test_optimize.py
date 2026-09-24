import pytest

from context_optimization.checker import ContextChecker
from context_optimization.io import content_hash
from context_optimization.optimize import IneligibleDecisionError, optimize_context
from context_optimization.proposer import ContextProposer, deterministic_filters
from context_optimization.render import derive_boundaries
from context_optimization.schema import (
    CheckResult,
    ChoiceOption,
    DecisionItem,
    DecisionResult,
    OptimizationConfig,
    Proposal,
)
from context_optimization.targets import CallableTarget


class FakeProposer(ContextProposer):
    def __init__(self) -> None:
        self.calls = []

    def propose(
        self,
        item,
        *,
        target_option,
        seed,
        current_source,
        prior_additions,
        optimization_feedback,
        neutral=False,
    ):
        self.calls.append(
            {
                "target_option": target_option,
                "feedback": optimization_feedback,
                "neutral": neutral,
            }
        )
        sentence = f"A contextual signal numbered {seed} was recorded."
        boundary_id = derive_boundaries(item)[0].id
        return Proposal(
            sentence=sentence,
            boundary_id=boundary_id,
            generator_revision="fake-proposer",
            prompt_hash="prompt",
            filter_results=deterministic_filters(
                item,
                sentence,
                boundary_id,
                prior_additions=prior_additions,
            ),
        )


class AcceptingChecker(ContextChecker):
    def check(self, item, proposal, *, prior_additions):
        return CheckResult(
            label_preserved=True,
            locally_coherent=True,
            direct_decision_cue=False,
            new_decisive_evidence=False,
            reason="The correct answer remains unchanged.",
        )


def make_item() -> DecisionItem:
    return DecisionItem(
        item_id="x",
        source="Original source.",
        question="Which option?",
        choices=[
            ChoiceOption(key="A", text="Alpha"),
            ChoiceOption(key="B", text="Beta"),
            ChoiceOption(key="C", text="Gamma"),
        ],
        gold_keys=["A"],
    )


def shifting_target() -> CallableTarget:
    def callback(unit, source):
        additions = source.count("contextual signal")
        target_probability = min(0.2 + 0.3 * additions, 0.95)
        remainder = 1.0 - target_probability
        probabilities = {"A": remainder * 0.75, "B": target_probability, "C": remainder * 0.25}
        selected = max(probabilities, key=probabilities.get)
        return DecisionResult(
            unit_id=unit.unit_id,
            target_id="fake",
            selected=selected,
            probabilities=probabilities,
            latency_ms=0,
            model_revision="fake-v1",
            request_hash=content_hash(source),
        )

    return CallableTarget("fake", callback)


def test_adaptive_optimization_uncovers_targeted_flip() -> None:
    proposer = FakeProposer()
    result = optimize_context(
        make_item(),
        shifting_target(),
        proposer,
        AcceptingChecker(),
        config=OptimizationConfig(
            particles=3,
            rounds=2,
            proposal_retries=0,
            success_threshold=0.7,
        ),
    )
    assert result.target_option == "B"
    assert result.targeted_flip
    assert result.threshold_reached
    assert result.target_calls == 6
    assert result.best_result is not None
    assert result.best_result.probabilities["B"] == pytest.approx(0.8)
    assert len(result.best_additions) == 2
    assert any(call["feedback"] is not None for call in proposer.calls)


def test_neutral_one_shot_hides_target_from_proposer() -> None:
    proposer = FakeProposer()
    result = optimize_context(
        make_item(),
        shifting_target(),
        proposer,
        AcceptingChecker(),
        config=OptimizationConfig(strategy="neutral_one_shot", proposal_retries=0),
    )
    assert result.target_calls == 1
    assert proposer.calls == [{"target_option": None, "feedback": None, "neutral": True}]


def test_initially_wrong_item_is_ineligible() -> None:
    def callback(unit, source):
        return DecisionResult(
            unit_id=unit.unit_id,
            target_id="wrong",
            selected="B",
            probabilities={"A": 0.1, "B": 0.8, "C": 0.1},
            latency_ms=0,
            model_revision="wrong-v1",
            request_hash=content_hash(source),
        )

    with pytest.raises(IneligibleDecisionError):
        optimize_context(
            make_item(),
            CallableTarget("wrong", callback),
            FakeProposer(),
            AcceptingChecker(),
        )


def test_multi_answer_selects_one_closest_absent_option() -> None:
    item = DecisionItem(
        item_id="multi",
        source="Original source.",
        question="Which options apply?",
        choices=[
            ChoiceOption(key="A", text="Alpha"),
            ChoiceOption(key="B", text="Beta"),
            ChoiceOption(key="C", text="Gamma"),
            ChoiceOption(key="D", text="Delta"),
        ],
        gold_keys=["A", "C"],
    )

    def callback(unit, source):
        key = unit.metadata["choice_key"]
        if key in {"A", "C"}:
            probability_true = 0.8
        else:
            clean = {"B": 0.4, "D": 0.3}[key]
            probability_true = 0.8 if "contextual signal" in source else clean
        probabilities = {"false": 1.0 - probability_true, "true": probability_true}
        return DecisionResult(
            unit_id=unit.unit_id,
            target_id="membership",
            selected=max(probabilities, key=probabilities.get),
            probabilities=probabilities,
            latency_ms=0,
            model_revision="membership-v1",
            request_hash=content_hash({"key": key, "source": source}),
        )

    result = optimize_context(
        item,
        CallableTarget("membership", callback),
        FakeProposer(),
        AcceptingChecker(),
        config=OptimizationConfig(
            strategy="targeted_one_shot",
            proposal_retries=0,
        ),
    )
    assert result.target_option == "B"
    assert result.unit_id == "multi:B"
    assert result.target_branch == "true"
    assert result.targeted_flip


def test_optimizer_reapplies_filters_to_custom_proposals() -> None:
    class DirectCueProposer:
        def propose(self, item, **kwargs):
            return Proposal(
                sentence="The correct answer is B.",
                boundary_id=derive_boundaries(item)[0].id,
                generator_revision="custom",
                prompt_hash="custom",
            )

    result = optimize_context(
        make_item(),
        shifting_target(),
        DirectCueProposer(),
        AcceptingChecker(),
        config=OptimizationConfig(
            strategy="targeted_one_shot",
            proposal_retries=0,
        ),
    )
    assert result.target_calls == 0
    assert result.attempts[0].rejection == "deterministic_filter"
