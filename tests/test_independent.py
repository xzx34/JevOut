import pytest
from test_optimize import AcceptingChecker, FakeProposer, make_item, shifting_target

from context_optimization import IndependentRootConfig, IndependentRootResult
from context_optimization.independent import sample_independent_context
from context_optimization.optimize import optimize_context
from context_optimization.schema import OptimizationConfig


def test_independent_parents_have_no_history_or_feedback():
    class RecordingProposer(FakeProposer):
        def propose(self, item, **kwargs):
            assert kwargs["current_source"] == item.source
            assert kwargs["prior_additions"] == []
            assert kwargs["optimization_feedback"] is None
            assert kwargs["target_option"] == "B"
            return super().propose(item, **kwargs)

    result = sample_independent_context(
        make_item(),
        shifting_target(),
        RecordingProposer(),
        AcceptingChecker(),
        config=IndependentRootConfig(target_budget=7, proposal_budget=14, batch_size=4),
    )
    assert result.target_calls == 7
    assert result.proposal_calls == 7
    assert result.stop_reason == "target_call_cap"
    assert all(len(attempt.additions) == 1 for attempt in result.attempts)
    assert IndependentRootResult.model_validate_json(result.model_dump_json()) == result


def test_high_probability_success_waits_for_batch_end():
    class HighProbabilityProposer(FakeProposer):
        def propose(self, item, **kwargs):
            proposal = super().propose(item, **kwargs)
            return proposal.model_copy(
                update={"sentence": proposal.sentence + " contextual signal"}
            )

    result = sample_independent_context(
        make_item(),
        shifting_target(),
        HighProbabilityProposer(),
        AcceptingChecker(),
        config=IndependentRootConfig(target_budget=8, proposal_budget=16, batch_size=4),
    )
    assert result.target_calls == 4
    assert result.threshold_reached
    assert result.stop_reason == "batch_end_threshold"


def test_rejections_do_not_consume_target_budget_and_attempt_cap_is_exact():
    class RejectingProposer(FakeProposer):
        def propose(self, item, **kwargs):
            proposal = super().propose(item, **kwargs)
            return proposal.model_copy(update={"sentence": "The correct answer is B."})

    result = sample_independent_context(
        make_item(),
        shifting_target(),
        RejectingProposer(),
        AcceptingChecker(),
        config=IndependentRootConfig(target_budget=4, proposal_budget=5, batch_size=2),
    )
    assert result.target_calls == 0
    assert result.proposal_calls == 5
    assert result.slots_completed == 3
    assert not result.targeted_flip
    assert result.stop_reason == "proposal_attempt_cap"


def test_duplicate_inputs_are_not_evaluated_twice():
    class RepeatingProposer(FakeProposer):
        def propose(self, item, **kwargs):
            proposal = super().propose(item, **kwargs)
            return proposal.model_copy(update={"sentence": "A contextual signal was recorded."})

    result = sample_independent_context(
        make_item(),
        shifting_target(),
        RepeatingProposer(),
        AcceptingChecker(),
        config=IndependentRootConfig(target_budget=4, proposal_budget=7, batch_size=2),
    )
    assert result.target_calls == 1
    assert result.proposal_calls == 7
    assert sum(attempt.rejection == "duplicate_source" for attempt in result.attempts) == 6


def test_exhausted_rejected_slot_is_refilled_with_independent_proposals():
    class InitiallyRejectingProposer(FakeProposer):
        def propose(self, item, **kwargs):
            proposal = super().propose(item, **kwargs)
            if len(self.calls) <= 2:
                return proposal.model_copy(update={"sentence": "The correct answer is B."})
            return proposal

    result = sample_independent_context(
        make_item(),
        shifting_target(),
        InitiallyRejectingProposer(),
        AcceptingChecker(),
        config=IndependentRootConfig(target_budget=3, proposal_budget=8, batch_size=16),
    )
    assert result.target_calls == 3
    assert result.proposal_calls == 5
    assert result.slots_completed == 4
    assert result.stop_reason == "target_call_cap"


def test_reference_reuses_original_unit_and_target_without_clean_evaluation():
    item = make_item()
    target = shifting_target()
    reference = optimize_context(
        item,
        target,
        FakeProposer(),
        AcceptingChecker(),
        config=OptimizationConfig(strategy="targeted_one_shot"),
    )
    original_evaluate = target.evaluate

    def augmented_only(unit, source=None):
        assert source is not None and source != item.source
        return original_evaluate(unit, source)

    target.evaluate = augmented_only
    result = sample_independent_context(
        item,
        target,
        FakeProposer(),
        AcceptingChecker(),
        reference=reference,
        config=IndependentRootConfig(target_budget=2, proposal_budget=4),
    )
    assert result.target_option == reference.target_option
    assert result.clean_result == reference.clean_result
    assert result.metadata["reused_reference"]


def test_reference_rejects_target_change():
    reference = optimize_context(
        make_item(),
        shifting_target(),
        FakeProposer(),
        AcceptingChecker(),
        config=OptimizationConfig(strategy="targeted_one_shot"),
    )
    with pytest.raises(ValueError, match="differs from the reference"):
        sample_independent_context(
            make_item(),
            shifting_target(),
            FakeProposer(),
            AcceptingChecker(),
            reference=reference,
            target_option="C",
        )


def test_revision_change_is_not_counted_as_a_failure():
    target = shifting_target()
    original_evaluate = target.evaluate

    def changed_revision(unit, source=None):
        result = original_evaluate(unit, source)
        if source is not None and source != unit.source:
            return result.model_copy(update={"model_revision": "different-revision"})
        return result

    target.evaluate = changed_revision
    with pytest.raises(ValueError, match="revision changed"):
        sample_independent_context(make_item(), target, FakeProposer(), AcceptingChecker())
