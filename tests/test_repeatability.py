import pytest
from test_optimize import AcceptingChecker, FakeProposer, make_item, shifting_target

from context_optimization import CachedTarget, OptimizationConfig, optimize_context, retest_context


def discovery():
    return optimize_context(
        make_item(),
        shifting_target(),
        FakeProposer(),
        AcceptingChecker(),
        config=OptimizationConfig(particles=3, rounds=2, proposal_retries=0),
    )


def test_retest_bypasses_nested_caches_and_preserves_all_observations(tmp_path):
    source = discovery()
    target = shifting_target()
    calls = []
    evaluate = target.evaluate

    def recording(unit, source=None):
        calls.append(source)
        return evaluate(unit, source)

    target.evaluate = recording
    cached = CachedTarget(CachedTarget(target, tmp_path / "inner"), tmp_path / "outer")
    result = retest_context(source, make_item(), cached)
    assert len(calls) == 20
    assert len(set(calls)) == 2
    assert result.target_hits == 10
    assert result.clean_gold_hits == 10
    assert len(result.clean_results) == len(result.augmented_results) == 10
    assert result.clean_and_flip_confirmed
    assert not list(tmp_path.rglob("*.json"))


def test_retest_retains_unsuccessful_repeats_in_the_denominator():
    source = discovery()
    target = shifting_target()
    evaluate = target.evaluate
    augmented_calls = 0

    def variable(unit, source=None):
        nonlocal augmented_calls
        result = evaluate(unit, source)
        if source != unit.source:
            augmented_calls += 1
            if augmented_calls > 7:
                return evaluate(unit, unit.source)
        return result

    target.evaluate = variable
    result = retest_context(source, make_item(), target)
    assert result.target_hits == 7
    assert result.repetitions == 10
    assert not result.flip_confirmed
    assert not result.clean_and_flip_confirmed


def test_context_is_selected_before_retesting_without_fallback():
    source = discovery()
    target = shifting_target()
    evaluate = target.evaluate
    repeated_sources = []

    def never_flip(unit, source=None):
        if source != unit.source:
            repeated_sources.append(source)
        return evaluate(unit, unit.source)

    target.evaluate = never_flip
    result = retest_context(source, make_item(), target)
    assert len(set(repeated_sources)) == 1
    assert result.target_hits == 0
    assert result.source_attempt_id in {attempt.attempt_id for attempt in source.attempts}


def test_changed_context_is_rejected_before_any_calls():
    source = discovery()
    for attempt in source.attempts:
        if attempt.targeted_flip:
            attempt.source_hash = "tampered"
    with pytest.raises(ValueError, match="differs from the recorded"):
        retest_context(source, make_item(), shifting_target())


def test_failed_requests_are_not_scored_as_model_errors():
    target = shifting_target()

    def failed(unit, source=None):
        raise TimeoutError("service unavailable")

    target.evaluate = failed
    with pytest.raises(TimeoutError):
        retest_context(discovery(), make_item(), target)


@pytest.mark.parametrize("repetitions,minimum_hits", [(0, 0), (10, 11), (10, 0)])
def test_invalid_threshold_is_rejected(repetitions, minimum_hits):
    with pytest.raises(ValueError, match="minimum_hits"):
        retest_context(
            discovery(),
            make_item(),
            shifting_target(),
            repetitions=repetitions,
            minimum_hits=minimum_hits,
        )
