from context_optimization.adaptive import (
    allocate_parent_indices,
    allocate_with_restarts,
    is_targeted_flip,
)
from context_optimization.schema import DecisionResult


def result() -> DecisionResult:
    return DecisionResult(
        unit_id="x",
        target_id="fake",
        selected="B",
        probabilities={"A": 0.2, "B": 0.8},
        latency_ms=0,
        model_revision="fake-v1",
        request_hash="hash",
    )


def test_allocation_is_deterministic_and_favors_larger_margins() -> None:
    indices = allocate_parent_indices(
        [-2.0, -1.0, 1.0], count=100, temperature=1.0, exploration=0.1, seed=7
    )
    assert indices == allocate_parent_indices(
        [-2.0, -1.0, 1.0], count=100, temperature=1.0, exploration=0.1, seed=7
    )
    assert indices.count(2) > indices.count(1) > indices.count(0)


def test_restart_allocation_reserves_root_slots() -> None:
    indices = allocate_with_restarts(
        [-2.0, -1.0, 1.0],
        count=8,
        temperature=1.0,
        exploration=0.1,
        restart_fraction=0.25,
        seed=7,
    )
    assert indices[:2] == [0, 0]
    assert len(indices) == 8


def test_targeted_flip_can_require_probability_threshold() -> None:
    assert is_targeted_flip(result(), "B")
    assert is_targeted_flip(result(), "B", 0.7)
    assert not is_targeted_flip(result(), "B", 0.9)
