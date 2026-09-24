from __future__ import annotations

import math
from collections.abc import Iterable

from .schema import DecisionResult, EvaluationUnit


def target_margin(result: DecisionResult, target_option: str) -> float:
    if target_option not in result.probabilities:
        raise ValueError(f"unknown target option: {target_option}")
    competitors = [
        probability
        for key, probability in result.probabilities.items()
        if key != target_option
    ]
    if not competitors:
        raise ValueError("a target margin requires at least two options")
    epsilon = 1e-12
    return math.log(result.probabilities[target_option] + epsilon) - math.log(
        max(competitors) + epsilon
    )


def select_target_option(unit: EvaluationUnit, clean_result: DecisionResult) -> str:
    return max(
        (branch for branch in clean_result.probabilities if branch != unit.gold_branch),
        key=clean_result.probabilities.get,
    )


def targeted_flip_rate(outcomes: Iterable[bool]) -> float:
    values = list(outcomes)
    if not values:
        raise ValueError("targeted flip rate requires at least one eligible decision")
    return sum(values) / len(values)
