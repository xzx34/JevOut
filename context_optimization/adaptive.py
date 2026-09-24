from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .io import content_hash
from .schema import ContextAddition, DecisionResult


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    additions: tuple[ContextAddition, ...]
    result: DecisionResult
    margin: float


def make_candidate(
    *,
    item_id: str,
    additions: tuple[ContextAddition, ...],
    result: DecisionResult,
    margin: float,
) -> Candidate:
    candidate_id = content_hash(
        {
            "item_id": item_id,
            "additions": [addition.model_dump(mode="json") for addition in additions],
        }
    )[:16]
    return Candidate(candidate_id, additions, result, margin)


def allocate_parent_indices(
    margins: list[float],
    *,
    count: int,
    temperature: float,
    exploration: float,
    seed: int,
) -> list[int]:
    if not margins:
        raise ValueError("margins cannot be empty")
    if count <= 0:
        raise ValueError("count must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 <= exploration <= 1:
        raise ValueError("exploration must be between zero and one")

    best = max(margins)
    unnormalized = [math.exp(max((margin - best) / temperature, -50.0)) for margin in margins]
    total = sum(unnormalized)
    adaptive = [value / total for value in unnormalized]
    uniform = 1.0 / len(margins)
    weights = [
        (1.0 - exploration) * probability + exploration * uniform
        for probability in adaptive
    ]

    rng = random.Random(seed)
    start = rng.random() / count
    positions = [start + index / count for index in range(count)]
    indices = []
    cumulative = weights[0]
    source_index = 0
    for position in positions:
        while position > cumulative and source_index < len(weights) - 1:
            source_index += 1
            cumulative += weights[source_index]
        indices.append(source_index)
    return indices


def allocate_with_restarts(
    margins: list[float],
    *,
    count: int,
    temperature: float,
    exploration: float,
    restart_fraction: float,
    seed: int,
) -> list[int]:
    if not 0 < restart_fraction < 1:
        raise ValueError("restart_fraction must be between zero and one")
    restart_count = min(max(1, round(count * restart_fraction)), count)
    adaptive_count = count - restart_count
    allocated = (
        allocate_parent_indices(
            margins,
            count=adaptive_count,
            temperature=temperature,
            exploration=exploration,
            seed=seed,
        )
        if adaptive_count
        else []
    )
    return [0] * restart_count + allocated


def proposal_seed(
    *,
    base_seed: int,
    item_id: str,
    parent_id: str,
    iteration: int,
    slot: int,
    retry: int,
) -> int:
    digest = content_hash(
        {
            "item_id": item_id,
            "parent_id": parent_id,
            "iteration": iteration,
            "slot": slot,
            "retry": retry,
        }
    )
    return base_seed + int(digest[:8], 16) % 1_000_000


def is_targeted_flip(
    result: DecisionResult,
    target_option: str,
    probability_threshold: float = 0.0,
) -> bool:
    return (
        result.selected == target_option
        and result.probabilities[target_option] >= probability_threshold
    )
