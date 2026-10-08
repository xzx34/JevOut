"""Fresh evaluations of a context selected before any repeat results are seen."""

from __future__ import annotations

from .io import content_hash
from .render import prepare_item, render_additions
from .schema import (
    DecisionItem,
    OptimizationResult,
    RepeatabilityResult,
    item_to_units,
)
from .targets import CachedTarget, DecisionTarget


def retest_context(
    source: OptimizationResult,
    item: DecisionItem,
    target: DecisionTarget,
    *,
    repetitions: int = 10,
    minimum_hits: int = 8,
) -> RepeatabilityResult:
    """Retest the highest-target-probability discovery success without reselection.

    Bypass all ``CachedTarget`` wrappers. Service-internal caching, if any,
    remains outside the client's control. Failed requests raise exceptions.
    """
    if not 1 <= minimum_hits <= repetitions:
        raise ValueError("require 1 <= minimum_hits <= repetitions")
    prepared = prepare_item(item)
    units = {unit.unit_id: unit for unit in item_to_units(prepared)}
    unit = units.get(source.unit_id)
    if source.item_id != prepared.item_id or unit is None:
        raise ValueError("source result and input must identify the same decision unit")
    if source.target_id != target.target_id:
        raise ValueError("repeatability must use the original target system")
    if source.clean_result.selected != unit.gold_branch:
        raise ValueError("source discovery must start from a correct decision")
    if source.target_branch not in unit.branches or source.target_branch == unit.gold_branch:
        raise ValueError("the fixed target must be an available wrong branch")
    input_hash = source.metadata.get("input_hash")
    if input_hash is not None and input_hash != content_hash(prepared):
        raise ValueError("input differs from the recorded discovery input")
    successful = [
        attempt
        for attempt in source.attempts
        if attempt.result is not None and attempt.result.selected == source.target_branch
    ]
    if not successful:
        raise ValueError("repeatability requires a recorded successful context")
    # Tie-breaking uses discovery records only, never repeat outcomes.
    chosen = max(
        successful,
        key=lambda attempt: (
            attempt.result.probabilities[source.target_branch],
            -(attempt.target_call_index or 0),
        ),
    )
    augmented = render_additions(prepared, chosen.additions)
    if chosen.source_hash != content_hash(augmented):
        raise ValueError("rendered context differs from the recorded discovery context")

    while isinstance(target, CachedTarget):
        target = target.target
    expected_revision = source.clean_result.model_revision
    clean_results = []
    augmented_results = []
    for _ in range(repetitions):
        for text, destination in (
            (prepared.source, clean_results),
            (augmented, augmented_results),
        ):
            result = target.evaluate(unit, text)
            if (
                result.unit_id != unit.unit_id
                or result.target_id != source.target_id
                or set(result.probabilities) != set(unit.branches)
                or result.model_revision != expected_revision
            ):
                raise ValueError("repeat response changed the unit, target, branches, or revision")
            destination.append(result)
    clean_hits = sum(result.selected == unit.gold_branch for result in clean_results)
    target_hits = sum(result.selected == source.target_branch for result in augmented_results)
    return RepeatabilityResult(
        item_id=prepared.item_id,
        unit_id=unit.unit_id,
        target_id=source.target_id,
        target_option=source.target_option,
        target_branch=source.target_branch,
        source_attempt_id=chosen.attempt_id,
        additions=chosen.additions,
        original_source_hash=content_hash(prepared.source),
        augmented_source_hash=content_hash(augmented),
        repetitions=repetitions,
        minimum_hits=minimum_hits,
        clean_gold_hits=clean_hits,
        target_hits=target_hits,
        flip_confirmed=target_hits >= minimum_hits,
        clean_and_flip_confirmed=(clean_hits >= minimum_hits and target_hits >= minimum_hits),
        clean_results=clean_results,
        augmented_results=augmented_results,
        model_revision=expected_revision,
    )
