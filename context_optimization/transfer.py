from __future__ import annotations

from .metrics import target_margin
from .render import prepare_item, render_additions
from .schema import (
    DecisionItem,
    OptimizationAttempt,
    OptimizationResult,
    TransferResult,
    item_to_units,
)
from .targets import DecisionTarget


def select_representative_attempt(
    result: OptimizationResult,
) -> OptimizationAttempt | None:
    evaluated = [attempt for attempt in result.attempts if attempt.result is not None]
    successful = [attempt for attempt in evaluated if attempt.targeted_flip]
    if successful:
        return min(
            successful,
            key=lambda attempt: (
                (attempt.target_probability or 0.0) < result.success_threshold,
                len(attempt.additions),
                -(attempt.target_probability or 0.0),
                attempt.target_call_index or 0,
            ),
        )
    return max(
        evaluated,
        key=lambda attempt: (
            attempt.margin if attempt.margin is not None else float("-inf"),
            -len(attempt.additions),
            -(attempt.target_call_index or 0),
        ),
        default=None,
    )


def transfer_context(
    source: OptimizationResult,
    item: DecisionItem,
    destination: DecisionTarget,
) -> TransferResult:
    prepared = prepare_item(item)
    if source.item_id != prepared.item_id:
        raise ValueError("source result and decision item do not match")
    units = {unit.unit_id: unit for unit in item_to_units(prepared)}
    unit = units.get(source.unit_id)
    if unit is None:
        raise ValueError("source-selected decision unit is absent from the item")
    if source.target_branch not in unit.branches:
        raise ValueError("source target branch is absent from the destination decision unit")

    clean = destination.evaluate(unit)
    attempt = select_representative_attempt(source)
    if attempt is None:
        additions = []
        selection_kind = "clean_fallback"
        source_flip = False
    else:
        additions = attempt.additions
        source_flip = attempt.targeted_flip
        selection_kind = "successful_context" if source_flip else "best_margin"

    if clean.selected != unit.gold_branch:
        return TransferResult(
            item_id=prepared.item_id,
            unit_id=unit.unit_id,
            source_target_id=source.target_id,
            destination_target_id=destination.target_id,
            target_option=source.target_option,
            target_branch=source.target_branch,
            matched_clean_correct=False,
            selection_kind=selection_kind,
            source_targeted_flip=source_flip,
            source_attempt_id=attempt.attempt_id if attempt else None,
            additions=additions,
            destination_clean_result=clean,
        )

    if destination.target_id == source.target_id:
        shifted = attempt.result if attempt is not None else source.clean_result
        assert shifted is not None
    else:
        shifted = destination.evaluate(unit, render_additions(prepared, additions))
    clean_margin = target_margin(clean, source.target_branch)
    shifted_margin = target_margin(shifted, source.target_branch)
    clean_probability = clean.probabilities[source.target_branch]
    shifted_probability = shifted.probabilities[source.target_branch]
    return TransferResult(
        item_id=prepared.item_id,
        unit_id=unit.unit_id,
        source_target_id=source.target_id,
        destination_target_id=destination.target_id,
        target_option=source.target_option,
        target_branch=source.target_branch,
        matched_clean_correct=True,
        selection_kind=selection_kind,
        source_targeted_flip=source_flip,
        source_attempt_id=attempt.attempt_id if attempt else None,
        additions=additions,
        destination_clean_result=clean,
        result=shifted,
        targeted_transfer=shifted.selected == source.target_branch,
        target_probability=shifted_probability,
        target_probability_delta=shifted_probability - clean_probability,
        margin=shifted_margin,
        margin_delta=shifted_margin - clean_margin,
    )


def targeted_transfer_rate(results: list[TransferResult]) -> float:
    matched = [result for result in results if result.matched_clean_correct]
    if not matched:
        raise ValueError("targeted transfer rate requires matched initially correct decisions")
    return sum(result.targeted_transfer for result in matched) / len(matched)
