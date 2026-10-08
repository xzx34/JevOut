"""Target-aware proposals drawn independently from the original input."""

from __future__ import annotations

import math

from .adaptive import is_targeted_flip, make_candidate
from .checker import ContextChecker
from .io import content_hash
from .metrics import target_margin
from .optimize import _extend_candidate, _select_eligible_decision
from .proposer import ContextProposer
from .render import prepare_item, render_additions
from .schema import (
    DecisionItem,
    IndependentRootConfig,
    IndependentRootResult,
    OptimizationAttempt,
    OptimizationConfig,
    OptimizationResult,
    item_to_units,
)
from .targets import DecisionTarget


def sample_independent_context(
    item: DecisionItem,
    target: DecisionTarget,
    proposer: ContextProposer,
    checker: ContextChecker,
    *,
    config: IndependentRootConfig | None = None,
    target_option: str | None = None,
    reference: OptimizationResult | None = None,
) -> IndependentRootResult:
    """Evaluate independent single additions with separate call and proposal caps.

    With ``reference``, reuse the original selected unit, clean distribution,
    and wrong target instead of running a new eligibility selection.
    """
    config = config or IndependentRootConfig()
    prepared = prepare_item(item)
    if reference is None:
        eligible = _select_eligible_decision(prepared, target, target_option)
        unit = eligible.record.unit
        clean = eligible.record.result
        target_option = eligible.target_option
        target_branch = eligible.target_branch
    else:
        units = {unit.unit_id: unit for unit in item_to_units(prepared)}
        unit = units.get(reference.unit_id)
        if reference.item_id != prepared.item_id or unit is None:
            raise ValueError("reference and input must identify the same decision unit")
        if reference.target_id != target.target_id:
            raise ValueError("reference and baseline must use the same target system")
        if target_option is not None and target_option != reference.target_option:
            raise ValueError("requested target differs from the reference target")
        clean = reference.clean_result
        target_option = reference.target_option
        target_branch = reference.target_branch
        expected_option = (
            target_branch if unit.primitive == "choice" else unit.metadata["choice_key"]
        )
        if (
            clean.unit_id != unit.unit_id
            or clean.target_id != target.target_id
            or clean.selected != unit.gold_branch
            or set(clean.probabilities) != set(unit.branches)
            or target_branch not in unit.branches
            or target_branch == unit.gold_branch
            or target_option != expected_option
        ):
            raise ValueError("reference must preserve the initially correct unit and wrong target")

    root_margin = target_margin(clean, target_branch)
    root = make_candidate(
        item_id=prepared.item_id,
        additions=(),
        result=clean,
        margin=root_margin,
    )
    attempts: list[OptimizationAttempt] = []
    seen_sources = {content_hash(prepared.source)}
    target_calls = 0
    slots = 0
    threshold_reached = False
    while target_calls < config.target_budget and len(attempts) < config.proposal_budget:
        # Reuse the public acceptance/rendering path, always with the empty root.
        # The one-shot strategy exposes neither previous proposals nor feedback.
        slot_config = OptimizationConfig(
            strategy="targeted_one_shot",
            proposal_retries=min(
                config.proposal_retries, config.proposal_budget - len(attempts) - 1
            ),
            seed=config.seed,
            success_threshold=config.success_threshold,
        )
        child, slot_attempts = _extend_candidate(
            item=prepared,
            unit=unit,
            parent=root,
            target_option=target_option,
            target_branch=target_branch,
            target=target,
            proposer=proposer,
            checker=checker,
            config=slot_config,
            root_margin=root_margin,
            clean_target_probability=clean.probabilities[target_branch],
            iteration=slots // config.batch_size + 1,
            slot=slots % config.batch_size,
            target_call_start=target_calls,
            seen_sources=seen_sources,
        )
        attempts.extend(slot_attempts)
        slots += 1
        if child is not None:
            if len(child.additions) != 1:
                raise AssertionError("independent proposals must contain exactly one addition")
            if child.result.model_revision != clean.model_revision:
                raise ValueError("target model revision changed during the baseline")
            target_calls += 1
            threshold_reached |= is_targeted_flip(
                child.result, target_branch, config.success_threshold
            )
        if slots % config.batch_size == 0 and threshold_reached:
            stop_reason = "batch_end_threshold"
            break
        if target_calls == config.target_budget:
            stop_reason = "target_call_cap"
            break
        if len(attempts) == config.proposal_budget:
            stop_reason = "proposal_attempt_cap"
            break

    evaluated = [attempt for attempt in attempts if attempt.result is not None]
    best = max(
        evaluated,
        key=lambda attempt: attempt.margin if attempt.margin is not None else -math.inf,
        default=None,
    )
    return IndependentRootResult(
        item_id=prepared.item_id,
        unit_id=unit.unit_id,
        target_id=target.target_id,
        target_option=target_option,
        target_branch=target_branch,
        clean_result=clean,
        root_margin=root_margin,
        targeted_flip=any(attempt.targeted_flip for attempt in evaluated),
        threshold_reached=threshold_reached,
        success_threshold=config.success_threshold,
        iterations_completed=math.ceil(slots / config.batch_size),
        proposal_calls=len(attempts),
        target_calls=target_calls,
        attempts=attempts,
        best_attempt_id=best.attempt_id if best else None,
        best_source=render_additions(prepared, best.additions) if best else None,
        best_additions=best.additions if best else [],
        best_result=best.result if best else None,
        best_margin=best.margin if best else None,
        config=config,
        slots_completed=slots,
        stop_reason=stop_reason,
        metadata={"input_hash": content_hash(prepared), "reused_reference": reference is not None},
    )
