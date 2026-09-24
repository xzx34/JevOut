from __future__ import annotations

from dataclasses import dataclass

from .adaptive import (
    Candidate,
    allocate_with_restarts,
    is_targeted_flip,
    make_candidate,
    proposal_seed,
)
from .checker import ContextChecker
from .evaluate import EvaluationRecord, evaluate
from .io import content_hash
from .metrics import select_target_option, target_margin
from .proposer import ContextProposer, deterministic_filters
from .render import prepare_item, render_additions
from .schema import (
    ContextAddition,
    DecisionItem,
    EvaluationUnit,
    OptimizationAttempt,
    OptimizationConfig,
    OptimizationResult,
)
from .targets import DecisionTarget


class IneligibleDecisionError(ValueError):
    """Raised when an item has no initially correct decision eligible for optimization."""


@dataclass(frozen=True)
class _EligibleDecision:
    record: EvaluationRecord
    target_option: str
    target_branch: str


def optimize_context(
    item: DecisionItem,
    target: DecisionTarget,
    proposer: ContextProposer,
    checker: ContextChecker,
    *,
    config: OptimizationConfig | None = None,
    target_option: str | None = None,
) -> OptimizationResult:
    config = config or OptimizationConfig()
    prepared = prepare_item(item)
    eligible = _select_eligible_decision(prepared, target, target_option)
    unit = eligible.record.unit
    clean_result = eligible.record.result
    target_option = eligible.target_option
    target_branch = eligible.target_branch
    root_margin = target_margin(clean_result, target_branch)
    root = make_candidate(
        item_id=prepared.item_id,
        additions=(),
        result=clean_result,
        margin=root_margin,
    )
    archive = [root]
    population = [root for _ in range(config.particles)]
    attempts: list[OptimizationAttempt] = []
    seen_sources = {content_hash(prepared.source)}
    target_calls = 0
    threshold_reached = False
    iterations_completed = 0

    iteration_count = 1 if config.strategy != "adaptive" else config.rounds
    slot_count = 1 if config.strategy != "adaptive" else config.particles

    for iteration in range(1, iteration_count + 1):
        iterations_completed = iteration
        if iteration == 1:
            parents = [root for _ in range(slot_count)]
        else:
            allocation_seed = config.seed + int(
                content_hash(
                    {"item_id": prepared.item_id, "iteration": iteration, "purpose": "allocate"}
                )[:8],
                16,
            ) % 1_000_000
            indices = allocate_with_restarts(
                [candidate.margin for candidate in archive],
                count=slot_count,
                temperature=config.temperature,
                exploration=config.exploration,
                restart_fraction=config.restart_fraction,
                seed=allocation_seed,
            )
            parents = [archive[index] for index in indices]

        next_population = []
        for slot, parent in enumerate(parents):
            child, child_attempts = _extend_candidate(
                item=prepared,
                unit=unit,
                parent=parent,
                target_option=target_option,
                target_branch=target_branch,
                target=target,
                proposer=proposer,
                checker=checker,
                config=config,
                root_margin=root_margin,
                clean_target_probability=clean_result.probabilities[target_branch],
                iteration=iteration,
                slot=slot,
                target_call_start=target_calls,
                seen_sources=seen_sources,
            )
            attempts.extend(child_attempts)
            target_calls += sum(attempt.result is not None for attempt in child_attempts)
            if child is None:
                next_population.append(parent)
            else:
                next_population.append(child)
                archive.append(child)
        population = next_population
        threshold_reached = any(
            is_targeted_flip(candidate.result, target_branch, config.success_threshold)
            for candidate in population
        )
        if threshold_reached:
            break

    evaluated = [attempt for attempt in attempts if attempt.result is not None]
    best = max(
        evaluated,
        key=lambda attempt: (
            attempt.margin if attempt.margin is not None else float("-inf")
        ),
        default=None,
    )
    any_flip = any(attempt.targeted_flip for attempt in evaluated)
    return OptimizationResult(
        item_id=prepared.item_id,
        unit_id=unit.unit_id,
        target_id=target.target_id,
        target_option=target_option,
        target_branch=target_branch,
        clean_result=clean_result,
        root_margin=root_margin,
        targeted_flip=any_flip,
        threshold_reached=threshold_reached,
        success_threshold=config.success_threshold,
        iterations_completed=iterations_completed,
        proposal_calls=len(attempts),
        target_calls=target_calls,
        attempts=attempts,
        best_attempt_id=best.attempt_id if best else None,
        best_source=(
            render_additions(prepared, best.additions) if best is not None else None
        ),
        best_additions=best.additions if best is not None else [],
        best_result=best.result if best is not None else None,
        best_margin=best.margin if best is not None else None,
        config=config,
    )


def _select_eligible_decision(
    item: DecisionItem,
    target: DecisionTarget,
    requested_target: str | None,
) -> _EligibleDecision:
    records = evaluate([item], target)
    if len(item.gold_keys) == 1:
        record = records[0]
        if not record.initially_correct:
            raise IneligibleDecisionError(
                "the target does not initially answer this item correctly"
            )
        branch = requested_target or select_target_option(record.unit, record.result)
        if branch == record.unit.gold_branch or branch not in record.unit.branches:
            raise ValueError("target option must be an available non-gold option")
        return _EligibleDecision(record, branch, branch)

    by_choice = {record.unit.metadata["choice_key"]: record for record in records}
    candidates = [
        record
        for record in records
        if record.initially_correct and record.unit.gold_branch == "false"
    ]
    if requested_target is not None:
        record = by_choice.get(requested_target)
        if record is None or record not in candidates:
            raise IneligibleDecisionError(
                "the requested multi-answer target must be an initially rejected non-gold option"
            )
    else:
        if not candidates:
            raise IneligibleDecisionError("no initially correct absent option is eligible")
        record = max(candidates, key=lambda value: value.result.probabilities["true"])
    return _EligibleDecision(record, str(record.unit.metadata["choice_key"]), "true")


def _extend_candidate(
    *,
    item: DecisionItem,
    unit: EvaluationUnit,
    parent: Candidate,
    target_option: str,
    target_branch: str,
    target: DecisionTarget,
    proposer: ContextProposer,
    checker: ContextChecker,
    config: OptimizationConfig,
    root_margin: float,
    clean_target_probability: float,
    iteration: int,
    slot: int,
    target_call_start: int,
    seen_sources: set[str],
) -> tuple[Candidate | None, list[OptimizationAttempt]]:
    if len(parent.additions) >= config.max_additions:
        return None, []
    current_source = render_additions(item, list(parent.additions))
    prior_sentences = [addition.sentence for addition in parent.additions]
    feedback = None
    if config.strategy == "adaptive" and iteration > 1 and parent.additions:
        gain = parent.margin - root_margin
        direction = "positive" if gain > 0.05 else "negative" if gain < -0.05 else "flat"
        feedback = {
            "clean_target_probability": clean_target_probability,
            "current_target_probability": parent.result.probabilities[target_branch],
            "clean_margin": root_margin,
            "current_margin": parent.margin,
            "margin_gain": gain,
            "direction": direction,
        }
    attempts = []
    for retry in range(config.proposal_retries + 1):
        seed = proposal_seed(
            base_seed=config.seed,
            item_id=item.item_id,
            parent_id=parent.candidate_id,
            iteration=iteration,
            slot=slot,
            retry=retry,
        )
        neutral = config.strategy == "neutral_one_shot"
        proposal = proposer.propose(
            item,
            target_option=None if neutral else target_option,
            seed=seed,
            current_source=current_source,
            prior_additions=prior_sentences,
            optimization_feedback=feedback,
            neutral=neutral,
        )
        proposal = proposal.model_copy(
            update={
                "filter_results": deterministic_filters(
                    item,
                    proposal.sentence,
                    proposal.boundary_id,
                    prior_additions=prior_sentences,
                )
            }
        )
        check = None
        rejection = None
        if not all(proposal.filter_results.values()):
            rejection = "deterministic_filter"
        else:
            check = checker.check(item, proposal, prior_additions=list(parent.additions))
            if not check.accepted:
                rejection = "semantic_checker"

        renderable = (
            proposal.filter_results.get("boundary_valid", False)
            and proposal.filter_results.get("nonempty", False)
        )
        additions = list(parent.additions)
        if renderable:
            additions.append(
                ContextAddition(
                    boundary_id=proposal.boundary_id,
                    sentence=proposal.sentence,
                )
            )
        augmented_source = render_additions(item, additions) if renderable else None
        source_hash = content_hash(augmented_source) if augmented_source is not None else None
        if rejection is None and source_hash in seen_sources:
            rejection = "duplicate_source"

        attempt = OptimizationAttempt(
            attempt_id=content_hash(
                {
                    "item_id": item.item_id,
                    "iteration": iteration,
                    "slot": slot,
                    "retry": retry,
                    "parent": parent.candidate_id,
                    "proposal": proposal,
                }
            )[:20],
            iteration=iteration,
            slot=slot,
            retry=retry,
            parent_id=parent.candidate_id,
            parent_margin=parent.margin,
            proposal=proposal,
            checker=check,
            rejection=rejection,
            additions=additions,
            source_hash=source_hash,
        )
        attempts.append(attempt)
        if rejection is not None:
            continue

        assert augmented_source is not None and source_hash is not None
        seen_sources.add(source_hash)
        result = target.evaluate(unit, augmented_source)
        margin = target_margin(result, target_branch)
        attempt = attempt.model_copy(
            update={
                "target_call_index": target_call_start + 1,
                "result": result,
                "margin": margin,
                "target_probability": result.probabilities[target_branch],
                "targeted_flip": result.selected == target_branch,
            }
        )
        attempts[-1] = attempt
        return (
            make_candidate(
                item_id=item.item_id,
                additions=tuple(additions),
                result=result,
                margin=margin,
            ),
            attempts,
        )
    return None, attempts
