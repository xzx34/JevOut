from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChoiceOption(StrictModel):
    key: str = Field(min_length=1)
    text: str = Field(min_length=1)


class InsertionBoundary(StrictModel):
    id: str = Field(min_length=1)
    offset: int = Field(ge=0)
    left: str = ""
    right: str = ""


class DecisionItem(StrictModel):
    item_id: str = Field(min_length=1)
    context_kind: Literal["question_only", "passage", "dialogue"] = "passage"
    source: str = Field(min_length=1)
    question: str = Field(min_length=1)
    choices: list[ChoiceOption] = Field(min_length=2)
    gold_keys: list[str] = Field(min_length=1)
    insertion_boundaries: list[InsertionBoundary] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_keys_and_boundaries(self) -> DecisionItem:
        choice_keys = [choice.key for choice in self.choices]
        if len(choice_keys) != len(set(choice_keys)):
            raise ValueError("choice keys must be unique")
        if len(self.gold_keys) != len(set(self.gold_keys)):
            raise ValueError("gold keys must be unique")
        if not set(self.gold_keys).issubset(choice_keys):
            raise ValueError("gold keys must name available choices")
        boundary_ids = [boundary.id for boundary in self.insertion_boundaries]
        if len(boundary_ids) != len(set(boundary_ids)):
            raise ValueError("insertion boundary identifiers must be unique")
        if any(boundary.offset > len(self.source) for boundary in self.insertion_boundaries):
            raise ValueError("insertion boundary offset exceeds source length")
        return self

    @property
    def choice_map(self) -> dict[str, str]:
        return {choice.key: choice.text for choice in self.choices}


class EvaluationUnit(StrictModel):
    unit_id: str
    item_id: str
    primitive: Literal["choice", "membership"]
    source: str
    instructions: str
    branches: dict[str, str]
    gold_branch: str
    group_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_branches(self) -> EvaluationUnit:
        if self.gold_branch not in self.branches:
            raise ValueError("gold branch is missing from branches")
        if self.primitive == "membership" and set(self.branches) != {"false", "true"}:
            raise ValueError("membership units must expose false and true branches")
        return self


class DecisionResult(StrictModel):
    unit_id: str
    target_id: str
    selected: str
    probabilities: dict[str, float]
    native_confidence: float | None = None
    latency_ms: float = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    model_revision: str
    request_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_distribution(self) -> DecisionResult:
        if not self.probabilities:
            raise ValueError("probabilities cannot be empty")
        if self.selected not in self.probabilities:
            raise ValueError("selected branch is missing from probabilities")
        if any(value < 0 or value > 1 for value in self.probabilities.values()):
            raise ValueError("probabilities must be between zero and one")
        if abs(sum(self.probabilities.values()) - 1.0) > 1e-5:
            raise ValueError("probabilities must sum to one")
        best = max(self.probabilities.values())
        if self.probabilities[self.selected] < best - 1e-8:
            raise ValueError("selected branch must have maximal probability")
        return self


class CheckResult(StrictModel):
    label_preserved: bool
    locally_coherent: bool
    direct_decision_cue: bool
    new_decisive_evidence: bool
    reason: str

    @property
    def accepted(self) -> bool:
        return (
            self.label_preserved
            and self.locally_coherent
            and not self.direct_decision_cue
            and not self.new_decisive_evidence
        )


class ContextAddition(StrictModel):
    boundary_id: str
    sentence: str = Field(min_length=1)


class Proposal(StrictModel):
    sentence: str
    boundary_id: str
    generator_revision: str
    prompt_hash: str
    filter_results: dict[str, bool] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OptimizationConfig(StrictModel):
    strategy: Literal["adaptive", "targeted_one_shot", "neutral_one_shot"] = "adaptive"
    particles: int = Field(default=16, gt=0)
    rounds: int = Field(default=4, gt=0)
    temperature: float = Field(default=1.0, gt=0)
    exploration: float = Field(default=0.1, ge=0, le=1)
    restart_fraction: float = Field(default=0.25, gt=0, lt=1)
    success_threshold: float = Field(default=0.7, gt=0, le=1)
    proposal_retries: int = Field(default=1, ge=0)
    seed: int = 20260921
    max_additions: int = Field(default=4, gt=0)


class OptimizationAttempt(StrictModel):
    attempt_id: str
    iteration: int
    slot: int
    retry: int
    parent_id: str
    parent_margin: float
    proposal: Proposal
    checker: CheckResult | None = None
    rejection: str | None = None
    additions: list[ContextAddition]
    source_hash: str | None = None
    target_call_index: int | None = None
    result: DecisionResult | None = None
    margin: float | None = None
    target_probability: float | None = None
    targeted_flip: bool = False


class OptimizationResult(StrictModel):
    item_id: str
    unit_id: str
    target_id: str
    target_option: str
    target_branch: str
    clean_result: DecisionResult
    root_margin: float
    targeted_flip: bool
    threshold_reached: bool
    success_threshold: float
    iterations_completed: int
    proposal_calls: int
    target_calls: int
    attempts: list[OptimizationAttempt]
    best_attempt_id: str | None = None
    best_source: str | None = None
    best_additions: list[ContextAddition] = Field(default_factory=list)
    best_result: DecisionResult | None = None
    best_margin: float | None = None
    config: OptimizationConfig
    metadata: dict[str, Any] = Field(default_factory=dict)


class TransferResult(StrictModel):
    item_id: str
    unit_id: str
    source_target_id: str
    destination_target_id: str
    target_option: str
    target_branch: str
    matched_clean_correct: bool
    selection_kind: Literal["successful_context", "best_margin", "clean_fallback"]
    source_targeted_flip: bool
    source_attempt_id: str | None = None
    additions: list[ContextAddition] = Field(default_factory=list)
    destination_clean_result: DecisionResult | None = None
    result: DecisionResult | None = None
    targeted_transfer: bool = False
    target_probability: float | None = None
    target_probability_delta: float | None = None
    margin: float | None = None
    margin_delta: float | None = None


def item_to_units(item: DecisionItem) -> list[EvaluationUnit]:
    if len(item.gold_keys) == 1:
        return [
            EvaluationUnit(
                unit_id=item.item_id,
                item_id=item.item_id,
                primitive="choice",
                source=item.source,
                instructions=item.question,
                branches=item.choice_map,
                gold_branch=item.gold_keys[0],
                group_id=item.item_id,
            )
        ]

    gold = set(item.gold_keys)
    return [
        EvaluationUnit(
            unit_id=f"{item.item_id}:{choice.key}",
            item_id=item.item_id,
            primitive="membership",
            source=item.source,
            instructions=(
                f"For the question '{item.question}', "
                f"is '{choice.text}' one of the correct answers?"
            ),
            branches={
                "false": "This candidate is not a correct answer",
                "true": "This candidate is one of the correct answers",
            },
            gold_branch="true" if choice.key in gold else "false",
            group_id=item.item_id,
            metadata={"choice_key": choice.key, "choice_text": choice.text},
        )
        for choice in item.choices
    ]
