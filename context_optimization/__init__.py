"""Reusable components for probability-guided context optimization."""

from .checker import ContextChecker, OpenAIContextChecker
from .evaluate import EvaluationRecord, evaluate
from .optimize import IneligibleDecisionError, optimize_context
from .proposer import ContextProposer, OpenAIContextProposer
from .schema import (
    CheckResult,
    ChoiceOption,
    ContextAddition,
    DecisionItem,
    DecisionResult,
    EvaluationUnit,
    InsertionBoundary,
    OptimizationAttempt,
    OptimizationConfig,
    OptimizationResult,
    Proposal,
    TransferResult,
)
from .targets import CachedTarget, CallableTarget, DecisionTarget, HttpDecisionTarget, JevTarget
from .transfer import select_representative_attempt, targeted_transfer_rate, transfer_context

__all__ = [
    "CachedTarget",
    "CallableTarget",
    "CheckResult",
    "ChoiceOption",
    "ContextAddition",
    "ContextChecker",
    "ContextProposer",
    "DecisionItem",
    "DecisionResult",
    "DecisionTarget",
    "EvaluationRecord",
    "EvaluationUnit",
    "HttpDecisionTarget",
    "IneligibleDecisionError",
    "InsertionBoundary",
    "JevTarget",
    "OpenAIContextChecker",
    "OpenAIContextProposer",
    "OptimizationAttempt",
    "OptimizationConfig",
    "OptimizationResult",
    "Proposal",
    "TransferResult",
    "evaluate",
    "optimize_context",
    "select_representative_attempt",
    "targeted_transfer_rate",
    "transfer_context",
]
