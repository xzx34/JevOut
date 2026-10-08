"""A deterministic, offline example; its probabilities are synthetic."""

from __future__ import annotations

from pathlib import Path

from .evaluate import evaluate
from .independent import sample_independent_context
from .io import content_hash, write_json, write_jsonl
from .optimize import optimize_context
from .render import derive_boundaries
from .repeatability import retest_context
from .schema import (
    CheckResult,
    ChoiceOption,
    DecisionItem,
    DecisionResult,
    IndependentRootConfig,
    OptimizationConfig,
    Proposal,
)
from .targets import CallableTarget


class DemoProposer:
    def propose(self, item, *, seed, **kwargs):
        return Proposal(
            sentence=f"A background note numbered {seed} accompanies the record.",
            boundary_id=derive_boundaries(item)[0].id,
            generator_revision="synthetic-demo",
            prompt_hash=content_hash("offline-demo-proposer"),
        )


class DemoChecker:
    def check(self, item, proposal, *, prior_additions):
        return CheckResult(
            label_preserved=True,
            locally_coherent=True,
            direct_decision_cue=False,
            new_decisive_evidence=False,
            reason="The extra numbered note does not change the stated arrival day.",
        )


def demo_target() -> CallableTarget:
    def score(unit, source):
        additions = source.count("A background note numbered")
        if additions == 0:
            probabilities = {"A": 0.7, "B": 0.2, "C": 0.1}
        elif additions == 1:
            probabilities = {"A": 0.5, "B": 0.45, "C": 0.05}
        else:
            probabilities = {"A": 0.2, "B": 0.75, "C": 0.05}
        return DecisionResult(
            unit_id=unit.unit_id,
            target_id="synthetic-demo",
            selected=max(probabilities, key=probabilities.get),
            probabilities=probabilities,
            latency_ms=0,
            model_revision="synthetic-demo-v1",
            request_hash=content_hash({"unit": unit.unit_id, "source": source}),
        )

    return CallableTarget("synthetic-demo", score)


def run_demo(output_dir: Path) -> dict:
    item = DecisionItem(
        item_id="offline-demo",
        source="The record states that the parcel arrived on Monday.",
        question="On which day did the parcel arrive?",
        choices=[
            ChoiceOption(key="A", text="Monday"),
            ChoiceOption(key="B", text="Tuesday"),
            ChoiceOption(key="C", text="Wednesday"),
        ],
        gold_keys=["A"],
    )
    target = demo_target()
    proposer, checker = DemoProposer(), DemoChecker()
    clean = evaluate([item], target)
    optimized = optimize_context(
        item,
        target,
        proposer,
        checker,
        config=OptimizationConfig(particles=4, rounds=2, proposal_retries=0),
    )
    baseline = sample_independent_context(
        item,
        target,
        proposer,
        checker,
        config=IndependentRootConfig(target_budget=8, proposal_budget=16, batch_size=4),
        reference=optimized,
    )
    repeated = retest_context(optimized, item, target)
    write_jsonl(output_dir / "items.jsonl", [item])
    write_jsonl(output_dir / "clean.jsonl", clean)
    write_jsonl(output_dir / "optimized.jsonl", [optimized])
    write_jsonl(output_dir / "independent.jsonl", [baseline])
    write_jsonl(output_dir / "repeatability.jsonl", [repeated])
    summary = {
        "synthetic": True,
        "network_calls": 0,
        "initially_correct": clean[0].initially_correct,
        "fixed_target": optimized.target_option,
        "optimization_flip": optimized.targeted_flip,
        "optimization_target_calls": optimized.target_calls,
        "independent_target_calls": baseline.target_calls,
        "repeat_target_hits": repeated.target_hits,
        "repetitions": repeated.repetitions,
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "summary.json", summary)
    return summary
