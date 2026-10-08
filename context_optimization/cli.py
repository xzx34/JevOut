from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .checker import OpenAIContextChecker
from .clients import OpenAICompatibleClient
from .evaluate import evaluate
from .independent import sample_independent_context
from .io import read_jsonl, write_jsonl
from .optimize import IneligibleDecisionError, optimize_context
from .proposer import OpenAIContextProposer
from .render import prepare_item
from .repeatability import retest_context
from .schema import (
    DecisionItem,
    IndependentRootConfig,
    IndependentRootResult,
    OptimizationConfig,
    OptimizationResult,
)
from .targets import CachedTarget, HttpDecisionTarget, JevTarget


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="context-opt")
    commands = value.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate")
    validate.add_argument("--input", required=True)
    validate.add_argument("--output")

    clean = commands.add_parser("evaluate")
    clean.add_argument("--input", required=True)
    clean.add_argument("--output", required=True)
    clean.add_argument("--target", choices=("jev", "http"), default="jev")
    clean.add_argument("--target-id", default="custom")
    clean.add_argument("--target-url")
    clean.add_argument("--cache-dir")

    optimize = commands.add_parser("optimize")
    optimize.add_argument("--input", required=True)
    optimize.add_argument("--output", required=True)
    optimize.add_argument("--target", choices=("jev", "http"), default="jev")
    optimize.add_argument("--target-id", default="custom")
    optimize.add_argument("--target-url")
    optimize.add_argument("--proposer-url", required=True)
    optimize.add_argument("--proposer-model", required=True)
    optimize.add_argument("--checker-model")
    optimize.add_argument("--proposer-api-key-env")
    optimize.add_argument("--cache-dir")
    optimize.add_argument(
        "--strategy",
        choices=("adaptive", "targeted_one_shot", "neutral_one_shot", "independent_root"),
        default="adaptive",
    )
    optimize.add_argument("--particles", type=int, default=16)
    optimize.add_argument("--rounds", type=int, default=4)
    optimize.add_argument("--temperature", type=float, default=1.0)
    optimize.add_argument("--exploration", type=float, default=0.1)
    optimize.add_argument("--restart-fraction", type=float, default=0.25)
    optimize.add_argument("--success-threshold", type=float, default=0.7)
    optimize.add_argument("--proposal-retries", type=int, default=1)
    optimize.add_argument("--seed", type=int, default=20260921)
    optimize.add_argument(
        "--target-budget", type=int, default=64, help="Independent-root accepted-call cap"
    )
    optimize.add_argument(
        "--proposal-budget", type=int, default=128, help="Independent-root proposal-attempt cap"
    )
    optimize.add_argument(
        "--batch-size", type=int, default=16, help="Independent-root slots per stopping check"
    )
    optimize.add_argument("--reference", help="Reuse fixed units and targets for independent_root")

    retest = commands.add_parser("retest")
    retest.add_argument("--input", required=True, help="The original decision items")
    retest.add_argument("--contexts", required=True, help="Saved optimization or baseline JSONL")
    retest.add_argument("--output", required=True)
    retest.add_argument("--target", choices=("jev", "http"), default="jev")
    retest.add_argument("--target-id", default="custom")
    retest.add_argument("--target-url")
    retest.add_argument("--repetitions", type=int, default=10)
    retest.add_argument("--minimum-hits", type=int, default=8)

    demo = commands.add_parser(
        "demo", help="Run the complete workflow offline on a synthetic model"
    )
    demo.add_argument("--output-dir", default="outputs/demo")

    return value


def _target(args: argparse.Namespace):
    if args.target == "jev":
        target = JevTarget()
    else:
        if not args.target_url:
            raise SystemExit("--target-url is required for --target http")
        target = HttpDecisionTarget(args.target_id, args.target_url)
    cache_dir = getattr(args, "cache_dir", None)
    return CachedTarget(target, cache_dir) if cache_dir else target


def _saved_results(path: str) -> dict[str, OptimizationResult]:
    results = {}
    for row in read_jsonl(path):
        model = (
            IndependentRootResult
            if row.get("config", {}).get("strategy") == "independent_root"
            else OptimizationResult
        )
        result = model.model_validate(row)
        if result.item_id in results:
            raise ValueError("saved results must contain at most one selected unit per item")
        results[result.item_id] = result
    return results


def main() -> None:
    load_dotenv()
    args = parser().parse_args()
    if args.command == "demo":
        from .demo import run_demo

        print(json.dumps(run_demo(Path(args.output_dir)), indent=2))
        return
    if getattr(args, "output", None):
        inputs = [getattr(args, name, None) for name in ("input", "contexts", "reference")]
        if any(value and Path(value).resolve() == Path(args.output).resolve() for value in inputs):
            raise ValueError("output must differ from input, contexts, and reference files")
    items = list(read_jsonl(args.input, DecisionItem))
    if len({item.item_id for item in items}) != len(items):
        raise ValueError("input item identifiers must be unique")
    prepared = [prepare_item(item) for item in items]

    if args.command == "validate":
        if args.output:
            write_jsonl(args.output, prepared)
        print(json.dumps({"valid_items": len(prepared), "output": args.output}, indent=2))
        return

    target = _target(args)
    if args.command == "evaluate":
        records = evaluate(prepared, target)
        write_jsonl(Path(args.output), records)
        summary = {
            "items": len(prepared),
            "decision_units": len(records),
            "initially_correct": sum(record.initially_correct for record in records),
            "output": args.output,
        }
    elif args.command == "retest":
        saved = _saved_results(args.contexts)
        by_item = {item.item_id: item for item in prepared}
        missing = set(saved) - set(by_item)
        if missing:
            raise ValueError(f"original input is missing {len(missing)} saved items")
        if not 1 <= args.minimum_hits <= args.repetitions:
            raise ValueError("require 1 <= minimum-hits <= repetitions")
        successful = [result for result in saved.values() if result.targeted_flip]
        records = []
        write_jsonl(args.output, records)
        for result in successful:
            records.append(
                retest_context(
                    result,
                    by_item[result.item_id],
                    target,
                    repetitions=args.repetitions,
                    minimum_hits=args.minimum_hits,
                )
            )
            write_jsonl(args.output, records)
        summary = {
            "successful_contexts_retested": len(records),
            "flip_confirmed": sum(record.flip_confirmed for record in records),
            "clean_and_flip_confirmed": sum(record.clean_and_flip_confirmed for record in records),
            "target_calls": len(records) * 2 * args.repetitions,
            "repetitions": args.repetitions,
            "minimum_hits": args.minimum_hits,
            "local_cache_bypassed": True,
            "output": args.output,
        }
    else:
        if args.reference and args.strategy != "independent_root":
            raise ValueError("--reference applies to --strategy independent_root")
        references = _saved_results(args.reference) if args.reference else None
        if references is not None and set(references) - {item.item_id for item in prepared}:
            raise ValueError("original input is missing saved reference items")
        api_key = os.environ[args.proposer_api_key_env] if args.proposer_api_key_env else "local"
        client = OpenAICompatibleClient(args.proposer_url, api_key=api_key)
        proposer = OpenAIContextProposer(
            client,
            args.proposer_model,
            cache_dir=args.cache_dir,
        )
        checker = OpenAIContextChecker(
            client,
            args.checker_model or args.proposer_model,
            cache_dir=args.cache_dir,
        )
        if args.strategy == "independent_root":
            config = IndependentRootConfig(
                target_budget=args.target_budget,
                proposal_budget=args.proposal_budget,
                batch_size=args.batch_size,
                proposal_retries=args.proposal_retries,
                success_threshold=args.success_threshold,
                seed=args.seed,
            )
        else:
            config = OptimizationConfig(
                strategy=args.strategy,
                particles=args.particles,
                rounds=args.rounds,
                temperature=args.temperature,
                exploration=args.exploration,
                restart_fraction=args.restart_fraction,
                success_threshold=args.success_threshold,
                proposal_retries=args.proposal_retries,
                seed=args.seed,
            )
        results = []
        skipped = []
        for item in prepared:
            if references is not None and item.item_id not in references:
                skipped.append({"item_id": item.item_id, "reason": "no eligible reference result"})
                continue
            try:
                if args.strategy == "independent_root":
                    result = sample_independent_context(
                        item,
                        target,
                        proposer,
                        checker,
                        config=config,
                        reference=references[item.item_id] if references is not None else None,
                    )
                else:
                    result = optimize_context(item, target, proposer, checker, config=config)
                results.append(result)
                write_jsonl(Path(args.output), results)
            except IneligibleDecisionError as error:
                skipped.append({"item_id": item.item_id, "reason": str(error)})
        write_jsonl(Path(args.output), results)
        summary = {
            "items": len(prepared),
            "eligible": len(results),
            "skipped": skipped,
            "targeted_flips": sum(result.targeted_flip for result in results),
            "targeted_flip_rate": (
                sum(result.targeted_flip for result in results) / len(results) if results else None
            ),
            "proposal_calls": sum(result.proposal_calls for result in results),
            "accepted_target_calls": sum(result.target_calls for result in results),
            "output": args.output,
        }
    print(json.dumps(summary, indent=2))
