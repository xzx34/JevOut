from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .checker import OpenAIContextChecker
from .clients import OpenAICompatibleClient
from .evaluate import evaluate
from .io import read_jsonl, write_jsonl
from .optimize import IneligibleDecisionError, optimize_context
from .proposer import OpenAIContextProposer
from .render import prepare_item
from .schema import DecisionItem, OptimizationConfig
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
        choices=("adaptive", "targeted_one_shot", "neutral_one_shot"),
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

    return value


def _target(args: argparse.Namespace):
    if args.target == "jev":
        target = JevTarget()
    else:
        if not args.target_url:
            raise SystemExit("--target-url is required for --target http")
        target = HttpDecisionTarget(args.target_id, args.target_url)
    return CachedTarget(target, args.cache_dir) if args.cache_dir else target


def main() -> None:
    load_dotenv()
    args = parser().parse_args()
    items = list(read_jsonl(args.input, DecisionItem))
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
    else:
        api_key = (
            os.environ[args.proposer_api_key_env]
            if args.proposer_api_key_env
            else "local"
        )
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
            try:
                results.append(
                    optimize_context(
                        item,
                        target,
                        proposer,
                        checker,
                        config=config,
                    )
                )
            except IneligibleDecisionError as error:
                skipped.append({"item_id": item.item_id, "reason": str(error)})
        write_jsonl(Path(args.output), results)
        summary = {
            "items": len(prepared),
            "eligible": len(results),
            "skipped": skipped,
            "targeted_flips": sum(result.targeted_flip for result in results),
            "output": args.output,
        }
    print(json.dumps(summary, indent=2))
