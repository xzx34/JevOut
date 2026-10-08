import json
import sys

from test_optimize import AcceptingChecker, FakeProposer, make_item, shifting_target
from test_repeatability import discovery

from context_optimization import cli
from context_optimization.io import write_jsonl


def test_independent_cli_preserves_reference_and_emits_budget_summary(
    tmp_path, monkeypatch, capsys
):
    items = tmp_path / "items.jsonl"
    reference = tmp_path / "reference.jsonl"
    output = tmp_path / "baseline.jsonl"
    write_jsonl(items, [make_item(), make_item().model_copy(update={"item_id": "extra"})])
    write_jsonl(reference, [discovery()])
    monkeypatch.setattr(cli, "_target", lambda args: shifting_target())
    monkeypatch.setattr(cli, "OpenAIContextProposer", lambda *args, **kwargs: FakeProposer())
    monkeypatch.setattr(cli, "OpenAIContextChecker", lambda *args, **kwargs: AcceptingChecker())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "context-opt",
            "optimize",
            "--input",
            str(items),
            "--output",
            str(output),
            "--strategy",
            "independent_root",
            "--reference",
            str(reference),
            "--target-budget",
            "3",
            "--proposal-budget",
            "6",
            "--proposer-url",
            "http://localhost:8000/v1",
            "--proposer-model",
            "fake",
        ],
    )
    cli.main()
    summary = json.loads(capsys.readouterr().out)
    assert summary["eligible"] == 1
    assert summary["skipped"] == [{"item_id": "extra", "reason": "no eligible reference result"}]
    assert summary["accepted_target_calls"] == 3
    saved = cli._saved_results(str(output))["x"]
    assert saved.config.strategy == "independent_root"
    assert saved.metadata["reused_reference"]


def test_retest_cli_uses_saved_successes(tmp_path, monkeypatch, capsys):
    items = tmp_path / "items.jsonl"
    contexts = tmp_path / "contexts.jsonl"
    output = tmp_path / "repeatability.jsonl"
    write_jsonl(items, [make_item()])
    write_jsonl(contexts, [discovery()])
    monkeypatch.setattr(cli, "_target", lambda args: shifting_target())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "context-opt",
            "retest",
            "--input",
            str(items),
            "--contexts",
            str(contexts),
            "--output",
            str(output),
            "--repetitions",
            "3",
            "--minimum-hits",
            "2",
        ],
    )
    cli.main()
    summary = json.loads(capsys.readouterr().out)
    assert summary["successful_contexts_retested"] == 1
    assert summary["target_calls"] == 6
    assert summary["flip_confirmed"] == 1
    assert json.loads(output.read_text())["target_hits"] == 3
