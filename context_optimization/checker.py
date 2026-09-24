from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol

from .clients import OpenAICompatibleClient
from .io import content_hash, write_json
from .proposer import load_prompt
from .render import render_additions
from .schema import CheckResult, ContextAddition, DecisionItem, Proposal

CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "label_preserved": {"type": "boolean"},
        "locally_coherent": {"type": "boolean"},
        "direct_decision_cue": {"type": "boolean"},
        "new_decisive_evidence": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": [
        "label_preserved",
        "locally_coherent",
        "direct_decision_cue",
        "new_decisive_evidence",
        "reason",
    ],
    "additionalProperties": False,
}


class ContextChecker(Protocol):
    def check(
        self,
        item: DecisionItem,
        proposal: Proposal,
        *,
        prior_additions: list[ContextAddition],
    ) -> CheckResult: ...


def _negated_near(reason: str, start: int) -> bool:
    prefix = reason[max(0, start - 96) : start]
    clause_start = max(prefix.rfind("."), prefix.rfind(";"), prefix.rfind("?"))
    clause = prefix[clause_start + 1 :]
    return re.search(r"\b(?:no|not|without|does not|doesn't|did not|didn't)\b", clause) is not None


class OpenAIContextChecker:
    def __init__(
        self,
        client: OpenAICompatibleClient,
        model: str,
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def check(
        self,
        item: DecisionItem,
        proposal: Proposal,
        *,
        prior_additions: list[ContextAddition],
    ) -> CheckResult:
        current_source = render_additions(item, prior_additions)
        augmented_source = render_additions(
            item,
            [
                *prior_additions,
                ContextAddition(
                    boundary_id=proposal.boundary_id,
                    sentence=proposal.sentence,
                ),
            ],
        )
        system = load_prompt("checker")
        user = {
            "context_kind": item.context_kind,
            "original_text": item.source,
            "current_text": current_source,
            "new_sentence": proposal.sentence,
            "augmented_text": augmented_source,
            "question": item.question,
            "choices": item.choice_map,
            "correct_answer": item.gold_keys,
        }
        cache_path = self._cache_path(system, user)
        if cache_path is not None and cache_path.exists():
            return CheckResult.model_validate_json(cache_path.read_text(encoding="utf-8"))
        try:
            output, _ = self.client.structured_chat(
                model=self.model,
                system=system,
                user=user,
                schema_name="semantic_check",
                schema=CHECK_SCHEMA,
                temperature=0.0,
                top_p=1.0,
                max_tokens=160,
                seed=0,
            )
            result = CheckResult.model_validate(output)
        except (TypeError, ValueError):
            result = CheckResult(
                label_preserved=False,
                locally_coherent=False,
                direct_decision_cue=False,
                new_decisive_evidence=False,
                reason="invalid structured checker response",
            )
        result = _repair_contradictory_reason(item, result)
        if cache_path is not None:
            write_json(cache_path, result)
        return result

    def _cache_path(self, system: str, user: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        key = content_hash(
            {"kind": "checker", "model": self.model, "system": system, "user": user}
        )
        return self.cache_dir / "checks" / f"{key}.json"


def _repair_contradictory_reason(item: DecisionItem, result: CheckResult) -> CheckResult:
    reason = result.reason.lower()
    contradiction = re.search(
        r"\b(changes?|changed|changing|alters?|altered|invalidates?)\b.{0,50}"
        r"\b(correct answer|answer set|gold label|label)\b",
        reason,
    )
    if contradiction is not None and not _negated_near(reason, contradiction.start()):
        result = result.model_copy(
            update={"label_preserved": False, "new_decisive_evidence": True}
        )
    dialogue_change = re.search(
        r"\b(introduces? a new request|new intent|new authorization|new action)\b",
        reason,
    )
    if (
        item.context_kind == "dialogue"
        and dialogue_change is not None
        and not _negated_near(reason, dialogue_change.start())
    ):
        result = result.model_copy(
            update={"label_preserved": False, "new_decisive_evidence": True}
        )
    return result
