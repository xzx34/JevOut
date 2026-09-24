from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol

from .clients import OpenAICompatibleClient
from .io import content_hash, write_json
from .render import derive_boundaries
from .schema import DecisionItem, Proposal

PROMPT_ROOT = Path(__file__).resolve().parent / "prompts"

_DIRECT_TERMS = re.compile(
    r"\b(correct answer|incorrect answer|right answer|wrong answer|choose|select|"
    r"option [a-z]|answer is)\b",
    re.IGNORECASE,
)


class ContextProposer(Protocol):
    def propose(
        self,
        item: DecisionItem,
        *,
        target_option: str | None,
        seed: int,
        current_source: str,
        prior_additions: list[str],
        optimization_feedback: dict[str, float | str] | None,
        neutral: bool = False,
    ) -> Proposal: ...


def load_prompt(name: str) -> str:
    return (PROMPT_ROOT / f"{name}.txt").read_text(encoding="utf-8").strip()


def proposal_schema(item: DecisionItem) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "insert_after": {
                "type": "string",
                "enum": [boundary.id for boundary in derive_boundaries(item)],
            },
            "sentence": {"type": "string"},
        },
        "required": ["insert_after", "sentence"],
        "additionalProperties": False,
    }


def proposer_input(
    item: DecisionItem,
    *,
    target_option: str | None,
    current_source: str,
    prior_additions: list[str],
    optimization_feedback: dict[str, float | str] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "context_kind": item.context_kind,
        "original_text": item.source,
        "current_text": current_source,
        "question": item.question,
        "choices": item.choice_map,
        "correct_answer_that_must_remain_correct": {
            "keys": item.gold_keys,
            "texts": [item.choice_map[key] for key in item.gold_keys],
        },
        "previously_added_sentences": prior_additions,
        "allowed_boundaries": [
            {"id": boundary.id, "left": boundary.left, "right": boundary.right}
            for boundary in derive_boundaries(item)
        ],
    }
    if target_option is not None:
        payload["alternative_to_support_only_indirectly"] = {
            "key": target_option,
            "text": item.choice_map[target_option],
        }
    if optimization_feedback is not None:
        # The key name is retained because it is part of the frozen experimental prompt.
        payload["search_feedback"] = optimization_feedback
    return payload


def deterministic_filters(
    item: DecisionItem,
    sentence: str,
    boundary_id: str,
    *,
    prior_additions: list[str],
) -> dict[str, bool]:
    sentence = sentence.strip()
    boundaries = {boundary.id for boundary in derive_boundaries(item)}
    normalized = re.sub(r"\W+", " ", sentence.lower()).strip()
    prior = {re.sub(r"\W+", " ", value.lower()).strip() for value in prior_additions}
    return {
        "boundary_valid": boundary_id in boundaries,
        "nonempty": bool(sentence),
        "not_duplicate": normalized not in prior,
        "no_direct_phrase": _DIRECT_TERMS.search(sentence) is None,
    }


class OpenAIContextProposer:
    def __init__(
        self,
        client: OpenAICompatibleClient,
        model: str,
        *,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 160,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def propose(
        self,
        item: DecisionItem,
        *,
        target_option: str | None,
        seed: int,
        current_source: str,
        prior_additions: list[str],
        optimization_feedback: dict[str, float | str] | None,
        neutral: bool = False,
    ) -> Proposal:
        prompt_name = "generator_neutral" if neutral else "generator"
        system = load_prompt(prompt_name)
        schema = proposal_schema(item)
        user = proposer_input(
            item,
            target_option=None if neutral else target_option,
            current_source=current_source,
            prior_additions=prior_additions,
            optimization_feedback=None if neutral else optimization_feedback,
        )
        cache_path = self._cache_path(system, schema, user, seed)
        if cache_path is not None and cache_path.exists():
            return Proposal.model_validate_json(cache_path.read_text(encoding="utf-8"))
        try:
            output, usage = self.client.structured_chat(
                model=self.model,
                system=system,
                user=user,
                schema_name="background_addition",
                schema=schema,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                seed=seed,
            )
            valid = True
        except (KeyError, TypeError, ValueError):
            output = {"insert_after": derive_boundaries(item)[0].id, "sentence": ""}
            usage = {}
            valid = False
        boundary_id = str(output.get("insert_after", ""))
        sentence = str(output.get("sentence", "")).strip()
        proposal = Proposal(
            sentence=sentence,
            boundary_id=boundary_id,
            generator_revision=self.model,
            prompt_hash=content_hash({"system": system, "schema": schema}),
            filter_results=deterministic_filters(
                item,
                sentence,
                boundary_id,
                prior_additions=prior_additions,
            ),
            metadata={
                "usage": usage,
                "prompt_name": prompt_name,
                "seed": seed,
                "structured_output_valid": valid,
            },
        )
        if cache_path is not None:
            write_json(cache_path, proposal)
        return proposal

    def _cache_path(
        self,
        system: str,
        schema: dict[str, Any],
        user: dict[str, Any],
        seed: int,
    ) -> Path | None:
        if self.cache_dir is None:
            return None
        key = content_hash(
            {
                "kind": "proposal",
                "model": self.model,
                "system": system,
                "schema": schema,
                "user": user,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "seed": seed,
            }
        )
        return self.cache_dir / "proposals" / f"{key}.json"
