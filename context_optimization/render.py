from __future__ import annotations

import re

from .schema import ContextAddition, DecisionItem, InsertionBoundary

_SENTENCE_END = re.compile(r"(?<=[.!?])(?:[\"'”’)]*)\s+")


def derive_boundaries(item: DecisionItem, *, window: int = 80) -> list[InsertionBoundary]:
    if item.insertion_boundaries:
        return item.insertion_boundaries
    if item.context_kind == "question_only":
        offsets = [0]
    else:
        offsets = [match.end() for match in _SENTENCE_END.finditer(item.source)] or [0]
    return [
        InsertionBoundary(
            id=f"b{index}",
            offset=offset,
            left=item.source[max(0, offset - window) : offset].strip(),
            right=item.source[offset : offset + window].strip(),
        )
        for index, offset in enumerate(offsets)
    ]


def prepare_item(item: DecisionItem) -> DecisionItem:
    return item.model_copy(update={"insertion_boundaries": derive_boundaries(item)})


def render_additions(item: DecisionItem, additions: list[ContextAddition]) -> str:
    if not additions:
        return item.source
    boundaries = {boundary.id: boundary for boundary in derive_boundaries(item)}
    by_offset: dict[int, list[str]] = {}
    for addition in additions:
        sentence = addition.sentence.strip()
        if addition.boundary_id not in boundaries:
            raise ValueError(f"unknown insertion boundary: {addition.boundary_id}")
        by_offset.setdefault(boundaries[addition.boundary_id].offset, []).append(sentence)

    pieces = []
    cursor = 0
    for offset, sentences in sorted(by_offset.items()):
        pieces.append(item.source[cursor:offset])
        inserted = " ".join(sentences)
        if offset == 0:
            pieces.append(f"{inserted}\n\n")
        else:
            before = "" if item.source[offset - 1 : offset].isspace() else " "
            after = "" if item.source[offset : offset + 1].isspace() else " "
            pieces.append(f"{before}{inserted}{after}")
        cursor = offset
    pieces.append(item.source[cursor:])
    return "".join(pieces)


def original_is_subsequence(original: str, rendered: str) -> bool:
    cursor = 0
    for character in rendered:
        if cursor < len(original) and character == original[cursor]:
            cursor += 1
    return cursor == len(original)
