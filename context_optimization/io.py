from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda item: (
            item.model_dump(mode="json") if isinstance(item, BaseModel) else str(item)
        ),
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def read_jsonl[T](
    path: str | Path, model: type[T] | None = None
) -> list[T] | list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            rows.append(model.model_validate(value) if model is not None else value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, BaseModel):
                row = row.model_dump(mode="json")
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
