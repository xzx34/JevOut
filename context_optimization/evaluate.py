from __future__ import annotations

from pydantic import Field

from .render import prepare_item
from .schema import DecisionItem, DecisionResult, EvaluationUnit, StrictModel, item_to_units
from .targets import DecisionTarget


class EvaluationRecord(StrictModel):
    item_id: str
    unit: EvaluationUnit
    result: DecisionResult
    initially_correct: bool
    metadata: dict = Field(default_factory=dict)


def evaluate(items: list[DecisionItem], target: DecisionTarget) -> list[EvaluationRecord]:
    records = []
    for item in items:
        prepared = prepare_item(item)
        units = item_to_units(prepared)
        results = target.evaluate_many(units)
        if [result.unit_id for result in results] != [unit.unit_id for unit in units]:
            raise ValueError("target results do not match input unit order")
        records.extend(
            EvaluationRecord(
                item_id=prepared.item_id,
                unit=unit,
                result=result,
                initially_correct=result.selected == unit.gold_branch,
            )
            for unit, result in zip(units, results, strict=True)
        )
    return records
