import pytest
from pydantic import ValidationError

from context_optimization.render import original_is_subsequence, prepare_item, render_additions
from context_optimization.schema import ChoiceOption, ContextAddition, DecisionItem, item_to_units


def item(**updates) -> DecisionItem:
    values = {
        "item_id": "x",
        "source": "One sentence. Another sentence.",
        "question": "Which option?",
        "choices": [ChoiceOption(key="A", text="Alpha"), ChoiceOption(key="B", text="Beta")],
        "gold_keys": ["A"],
    }
    values.update(updates)
    return DecisionItem(**values)


def test_prepare_and_render_preserve_original_text() -> None:
    prepared = prepare_item(item())
    assert [boundary.id for boundary in prepared.insertion_boundaries] == ["b0"]
    rendered = render_additions(
        prepared,
        [ContextAddition(boundary_id="b0", sentence="A locally relevant detail was recorded.")],
    )
    assert original_is_subsequence(prepared.source, rendered)


def test_unknown_gold_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="gold keys"):
        item(gold_keys=["C"])


def test_multi_answer_item_expands_to_membership_units() -> None:
    units = item_to_units(
        item(
            choices=[
                ChoiceOption(key="A", text="Alpha"),
                ChoiceOption(key="B", text="Beta"),
                ChoiceOption(key="C", text="Gamma"),
            ],
            gold_keys=["A", "C"],
        )
    )
    assert [unit.gold_branch for unit in units] == ["true", "false", "true"]
