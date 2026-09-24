from context_optimization.proposer import deterministic_filters, load_prompt, proposer_input
from context_optimization.render import prepare_item
from context_optimization.schema import ChoiceOption, DecisionItem


def item() -> DecisionItem:
    return prepare_item(
        DecisionItem(
            item_id="x",
            source="Original source.",
            question="Which option?",
            choices=[
                ChoiceOption(key="A", text="Alpha"),
                ChoiceOption(key="B", text="Beta"),
            ],
            gold_keys=["A"],
        )
    )


def test_packaged_prompts_are_available() -> None:
    assert "strongest answer-preserving addition" in load_prompt("generator")
    assert "exactly the same complete set" in load_prompt("checker")


def test_proposer_input_uses_paper_level_terms_but_preserves_frozen_feedback_key() -> None:
    value = proposer_input(
        item(),
        target_option="B",
        current_source="Original source.",
        prior_additions=[],
        optimization_feedback={"current_margin": -0.2},
    )
    assert value["alternative_to_support_only_indirectly"]["key"] == "B"
    assert value["search_feedback"] == {"current_margin": -0.2}


def test_deterministic_filter_rejects_direct_answer_cue() -> None:
    value = item()
    result = deterministic_filters(
        value,
        "The correct answer is B.",
        "b0",
        prior_additions=[],
    )
    assert not result["no_direct_phrase"]
