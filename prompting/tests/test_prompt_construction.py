import pytest
from typing import List
from kromaplus.algorithms.data_structures.graph import (
    Concept, 
    EquivalentClass, 
    EquivalentClassRelation,
)
from kromaplus.prompting.prompt_construction import PromptConstructor
from kromaplus.prompting.prompt_templates import YES_NO_CONFIDENCE_TEMPLATE


def make_equiv_class(names: List[str]) -> EquivalentClass:
    concepts = [Concept(name=n) for n in names]
    return EquivalentClass(equiv_concepts=concepts)

def make_relation(
    src_names: List[str],
    tgt_names: List[str],
    relation: str = "yes",
    score: float = 0.8,
) -> EquivalentClassRelation:
    src = make_equiv_class(src_names)
    tgt = make_equiv_class(tgt_names)
    return EquivalentClassRelation(src=src, tgt=tgt, relation=relation, score=score)

@pytest.fixture
def few_shot() -> List[EquivalentClassRelation]:
    return [
        make_relation(["Cat"], ["Feline"], relation="yes", score=0.95),
        make_relation(["Dog"], ["Canine"], relation="no", score=0.40),
    ]

def test_format_example_fills_yes_no_and_confidence_tags(few_shot):
    ex_yes = few_shot[0]
    pc = PromptConstructor("Instr", [ex_yes])
    snippet = pc._format_example(ex_yes)
    # should contain the question header
    assert snippet.splitlines()[0] == "Q: Are these two equivalence classes semantically equivalent?"
    # bullet lines for both classes
    assert f"* Equivalence class A: {ex_yes.src.describe()}" in snippet
    assert f"* Equivalence class B: {ex_yes.tgt.describe()}" in snippet
    # answer tags: yes and score scaled to 0–10 (0.95→round(9.5)=10)
    expected_tags = YES_NO_CONFIDENCE_TEMPLATE.replace("[yes|no]", "yes").replace("[0-10]", "10")
    assert f"A: {expected_tags}" in snippet

def test_format_example_handles_no_relation_and_low_confidence(few_shot):
    ex_no = few_shot[1]
    pc = PromptConstructor("Instr", [ex_no])
    snippet = pc._format_example(ex_no)
    # answer tags: no and score scaled to 0–10 (0.40→round(4.0)=4)
    expected_tags = YES_NO_CONFIDENCE_TEMPLATE.replace("[yes|no]", "no").replace("[0-10]", "4")
    assert f"A: {expected_tags}" in snippet

def test_build_prompt_includes_instruction_examples_and_final_placeholder(few_shot):
    instr = "Determine equivalence."
    pc = PromptConstructor(instr, few_shot)
    query = make_relation(["Lion"], ["Big Cat"], relation="yes", score=0.75)
    prompt = pc.build_prompt(query)
    lines = prompt.splitlines()
    # 1) instruction then blank line
    assert lines[0] == instr
    assert lines[1] == ""
    # 2) each few-shot example appears via Q:
    q_lines = [l for l in lines if l.startswith("Q: ")]
    assert len(q_lines) == len(few_shot) + 1  # two examples + one final query
    # 3) final query placeholder uses the raw template
    assert prompt.endswith(f"A: {YES_NO_CONFIDENCE_TEMPLATE}")