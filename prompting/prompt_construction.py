from __future__ import annotations
from typing import List, Tuple, Dict, Any
import abc

import torch
from embeddings.text_embedding import TextEmbedding
from embeddings.graph_embedding import GraphEmbedding
from algorithms.data_structures.graph import (
    EquivalentClass, 
    EquivalentClassRelation, 
    ConceptGraph,
)
from prompting.prompt_templates import (
    YES_NO_CONFIDENCE_TEMPLATE
)


class PromptConstructor:
    """build LLM prompts for deciding equivalence of two ontology classes"""
    def __init__(
        self,
        task_description: str,
        examples: List[EquivalentClassRelation],
        output_format: str = YES_NO_CONFIDENCE_TEMPLATE,
    ):
        self.task_description = task_description
        self.examples = examples # few-shot examples
        self.output_format = output_format.strip()

    def _format_example(self, ex: EquivalentClassRelation) -> str:
        """turn example relation into a prompt snippet"""
        src_summary = ex.src.describe()
        tgt_summary = ex.tgt.describe()
        yesno = "yes" if ex.relation == "yes" else "no"
        # scale score [0–1] to [0–10]:
        conf10 = int(round(ex.score * 10))
        answer_tags = self.output_format \
            .replace("[yes|no]", yesno) \
            .replace("[0-10]", str(conf10))
        return (
            "Q: Are these two equivalence classes semantically equivalent?\n"
            f"* Equivalence class A: {src_summary}\n"
            f"* Equivalence class B: {tgt_summary}\n"
            f"A: {answer_tags}\n"
        )

    def build_prompt(
        self,
        query: EquivalentClassRelation,
    ) -> str:
        """assemble final prompt"""
        parts: List[str] = []
        
        # instruction
        parts.append(self.task_description)
        parts.append("")

        # few-shot examples
        for ex in self.examples:
            parts.append(self._format_example(ex))

        # new query
        src_summary = query.src.describe()
        tgt_summary = query.tgt.describe()
        parts.append("Q: Are these two equivalence classes semantically equivalent?")
        parts.append(f"* Equivalence class A: {src_summary}")
        parts.append(f"* Equivalence class B: {tgt_summary}")
        parts.append(f"A: {self.output_format}")

        # join all parts
        return "\n".join(parts)