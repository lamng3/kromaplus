import torch
from transformers import AutoTokenizer, AutoModel
from typing import Optional, List, Sequence
from kromaplus.algorithms.data_structures.graph import (
    ConceptGraph, EquivalentClass
)


class TextEmbedding:
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    def compute_embedding(self, node: EquivalentClass):
        """
        build combined prompt from an equivalence class:
            * include name and a short list of ground set examples
        tokenize and embed the whole string
        """
        parts: List[str] = []
        for concept in node.equiv_concepts:
            seg = concept.name
            if getattr(concept, "ground_set", None):
                examples = ", ".join(concept.ground_set)
                seg += f" (examples: {examples})"
            parts.append(seg)
        # join multiple concepts with a separator
        prompt = " | ".join(parts)
        emb = self.to_embedding(prompt)
        # cache on the node
        setattr(node, "embedding", emb)
        return emb

    def to_embedding(
        self,
        text: str, 
        max_length: int = 512
    ) -> torch.Tensor:
        """
        tokenize `text` using embedding model and return a single embedding vector
        (mean-pooled over tokens)
        """
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding='longest',
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs).last_hidden_state # [1, seq_len, dim]
        return out.mean(dim=1).squeeze(0).cpu()