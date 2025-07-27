from __future__ import annotations
import torch
from typing import Optional, List, Set, Dict
from collections import defaultdict
from embeddings.text_embedding import TextEmbedding
from embeddings.graph_embedding import GraphEmbedding

class Concept:
    """concepts of an ontology"""
    def __init__(
        self, 
        name: str, 
        parents: Optional[List[Concept]] = None,
        children: Optional[List[Concept]] = None, 
        ground_set: Optional[List[str]] = None,
    ):
        self.name = name
        self.rank = None # [issue] can we make rank update if ontology dynamically updates? 
        self.parents: List[Concept] = parents if parents is not None else []
        self.children: List[Concept] = children if children is not None else []
        self.ground_set: List[str] = ground_set if ground_set is not None else []

    def compute_rank(self) -> int:
        # [issue] can we defer update to concept graph update or recompute rank?
        if self.rank is None:
            self.rank = 1 + max((child.compute_rank() for child in self.children), default=-1)
        return self.rank

    def add_parent(self, p: Concept):
        self.parents.append(p)

    def add_child(self, c: Concept):
        self.children.append(c)

    def __repr__(self) -> str:
        parent_names = [p.name for p in self.parents]
        child_names  = [c.name for c in self.children]
        return (
            f"Concept(name={self.name!r}, rank={self.rank}, "
            f"parents={parent_names}, children={child_names}, "
            f"ground_set={self.ground_set})"
        )

class ConceptRelation:
    """relations between concepts"""
    def __init__(self, src: Concept, tgt: Concept, score: float = 0.0):
        self.src = src
        self.tgt = tgt
        self.score = score

    def __repr__(self) -> str:
        return f"ConceptRelation({self.src.name} -> {self.tgt.name}, score={self.score})"

class EquivalentClass:
    """bisimilar concepts treated as one equivalence class"""
    def __init__(
        self, 
        equiv_concepts: List[Concept],
        parents: Optional[List[EquivalentClass]] = None,
        children: Optional[List[EquivalentClass]] = None, 
    ):
        if not equiv_concepts:
            raise ValueError("EquivalentClass requires at least one Concept")
        self.equiv_concepts = equiv_concepts
        self.id: str = equiv_concepts[0].name # use first concept name as id
        self.rank = None # [issue] can we make rank update if ontology dynamically updates? 
        self.parents: List[EquivalentClass] = parents if parents is not None else []
        self.children: List[EquivalentClass] = children if children is not None else []
        self.text_embedding: torch.Tensor = None
        self.graph_embedding: torch.Tensor = None

    def compute_embedding(self, alpha: float = 0.5) -> torch.Tensor:
        """
        fuse graph- and text-embeddings into a single vector
            z_c = alpha * graph_emb + (1-alpha) * text_emb
        cache results in self.embedding
        """
        if hasattr(self, "embedding") and self.embedding is not None:
            return self.embedding
        # ensure text_embedding exists
        if self.text_embedding is None:
            # compute & cache
            self.text_embedding = TextEmbedding().compute_embedding(self)
            if self.text_embedding is None:
                raise RuntimeError("TextEmbedding returned None")
        # ensure graph_embedding exists
        if self.graph_embedding is None:
            # compute & cache
            self.graph_embedding = GraphEmbedding().compute_embedding(self)
            if self.graph_embedding is None:
                raise RuntimeError("GraphEmbedding returned None")
        # check dims match: should be matched by default
        if self.graph_embedding.shape != self.text_embedding.shape:
            raise ValueError(
                f"Dimension mismatch: "
                f"text {tuple(self.text_embedding.shape)}, "
                f"graph {tuple(self.graph_embedding.shape)}"
            )
        # fuse and cache
        # [issue] fusing is a bit strict: future work can relax this with learnable alpha
        z = alpha * self.graph_embedding + (1.0 - alpha) * self.text_embedding
        self.embedding = z
        return z

    def compute_rank(self) -> int:
        # [issue] can we defer update to concept graph update or recompute rank?
        if self.rank is None:
            self.rank = 1 + max((child.compute_rank() for child in self.children), default=-1)
        return self.rank

    def members(self) -> str:
        """get members within an equivalent class"""
        return ", ".join(c.name for c in self.equiv_concepts)

    def describe(self) -> str:
        """narrative summary of an equivalent class and its relations"""
        members   = self.members() or "none"
        parents  = ", ".join(p.members() for p in self.parents) or "none"
        children = ", ".join(c.members() for c in self.children)  or "none"
        return (
            f"This equivalence class, comprised of {members}, "
            "brings together semantically aligned concepts into a unified whole. "
            f"It is rooted in the broader notions of {parents}, from which it descends, "
            f"and it branches out to give rise to {children} as its sub-concepts."
        )

    def __repr__(self) -> str:
        names = [c.name for c in self.equiv_concepts]
        return f"EquivalentClass(id={self.id!r}, concepts={names})"

class EquivalentClassRelation:
    """relations between equivalent classes"""
    def __init__(
        self, 
        src: EquivalentClass, 
        tgt: EquivalentClass, 
        relation: str, 
        score: float = 0.0
    ):
        assert relation in ("yes", "no"), "relation must be 'yes' or 'no'"
        self.src = src
        self.tgt = tgt
        self.relation = relation
        self.score = score

    def describe(self) -> str:
        source_summary = self.src.describe()
        target_summary = self.tgt.describe()
        relation_label = (
            "semantically equivalent"
            if self.relation == "yes"
            else "not semantically equivalent"
        )
        return (
            f"Equivalence class A: {source_summary}; "
            f"Equivalence class B: {target_summary}. "
            f"Equivalence class A are {relation_label} to equivalence class B."
        )

    # [issue] test representation of this repr
    def __repr__(self) -> str:
        return (f"EquivalentClassRelation("
                f"src={self.src!r}, "
                f"tgt={self.tgt!r}, "
                f"score={self.score})")

class DSU:
    """disjoint set union with path compression and union by size to check cycle inside ontology"""
    def __init__(self):
        self.parent: Dict[Any, Any] = {}
        self.size:   Dict[Any, int] = {}
    
    def find(self, v):
        if v not in self.parent: # lazy create v
            self.parent[v] = v
            self.size[v] = 1
            return v
        if self.parent[v] != v:
            # path compression -- inverse ackermann time complexity
            self.parent[v] = self.find(self.parent[v])
        return self.parent[v]

    def union(self, a, b):
        a = self.find(a)
        b = self.find(b)
        if a == b: return False
        if self.size[a] < self.size[b]:
            a, b = b, a # swap
        self.parent[b] = a
        self.size[a] += self.size[b]
        return True

class ConceptGraph:
    """a DAG whose nodes are equivalent classes"""
    def __init__(self, nodes: List[EquivalentClass], edges: List[EquivalentClassRelation]):
        self.nodes: Dict[str, EquivalentClass] = {n.id: n for n in nodes} # fast node lookup
        self.edges: List[EquivalentClassRelation] = []
        self.parents:  Dict[str, List[EquivalentClass]] = defaultdict(list)
        self.children: Dict[str, List[EquivalentClass]] = defaultdict(list)
        # add initial edges
        for e in edges:
            self.add_edge(e)

    def add_node(self, node: EquivalentClass) -> bool:
        """add an equivalent class to concept graph, return status"""
        if node is not None:
            self.nodes[node.id] = node
            return True
        return False
    
    def add_edge(self, edge: EquivalentClassRelation) -> bool:
        """add a directed relation between two equivalence classes"""
        self.edges.append(edge)
        self.parents[edge.tgt.id].append(edge.src)
        self.children[edge.src.id].append(edge.tgt)
        # ensure both nodes exist in concept graph
        self.nodes.setdefault(edge.src.id, edge.src)
        self.nodes.setdefault(edge.tgt.id, edge.tgt)

    def parents_of(self, node: EquivalentClass) -> List[EquivalentClass]:
        return list(self.parents[node.id])

    def children_of(self, node: EquivalentClass) -> List[EquivalentClass]:
        return list(self.children[node.id])

    def compute_ranks(self) -> Dict[str, int]:
        """
        bottom-up rank computation
        rank(node): 0, if no children
                    1 + max{ rank(child) }, otherwise
        memoizes into each equivalence class rank
        """
        ranks: Dict[str, int] = {}
        
        def dfs(node: EquivalentClass) -> int:
            if node.id in ranks:
                return ranks[node.id]
            kids = self.children.get(node.id, [])
            r = 0 if not kids else 1 + max(dfs(k) for k in kids)
            ranks[node.id] = r
            node.rank = r
            return r

        for n in self.nodes.values():
            dfs(n)

        return ranks

    def has_cycle(self):
        """
        union-find algorithm to check consistency
        [issue] can we use this to reward model active sampling?
        """
        dsu = DSU()

        # preprocess 
        parent = dsu.parent
        size   = dsu.size
        for node_id in self.nodes:
            parent[node_id] = node_id
            size[node_id]   = 1

        # local binding of union function
        union = dsu.union
        
        for e in self.edges:
            u, v = e.src.id, e.tgt.id
            if not union(u, v):
                return True
        return False

    def __repr__(self) -> str:
        return (f"ConceptGraph(num_nodes={len(self.nodes)}, "
                f"num_edges={len(self.edges)})")

    def __str__(self) -> str:
        lines = [repr(self)]
        for node in self.nodes.values():
            lines.append(f"  {node!r}  children -> {[c.id for c in self.children_of(node)]}")
        return "\n".join(lines)
                
