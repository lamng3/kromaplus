from __future__ import annotations
from typing import Optional, List, Set, Dict
from collections import defaultdict
from algorithms.data_structures.graph import (
    Concept,
    ConceptGraph,
    EquivalentClass,
    EquivalentClassRelation,
)


class HypergraphView(ConceptGraph):
    """
    a concept can only belong to one equivalence class
    a hyperedge can contain multiple concepts coming from multiple equivalence classes
    """
    def __init__(
        self,
        eq_classes: Optional[List[EquivalentClass]] = None,
        eq_relations: Optional[List[EquivalentClassRelation]] = None,
        extra_hyperedges: Optional[Dict[str, Set[Concept]]] = None,
    ):  
        # initialize concept graph
        super().__init__(
            nodes=eq_classes or [],
            edges=eq_relations or []
        )
        
        # hypergraph view
        self.hyperedges: Dict[str, Set[Concept]] = {}
        self.incidence: Dict[Concept, Set[str]] = defaultdict(set) # fast lookup which edges a node belongs to
        
        # register each equivalence class as a hyperedge by its id
        if eq_classes:
            for eq in eq_classes:
                members = set(eq.equiv_concepts)
                self._register_hyperedge(eq.id, members)

        # register any extra hyperedges
        if extra_hyperedges:
            for eid, members in extra_hyperedges.items():
                self._register_hyperedge(eid, members)

    def _register_hyperedge(self, edge_id: str, members: Set[Concept]) -> None:
        """add hyperedge and update incidence"""
        self.hyperedges[edge_id] = set(members)
        for c in members:
            self.incidence[c].add(edge_id)

    def add_hyperedge(self, edge_id: str, members: Set[Concept]) -> None:
        """
        add a hyperedge with given id and member concepts
        overwrite if edge_id already exists
        """
        self._register_hyperedge(edge_id, members)

    def remove_hyperedge(self, edge_id: str) -> bool:
        """
        return an entire hyperedge by its id
        update incidence of member nodes
        """
        members = self.hyperedges.pop(edge_id, None)
        if members is None:
            return False
        for c in members:
            self.incidence[c].discard(edge_id)
        return True

    def hyperedges_of(self, concept: Concept) -> List[str]:
        """return list of hyperedges a given concept belongs to"""
        return list(self.incidence.get(concept, []))

    def nodes_of(self, edge_id: str) -> Set[Concept]:
        """return set of concepts belonging to given hyperedge"""
        return set(self.hyperedges.get(edge_id, []))

    def degree(self, concept: Concept) -> int:
        """return the number of hyperedges the concept participates in"""
        return len(self.incidence.get(concept, []))

    def __repr__(self) -> str:
        return (
            f"HypergraphView(#classes={len(self.nodes)}, #hyperedges={len(self.hyperedges)})"
        )

    def __str__(self) -> str:
        return

class Multigraph:
    def __init__(self):
        pass