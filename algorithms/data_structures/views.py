from __future__ import annotations
from typing import Optional, List, Set, Dict
from collections import defaultdict
from algorithms.data_structures.graph import (
    Concept,
    ConceptGraph,
    EquivalentClass,
    EquivalentClassRelation,
)


class HypergraphView:
    """
    a concept can only belong to one equivalence class
    a hyperedge can contain multiple concepts coming from multiple equivalence classes
    """
    def __init__(
        self,
        graph: ConceptGraph,
        hyperedges: Optional[Dict[str, Set[Concept]]] = None,
    ):  
        # hypergraph view
        self.hyperedges: Dict[str, Set[Concept]] = {}
        self.incidence: Dict[Concept, Set[str]] = defaultdict(set) # fast lookup which edges a node belongs to
        
        # register each equivalence class as a hyperedge by its id
        if graph.nodes:
            for eq in graph.nodes:
                members = set(eq.equiv_concepts)
                self._register_hyperedge(eq.id, members)

        # register any extra hyperedges
        if hyperedges:
            for eid, members in hyperedges.items():
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

class MultigraphView:
    """
    allow multiple parallel edges and loops.
    edges allow more than one edge between same two vertices, and self-loops.

    adjacency maps each source node id to a list of outgoing edge ids.
    reverse_adjacency maps each target node id to a list of incoming edge ids.
    """
    def __init__(
        self,
        graph: ConceptGraph,
        multiedges: Optional[Dict[str, Set[Concept]]] = None,
    ):
        # storage for multigraph edges
        self.edges: List[tuple[str, str, str, dict]] = []
        self.adjacency: Dict[str, List[str]] = defaultdict(list)
        self.reverse_adjacency: Dict[str, List[str]] = defaultdict(list)

        if multiedges:
            for edge_id, src_id, tgt_id, metadata in multiedges:
                self.add_edge(edge_id, src_id, tgt_id, metadata)

    def add_edge(
        self,
        edge_id: str,
        src_id: str,
        tgt_id: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """add a parallel edge or loop to the multigraph"""
        if src_id not in self.nodes or tgt_id not in self.nodes:
            raise KeyError("Node not found in base graph")
        meta = metadata or {}
        self.edges.append((edge_id, src_id, tgt_id, meta))
        self.adjacency[src_id].append(edge_id)
        self.reverse_adjacency[tgt_id].append(edge_id)

    def remove_edge(self, edge_id: str) -> bool:
        """remove all edges with the given id"""
        found = False
        for e in list(self.edges):
            if e[0] == edge_id:
                _, src_id, tgt_id, _ = e
                self.edges.remove(e)
                self.adjacency[src_id].remove(edge_id)
                self.reverse_adjacency[tgt_id].remove(edge_id)
                found = True
        return found

    def edges_between(self, src_id: str, tgt_id: str) -> List[str]:
        """list edge ids connecting src_id to tgt_id"""
        return [eid for eid, s, t, _ in self.edges if s == src_id and t == tgt_id]

    def incident_edges(self, node_id: str) -> List[str]:
        """list edge ids incident to node_id (including loops)"""
        out = list(self.adjacency.get(node_id, []))
        inc = list(self.reverse_adjacency.get(node_id, []))
        # build a new dictinary whose keys are items in the concatenated list in order
        return list(dict.fromkeys(out + inc))

    def __repr__(self) -> str:
        return f"MultigraphView(#nodes={len(self.nodes)}, #edges={len(self.edges)})"

    def __str__(self) -> str:
        return