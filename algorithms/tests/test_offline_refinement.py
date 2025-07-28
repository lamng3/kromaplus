import pytest
from typing import List
from algorithms.data_structures.graph import (
    Concept,
    EquivalentClass,
    EquivalentClassRelation,
    ConceptGraph,
)
from algorithms.offline_refinement import offline_refine_graph


def make_equiv_class(name: str) -> EquivalentClass:
    return EquivalentClass(equiv_concepts=[Concept(name=name)])

def make_relation(src: EquivalentClass, tgt: EquivalentClass) -> EquivalentClassRelation:
    # relation/score don’t matter for structure
    return EquivalentClassRelation(src=src, tgt=tgt, relation="yes", score=1.0)

def extract_ids(graph: ConceptGraph) -> set:
    return set(graph.nodes.keys())

def test_all_leaf_nodes_collapse_to_one():
    # three disconnected leaf nodes A, B, C → all rank 0
    A = make_equiv_class("A")
    B = make_equiv_class("B")
    C = make_equiv_class("C")
    cg = ConceptGraph(nodes=[A, B, C], edges=[])
    refined = offline_refine_graph(cg)
    # should collapse all three into a single node "A+B+C"
    assert extract_ids(refined) == {"A+B+C"}
    assert refined.edges == []

def test_star_graph_collapses_siblings():
    # star: A parent of B and C
    A = make_equiv_class("A")
    B = make_equiv_class("B")
    C = make_equiv_class("C")
    edges = [make_relation(A, B), make_relation(A, C)]
    cg = ConceptGraph(nodes=[A, B, C], edges=edges)
    refined = offline_refine_graph(cg)
    # B and C (rank 0) should collapse to "B+C", A stays alone.
    ids = extract_ids(refined)
    assert ids == {"A", "B+C"}
    # edges should now be from A -> B+C (two edges, one per original)
    tgts = [e.tgt.id for e in refined.edges]
    assert sorted(tgts) == ["B+C", "B+C"]
    for e in refined.edges:
        assert e.src.id == "A"
        assert e.tgt.id == "B+C"

def test_chain_graph_preserves_hierarchy():
    # A->B->D chain (no siblings)
    A = make_equiv_class("A")
    B = make_equiv_class("B")
    D = make_equiv_class("D")
    edges = [make_relation(A, B), make_relation(B, D)]
    cg = ConceptGraph(nodes=[A, B, D], edges=edges)
    refined = offline_refine_graph(cg)
    # no siblings at same rank, so no collapse beyond singleton blocks
    assert extract_ids(refined) == {"A", "B", "D"}
    # edges should preserve the same sequence
    seq = [(e.src.id, e.tgt.id) for e in refined.edges]
    assert seq == [("A", "B"), ("B", "D")]

if __name__ == "__main__":
    pytest.main()
