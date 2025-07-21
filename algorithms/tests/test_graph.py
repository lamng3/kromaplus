import pytest
from collections import defaultdict

from kromaplus.algorithms.data_structures.graph import (
    Concept, ConceptRelation,
    EquivalentClass, EquivalentClassRelation,
    DSU, ConceptGraph,
)


def make_simple_concepts():
    # creates A → B → C
    a = Concept("A")
    b = Concept("B")
    c = Concept("C")
    a.add_child(b); b.add_parent(a)
    b.add_child(c); c.add_parent(b)
    return a, b, c

def test_concept_graph_no_cycle_and_ranks():
    a, b, c = make_simple_concepts()
    eq_ab = EquivalentClass([a, b])
    eq_c  = EquivalentClass([c])
    # eq_ab → eq_c
    rel = EquivalentClassRelation(eq_ab, eq_c, score=1.0)
    cg = ConceptGraph(nodes=[eq_ab, eq_c], edges=[rel])
    # no cycle here
    assert cg.has_cycle() is False
    # compute ranks
    ranks = cg.compute_ranks()
    assert ranks[eq_c.id] == 0   # leaf
    assert ranks[eq_ab.id] == 1  # parent of leaf

def test_concept_graph_with_cycle():
    # build two classes and add mutual edges
    x_cls = EquivalentClass([Concept("X")])
    y_cls = EquivalentClass([Concept("Y")])
    rel1 = EquivalentClassRelation(x_cls, y_cls)
    rel2 = EquivalentClassRelation(y_cls, x_cls)
    cg = ConceptGraph(nodes=[x_cls, y_cls], edges=[rel1, rel2])
    # mutual edges form a cycle
    assert cg.has_cycle() is True

def test_concept_repr_and_rank():
    a, b, c = make_simple_concepts()
    # before computing rank, .rank is None
    assert a.rank is None
    # compute rank
    a.compute_rank()
    b.compute_rank()
    c.compute_rank()
    # check rank
    assert c.rank == 0      # no children
    assert b.rank == 1      # 1 + max(child=c.rank)
    assert a.rank == 2      # 1 + max(child=b.rank)
    # repr should mention name and rank
    r = repr(a)
    assert "Concept(name='A'" in r
    assert "rank=2" in r

def test_concept_relation_repr():
    a, b, _ = make_simple_concepts()
    rel = ConceptRelation(a, b, score=0.55)
    s = repr(rel)
    assert "ConceptRelation('A' → 'B'" in s or "ConceptRelation(A -> B" in s
    assert "0.55" in s

def test_equivalent_class_and_relation_repr():
    a, b, c = make_simple_concepts()
    # put A and B into one equivalence class
    eq_ab = EquivalentClass([a, b])
    eq_c  = EquivalentClass([c])
    # test repr
    r = repr(eq_ab)
    assert "EquivalentClass(id='A'" in r
    assert "concepts=['A', 'B']" in r
    # create a relation between two eq classes
    eq_rel = EquivalentClassRelation(eq_ab, eq_c, score=0.99)
    s = repr(eq_rel)
    assert "src=EquivalentClass" in s
    assert "tgt=EquivalentClass" in s
    assert "score=0.99" in s

def test_concept_graph_basic_adjacency_and_ranks():
    a, b, c = make_simple_concepts()
    # two eq classes: {A,B} and {C}
    eq_ab = EquivalentClass([a, b])
    eq_c  = EquivalentClass([c])
    # define a single relation eq_ab → eq_c
    rel = EquivalentClassRelation(eq_ab, eq_c, score=1.0)
    # build graph
    cg = ConceptGraph(nodes=[eq_ab, eq_c], edges=[rel])
    # children/parents lookup
    children = cg.children_of(eq_ab)
    parents  = cg.parents_of(eq_c)
    assert children == [eq_c]
    assert parents  == [eq_ab]
    # rank computation on the eq‑class graph
    ranks = cg.compute_ranks()
    # eq_c has no children → rank 0, eq_ab→eq_c → rank 1
    assert ranks[eq_c.id] == 0
    assert ranks[eq_ab.id] == 1

# test DSU
def test_dsu_basic_union_find():
    dsu = DSU()
    # fresh elements point to themselves
    assert dsu.find("x") == "x"
    assert dsu.find("y") == "y"
    # union two different sets
    assert dsu.union("x", "y") is True
    root_x = dsu.find("x")
    root_y = dsu.find("y")
    assert root_x == root_y
    # unioning again returns False (already connected)
    assert dsu.union("x", "y") is False

def test_dsu_cycle_detection_via_union():
    dsu = DSU()
    edges = [("a", "b"), ("b", "c"), ("c", "a")]
    seen_cycle = False
    for u, v in edges:
        if not dsu.union(u, v):
            seen_cycle = True
            break
    assert seen_cycle, "Should detect a cycle when union(a,b), union(b,c), union(c,a)"


if __name__ == "__main__":
    pytest.main()
