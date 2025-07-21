import pytest
import numpy as np
import networkx as nx
from kromaplus.algorithms.data_structures.graph import (
    Concept,
    ConceptGraph,
    EquivalentClass,
    EquivalentClassRelation,
)
from kromaplus.embeddings.graph_embedding import GraphEmbedding


@pytest.fixture
def small_concept_graph():
    # create two EquivalentClass nodes A and B
    a = EquivalentClass([Concept(name="A")])
    b = EquivalentClass([Concept(name="B")])
    # create a relation A -> B with score 0.5
    rel = EquivalentClassRelation(src=a, tgt=b, score=0.5)
    # build and return the ConceptGraph
    cg = ConceptGraph(nodes=[a, b], edges=[rel])
    return cg

def test_from_concept_graph_populates_nx(small_concept_graph):
    ge = GraphEmbedding()
    G = ge.from_concept_graph(small_concept_graph)
    # should have exactly two nodes, A and B
    assert set(G.nodes()) == {"A", "B"}
    # should have exactly one directed edge A -> B with the correct weight
    assert G.number_of_edges() == 1
    assert G.has_edge("A", "B")
    assert pytest.approx(G["A"]["B"]["weight"], rel=1e-6) == 0.5

def test_learn_node2vec_empty_graph_raises():
    ge = GraphEmbedding()
    with pytest.raises(ValueError):
        ge.learn_node2vec()

def test_learn_node2vec_returns_embeddings_of_correct_shape(small_concept_graph):
    ge = GraphEmbedding(small_concept_graph)
    # use very small dims and walks for speed
    emb_dict = ge.learn_node2vec(
        walk_length=5, num_walks=10, window=3, epochs=1, workers=1
    )
    # we should get embeddings for exactly the two nodes
    assert set(emb_dict.keys()) == {"A", "B"}

    # each embedding should be a python list of floats of length ge.dimensions
    for vec in emb_dict.values():
        assert isinstance(vec, list)
        assert len(vec) == ge.dimensions
        # Check all entries are floats
        assert all(isinstance(x, float) for x in vec)
        # Check it’s not all zeros
        assert not np.allclose(vec, [0.0] * ge.dimensions)

def test_learn_node2vec_tunable_dimensions(small_concept_graph):
    ge = GraphEmbedding(small_concept_graph)
    for dim in (4, 12):
        ge.dimensions = dim
        emb = ge.learn_node2vec(walk_length=5, num_walks=5, window=2, epochs=1)
        # all vectors should respect the requested dimensionality
        assert all(len(v) == dim for v in emb.values())

def test_init_with_cg_populates_embs_keys(small_concept_graph):
    # when constructed with a ConceptGraph, embds should already be populated
    ge = GraphEmbedding(small_concept_graph)
    # emb keys must exactly match node ids
    assert set(ge.embs.keys()) == {"A", "B"}

def test_compute_embedding_returns_list_and_matches_embs(small_concept_graph):
    ge = GraphEmbedding(small_concept_graph)
    # pick one node
    eq_a = small_concept_graph.nodes["A"]
    vec = ge.compute_embedding(eq_a)
    # should pull directly from ge.embs
    assert isinstance(vec, list)
    assert vec == ge.embs["A"]
    # default dimensions is ge.dimensions
    assert len(vec) == ge.dimensions
    # entries are floats and not all zero
    assert all(isinstance(x, float) for x in vec)
    assert not np.allclose(vec, [0.0] * ge.dimensions)

def test_compute_embedding_without_any_embeddings_raises():
    ge = GraphEmbedding()
    # a dummy equivalent class not in ge.embs
    dummy = EquivalentClass([Concept(name="X")])
    with pytest.raises(ValueError):
        _ = ge.compute_embedding(dummy)

