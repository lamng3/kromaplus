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
        dimensions=8, walk_length=5, num_walks=10, window=3, epochs=1, workers=1
    )
    # we should get embeddings for exactly the two nodes
    assert set(emb_dict.keys()) == {"A", "B"}

    # each embedding should be a python list of floats of length 8
    for vec in emb_dict.values():
        assert isinstance(vec, list)
        assert len(vec) == 8
        # Check all entries are floats
        assert all(isinstance(x, float) for x in vec)
        # Check it’s not all zeros
        assert not np.allclose(vec, [0.0] * 8)

def test_learn_node2vec_tunable_dimensions(small_concept_graph):
    ge = GraphEmbedding(small_concept_graph)
    for dim in (4, 12):
        emb = ge.learn_node2vec(dimensions=dim, walk_length=5, num_walks=5, window=2, epochs=1)
        # all vectors should respect the requested dimensionality
        assert all(len(v) == dim for v in emb.values())

