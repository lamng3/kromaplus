import networkx as nx
from node2vec import Node2Vec
from typing import Optional
from kromaplus.algorithms.data_structures.graph import (
    ConceptGraph, EquivalentClass
)

class GraphEmbedding:
    def __init__(self, cg: Optional[ConceptGraph] = None):
        """if concept graph is passed, it will be converted to self.G"""
        self.G: nx.DiGraph = nx.DiGraph()
        self.embs: dict[str, list[float]] = {}
        if cg:
            self.from_concept_graph(cg)
            self.embs = self.learn_node2vec()

    def compute_embedding(self, node: EquivalentClass):
        """retrieve embedding based on graph"""
        if not self.embs:
            raise ValueError(
                "No embeddings found: ensure you called from_concept_graph() and "
                "then learn_node2vec() (or passed cg into __init__)."
            )
        if node.id not in self.embs:
            self.embs = self.learn_node2vec()
        emb = self.embs[node.id]
        setattr(node, "graph_embedding", emb)
        return emb

    def from_concept_graph(self, cg: ConceptGraph) -> nx.DiGraph:
        """populate self.G from a concept graph's nodes and edges"""
        self.G.clear()
        for n in cg.nodes.values():
            self.G.add_node(n.id)
        for e in cg.edges:
            self.G.add_edge(e.src.id, e.tgt.id, weight=e.score)
        return self.G

    def learn_node2vec(
        self,
        dimensions: int = 16,
        walk_length: int = 10,
        num_walks: int = 50,
        p: float = 1.0,
        q: float = 1.0,
        workers: int = 1,
        window: int = 5,
        epochs: int = 1,
    ):
        """run node2vec on self.G and return a dict[node_id -> vector]"""
        if self.G.number_of_nodes() == 0:
            raise ValueError("GraphEmbedding.G is empty; call from_concept_graph() first.")
        # set up random walks
        # [issue] tune parameters to have a better node2vec
        node2vec = Node2Vec(
            self.G, 
            dimensions=dimensions, 
            walk_length=walk_length,
            num_walks=num_walks,
            p=p,
            q=q,
            workers=workers,
            weight_key="weight",
        )
        # train skip-gram
        model = node2vec.fit(
            window=window,
            min_count=1,
            batch_words=4,
            epochs=epochs,
        )
        # extract embeddings as plain Python lists
        return {
            node: model.wv.get_vector(node).tolist()
            for node in self.G.nodes()
        }

    