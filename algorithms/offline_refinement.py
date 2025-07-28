from typing import Set, List
from itertools import chain

from algorithms.data_structures.graph import (
    EquivalentClass,
    EquivalentClassRelation,
    ConceptGraph,
)
# prepare ontologies into configuration for refinement
# from preprocess.prepare_configuration import build_configuration_w


def offline_refine_graph(graph: ConceptGraph) -> ConceptGraph:
    """
    given a concept graph, returns a new refined concept graph
    """
    # deep copy nodes and edges
    import copy
    G = copy.deepcopy(graph)

    # compute ranks
    ranks = G.compute_ranks()
    max_rank = max(ranks.values(), default=0)

    # build buckets B0...Bρ
    B = [
        {nid for nid, r in ranks.items() if r == i}
        for i in range(max_rank + 1)
    ]

    # initial partition P = [B0, B1, …]
    P = [set(bucket) for bucket in B]

    # iterate over ranks
    for i in range(max_rank + 1):
        # find Di = blocks wholly contained in Bi
        Di = [block for block in P if block.issubset(B[i])]

        # collapse each block in Di
        for block in Di:
            new_id = _collapse_block(G, block)
            P.remove(block)
            P.append({new_id})

        # split blocks C in U_{j>i} B_j by adjacency to each c belonging to Bi
        higher_union = set().union(*(B[j] for j in range(i + 1, max_rank + 1)))
        for cid in B[i]:
            # skip cid that was merged away
            if cid not in G.nodes:
                continue
            center = G.nodes[cid]
            for blk in list(P):
                if blk and blk.issubset(higher_union):
                    C1, C2 = _split_block_by_center(G, blk, center)
                    if C1 and C2:
                        P.remove(blk)
                        P.extend([C1, C2])
    
    return G

def _collapse_block(
    graph: ConceptGraph,
    ids_to_merge: Set[str]
) -> str:
    """
    merges all EquivalentClass nodes with id in ids_to_merge into one.
    returns the new node's id.
    """
    from algorithms.data_structures.graph import EquivalentClass, EquivalentClassRelation

    # collect EquivalentClass instances to merge
    equivalence_nodes = [graph.nodes[nid] for nid in sorted(ids_to_merge)]

    # create the merged EquivalentClass with combined concepts
    merged_concepts = list(chain.from_iterable(ec.equiv_concepts for ec in equivalence_nodes))
    merged_node = EquivalentClass(equiv_concepts=merged_concepts)
    merged_node.id = "+".join(sorted(ids_to_merge))

    # rewire edges
    updated_edges: List[EquivalentClassRelation] = []
    for edge in graph.edges:
        src = merged_node if edge.src.id in ids_to_merge else graph.nodes[edge.src.id]
        tgt = merged_node if edge.tgt.id in ids_to_merge else graph.nodes[edge.tgt.id]
        # preserve relation type and score
        updated_edges.append(
            EquivalentClassRelation(src=src, tgt=tgt, relation=edge.relation, score=edge.score)
        )

    # replace nodes and edges in the graph
    for nid in ids_to_merge:
        del graph.nodes[nid]
    graph.nodes[merged_node.id] = merged_node
    graph.edges = updated_edges

    # rebuild adjacency lists
    graph.parents.clear()
    graph.children.clear()
    for edge in updated_edges:
        graph.parents[edge.tgt.id].append(edge.src)
        graph.children[edge.src.id].append(edge.tgt)

    return merged_node.id

def _get_adjacent_ids(
    graph: ConceptGraph,
    center: EquivalentClass,
) -> set[str]:
    """
    return set of all node ids that are directly connected to center
    either as a parent or a child
    """
    parents = {ec.id for ec in graph.parents_of(center)}
    children = {ec.id for ec in graph.children_of(center)}
    return parents.union(children)

def _split_block_by_center(
    graph: ConceptGraph,
    block: Set[str],
    center: EquivalentClass
) -> (Set[str], Set[str]):
    """
    splits block into two sets:
        * C1 = ids adjacent to center
        * C2 = the rest
    """
    # adjacency of center
    adjacent_ids = _get_adjacent_ids(graph, center)

    # C1 = ids in block that touch the center
    C1 = {nid for nid in block if nid in adjacent_ids}

    # C2 = everything else
    C2 = block.difference(C1)

    return C1, C2