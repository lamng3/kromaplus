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

    # build buckets Bi
    B: List[Set[str]] = [
        {nid for nid, r in ranks.items() if r == i}
        for i in range(max_rank + 1)
    ]

    # initialize partition P = [B0, B1, ...] by rank
    P: List[Set[str]] = [set(bucket) for bucket in B]

    # iterate over ranks
    for i in range(max_rank + 1):
        # find Di = blocks fully contained in Bi
        Di = [block for block in P if block.issubset(B[i])]

        # collapse each block in Di
        for block in Di:
            merge_id = _collapse_block(G, block)
            P.remove(block)
            P.append({merged_id})

        # for each node c of rank i, split larger blocks
        for cid in B[i]:
            center = G.nodes[cid]
            # any block with some members in higher-rank buckets
            higher_blocks = [
                blk for blk in P
                if any(ranks[nid] > i for nid in blk)
            ]
            for blk in higher_blocks:
                C1, C2 = _split_block_by_center(G, blk, center)
                if C1 and C2:
                    P.remove(blk)
                    P.extend([C1, C2])
    
    return G

def _collapse_block(
    G: ConceptGraph,
    ids_to_merge: Set[str]
) -> str:
    """
    merges all EquivalentClass nodes with IDs in ids_to_merge into one.
    Returns the new node's ID.
    """
    from algorithms.data_structures.graph import EquivalentClass, EquivalentClassRelation

    # gather and merge concepts
    ecs = [G.nodes[nid] for nid in sorted(ids_to_merge)]
    merged = EquivalentClass(
        equiv_concepts=list(chain.from_iterable(ec.equiv_concepts for ec in ecs))
    )
    merged.id = "+".join(sorted(ids_to_merge))

    # rewire edges
    new_edges: List[EquivalentClassRelation] = []
    for e in G.edges:
        src_id, tgt_id = e.src.id, e.tgt.id
        new_src = merged if src_id in ids_to_merge else G.nodes[src_id]
        new_tgt = merged if tgt_id in ids_to_merge else G.nodes[tgt_id]
        new_edges.append(EquivalentClassRelation(new_src, new_tgt, e.relation, e.score))

    # replace nodes & edges in the graph
    for nid in ids_to_merge:
        del G.nodes[nid]
    G.nodes[merged.id] = merged
    G.edges = new_edges

    # rebuild adjacency lists
    G.parents.clear()
    G.children.clear()
    for e in new_edges:
        G.parents[e.tgt.id].append(e.src)
        G.children[e.src.id].append(e.tgt)

    return merged.id

def _split_block_by_center(
    G: ConceptGraph,
    block: Set[str],
    center: EquivalentClass
) -> (Set[str], Set[str]):
    """
    splits block into two sets:
        * C1 = IDs adjacent to center
        * C2 = the rest
    """
    # adjacency of center
    adj = {
        ec.id
        for ec in G.parents_of(center)
    } | {
        ec.id
        for ec in G.children_of(center)
    }
    C1 = {nid for nid in block if nid in adj}
    C2 = block - C1
    return C1, C2