from collections import deque
from typing import List, Set

from algorithms.data_structures.graph import (
    EquivalentClass, 
    EquivalentClassRelation, 
    ConceptGraph,
)


def neighborhood_sample(
    graph: ConceptGraph,
    center: EquivalentClass,
    max_hops: int = 2
) -> List[EquivalentClass]:
    """
    traverse up to max_hops away from center in the concept graph,
    collecting parents, children, and sibling classes at each step,
    then return the induced subgraph's nodes
    """
    visited: Set[str] = set([center.id])
    result: Set[EquivalentClass] = set()
    queue = deque([(center, 0)])

    while queue:
        node, depth = queue.popleft()
        if depth >= max_hops:
            continue

        # 1-hop neighbors: parents and children of center
        neighbors = graph.parents_of(node) + graph.children_of(node)

        # siblings: children of parents of center
        for p in graph.parents_of(node):
            siblings = graph.children_of(p)
            neighbors.extend([sib for sib in siblings if sib.id != node.id])
        
        for nbr in neighbors:
            if nbr.id in visited:
                continue
            visited.add(nbr.id)
            result.add(nbr)
            queue.append((nbr, depth + 1))

    return list(result)