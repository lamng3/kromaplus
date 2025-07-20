from typing import Optional, List, Set, Dict

class Concept:
    """concepts of an ontology"""
    def __init__(
        self, 
        name: str, 
        parent: Optional[List[Concept]] = None,
        children: Optional[List[Concept]] = None, 
        ground_set: Optional[List[str]] = None,
    ):
        self.name = name
        self.rank = None # [issue] can we make rank update if ontology dynamically updates? 
        self.parents: List[Concept] = parents if parents is not None else []
        self.children: List[Concept] = children if children is not None else []
        self.ground_set: List[str] = ground_set if ground_set is not None else []

    def rank(self) -> int:
        # [issue] can we defer update to concept graph update or recompute rank?
        if self.rank is None:
            self.rank = 1 + max((child.rank() for child in self.children), default=0)
        return self.rank

    def __repr__(self) -> str:
        return f"Concept(name={self.name!r}, rank={self.rank}, ground_set={self.ground_set})"

class ConceptRelation:
    """relations between concepts"""
    def __init__(self, src: Concept, tgt: Concept, score: float = 0.0):
        self.src = src
        self.tgt = tgt
        self.score = score

    def __repr__(self) -> str:
        return f"ConceptRelation({self.src.name} -> {self.tgt.name}, score={self.score})"

class EquivalentClass:
    """bisimilar concepts treated as one equivalence class"""
    def __init__(self, equiv_concepts: List[Concept]):
        if not equiv_concepts:
            raise ValueError("EquivalentClass requires at least one Concept")
        self.equiv_concepts = equiv_concepts
        self.id: str = equiv_concepts[0].name # use first concept name as id
        self.rank = None # [issue] can we make rank update if ontology dynamically updates? 
        self.parents: List[EquivalentClass] = parents if parents is not None else []
        self.children: List[EquivalentClass] = children if children is not None else []

    def rank(self) -> int:
        # [issue] can we defer update to concept graph update or recompute rank?
        if self.rank is None:
            self.rank = 1 + max((child.rank() for child in self.children), default=0)
        return self.rank

    def __repr__(self) -> str:
        names = [c.name for c in self.equiv_concepts]
        return f"EquivalentClass(id={self.id!r}, concepts={names})"

class EquivalentClassRelation:
    """relations between equivalent classes"""
    def __init__(self, src: EquivalentClass, tgt: EquivalentClass, score: float = 0.0):
        self.src = src
        self.tgt = tgt
        self.score = score

    # [issue] test representation of this repr
    def __repr__(self) -> str:
        return (f"EquivalentClassRelation("
                f"src={self.src!r}, "
                f"tgt={self.tgt!r}, "
                f"score={self.score})")

class ConceptGraph:
    """a DAG whose nodes are equivalent classes"""
    def __init__(self, nodes: List[EquivalentClass], edges: List[EquivalentClassRelation]):
        self.nodes: Dict[str, EquivalentClass] = {n.id: n for n in nodes} # fast node lookup
        self.edges: List[EquivalentClassRelation] = []
        self.parents:  Dict[str, List[EquivalentClass]] = defaultdict(list)
        self.children: Dict[str, List[EquivalentClass]] = defaultdict(list)
        # add initial edges
        for e in edges:
            self.add_edge(e)

    def add_node(self, node: EquivalentClass) -> bool:
        """add an equivalent class to concept graph, return status"""
        if node is not None:
            self.nodes[node.id] = node
            return True
        return False
    
    def add_edge(self, edge: EquivalentClassRelation) -> bool:
        """add a directed relation between two equivalence classes"""
        self.edges.append(edge)
        self.parents[edge.src.id].append(edge.tgt)
        self.children[edge.tgt.id].append(edge.src)
        # ensure both nodes exist in concept graph
        self.nodes.setdefault(edge.src.id, edge.src)
        self.nodes.setdefault(edge.tgt.id, edge.tgt)

    def parents_of(self, node: EquivalentClass) -> List[EquivalentClass]:
        return list(self.parent[node.id])

    def children_of(self, node: EquivalentClass) -> List[EquivalentClass]:
        return list(self.children[node.id])

    def compute_ranks(self) -> Dict[str, int]:
        """
        bottom-up rank computation
        rank(node): 0, if no children
                    1 + max{ rank(child) }, otherwise
        memoizes into each equivalence class rank
        """
        rank: Dict[str, int] = {}
        
        def dfs(node: EquivalentClass) -> int:
            if node.id in ranks:
                return ranks[node.id]
            children = self.children.get(node.id, [])
            r = 0 if not children else 1 + max(dfs(child) for child in children)
            ranks[node.id] = r
            node.rank = r
            return r

        for node in self.nodes.values():
            dfs(node)

    def __repr__(self) -> str:
        return (f"ConceptGraph(num_nodes={len(self.nodes)}, "
                f"num_edges={len(self.edges)})")

    def __str__(self) -> str:
        lines = [repr(self)]
        for node in self.nodes.values():
            lines.append(f"  {node!r}  children -> {[c.id for c in self.children_of(node)]}")
        return "\n".join(lines)
                
