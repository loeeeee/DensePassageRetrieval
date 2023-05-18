import faiss

class HierarchicalNavigableSmallWorld:
    def __init__(self, vectors, vector_size: int = 768, M: int = 32,) -> None:
        self.vectors = vectors
        self.leaf_size = leaf_size

    def construct(self) -> None:
        self.kdtree = sp.spatial.KDTree(self.vector, leafsize = self.leaf_size)
        return
    
    def search(self, query, top_n: int = 20) -> list:
        return self.kdtree.query(query, k = top_n)
