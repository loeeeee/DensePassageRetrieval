import scipy as sp
class KDTree:
    def __init__(self, vectors, leaf_size: int = 10) -> None:
        self.vectors = vectors
        self.leaf_size = leaf_size

    def construct(self) -> None:
        self.kdtree = sp.spatial.KDTree(self.vectors, leafsize = self.leaf_size)
        return
    
    def search(self, query, top_n: int = 20) -> list:
        return self.kdtree.query(query, k = top_n)
