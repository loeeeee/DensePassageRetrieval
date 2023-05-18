from annoy import AnnoyIndex

class Annoy:
    def __init__(self, vectors, no_of_trees: int = 10, dimension_of_input: int = 768, distance_method: str = "euclidean") -> None:
        """
        distance_method can be one of the following:
        "angular", "euclidean", "manhattan", "hamming", "dot"
        (default to be "euclidean")
        """
        self.vectors = vectors
        self.no_of_trees = no_of_trees
        self.ann = AnnoyIndex(dimension_of_input, distance_method)

    def construct(self) -> None:
        for index, vector in enumerate(self.vectors):
            self.ann.add_item(index, vector)
        self.ann.build(self.no_of_trees)
        return

    def search(self, query, top_n: int = 20) -> list:
        nn = self.ann.get_nns_by_vector(query, top_n)
        return nn
