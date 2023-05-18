from lshashpy3 import LSHash

class LSH:
    def __init__(self, vectors, no_of_hash_tables: int = 1, no_of_hash_bits: int = 3, dimension_of_input: int = 768) -> None:
        self.vectors = vectors
        self.no_of_hash_tables = no_of_hash_tables
        self.no_of_hash_bits = no_of_hash_bits
        self.dimension_of_input = dimension_of_input

        # Runtime variable
        self.hash_table = LSHash(no_of_hash_bits, dimension_of_input, num_hashtables=no_of_hash_tables)
        
    def construct(self) -> None:
        for vector in self.vectors:
            self.hash_table.index(vector)
        return
    
    def search(self, query, top_n: int = 20) -> list:
        nn = self.hash_table.query(query, num_results = top_n)
        return nn
