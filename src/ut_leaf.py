import k_tree_new
import json
import numpy as np
with open("../.data/example_embeddings_100.json", "r", buffering= 4096) as f:
    example_embeddings = json.load(f)

vectors = np.array(example_embeddings)
assert vectors.shape == (100, 768)
kt = k_tree_new.KTreeLeaf(2, vectors)
result = kt.extrusion()

print(result[0].center_point)
print(result[1].center_point)

assert result[0].child_leaf.vectors.shape[0] == 69
assert result[1].child_leaf.vectors.shape[0] == 31
