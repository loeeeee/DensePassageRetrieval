# New new kTree
# Need the K Nearest Neighbor

import json
import time

VECTOR_LENGTH = 768
K = 50

def main():
    # Read the json
    with open("../.data/example_embeddings_100.json", "r", buffering= 4096) as f:
        example_embeddings = json.load(f)
    # Build K Tree

    # Search
    return

def search():
    # 
    return

def build(vectors: list) -> Node:
    # Segmenting the vectors
    # Build the first layer
    base_layer = []
    for i in range(int(len(vectors)/K)):
        base_layer.append(Node(str(time.time())))
        base_layer[-1].children = [j for j in vector[i*k:(i+1)*k]]
        base_layer[-1].center_point = np.avg(np.asarray(base_layer[-1].children), axis=0)

    isUpdated = True
    while(isUpdated):
        new_base_layer = [Node(str(time.time())) for i in range(int(len(vectors)/K))]
        for old_bl, new_bl in zip(base_layer, new_base_layer):
            new_bl.center = old_bl.center
        for vector in vectors:
            min_distance = 0xefefefef
            flag = -1
            for index, bl in enumerate(new_base_layer):
                if (distance:= np.norm(np.array(vector)-bl.center_point)<min_distance):
                    min_distance = distance
                    flag = index
            new_base_layer[flag].
    # Build the rest of the layers
    layer_created: int = 0

    return

class Node:
    def __init__(self, name: str):
        # Data structure
        self,name = name
        self.center = np.zeros(shape=(768))
        self.children = []
        self.isChildrenEnd = False
