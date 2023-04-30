import logging
import math
import random
import numpy as np
from tqdm import tqdm

class KTree:
    def __init__(self, k: int, vectors: np.array, depth: int) -> None:
        self.vectors = vectors
        self.k = k
        self.depth = depth
        self.root = None

    def construct(self):
        leaf_node = KTreeLeaf(self.k, self.vectors)
        branches = leaf_node.extrusion()
        self.root = KTreeBranch(child_branches=branches)
        for i in range(self.depth-1):
            temp_branches = []
            for branch in branches:
                new_layer_branches: list[KTreeBranch] = branch.child_leaf.extrusion()
                branch.child_branches = new_layer_branches
                temp_branches.extend(new_layer_branches)
                branch.child_leaf = None
            branches = temp_branches.copy()
            
    
    def search(self, vector: np.array) -> list:
        branches = self.root.child_branches
        for i in range(self.depth):
            min_distance = 0x3f3f3f3f
            flag = -1
            for j, branch in enumerate(branches):
                if (distance := branch.distance_to_center_point(vector)) < min_distance:
                    flag = j
                    min_distance = distance
                #print(f"Distance with branch {j}: {distance}")
            #print()
            if i == self.depth - 1:
                leaf = branches[0].child_leaf
            branches = branches[flag].child_branches
        return leaf.vectors
    
    def print(self):
        branches = self.root.child_branches
        for i in range(self.depth):
            temp_branches = []
            for j in branches:
                temp_branches.extend(j.child_branches)
            branches = temp_branches.copy()
            print(f"Floor {i}: {len(branches)}")


class KTreeBranch:
    def __init__(self, child_leaf = None, child_branches: list = [], center_point: np.array = None) -> None:
        self.child_leaf = child_leaf
        self.child_branches = child_branches
        self.center_point = center_point

    def distance_to_center_point(self, vector: np.array) -> None:
        return np.linalg.norm(vector - self.center_point)


class KTreeLeaf:
    def __init__(self, k: int, vectors: np.array) -> None:
        """
        Vectors should be 2-D numpy array
        """
        self.k = k
        self.vectors = vectors
        self.center_point = np.zeros(shape=vectors.shape[1])
        
        self._temp_vectors = []
    
    def update_center_point(self) -> bool:
        """
        Return if it is changed
        """
        old_center_point = np.copy(self.center_point)
        self.center_point = np.mean(self.vectors, axis=0)
        isChanged = True
        if np.allclose(old_center_point, self.center_point):
            isChanged = False
        return isChanged
    
    def append_new(self, vector: np.array) -> None:
        self._temp_vectors.append(vector)

    def finish_append(self) -> None:
        self.vectors = np.stack(self._temp_vectors, axis=0)
        self._temp_vectors = []
    
    def distance_to_center_point(self, vector: np.array) -> None:
        return np.linalg.norm(vector - self.center_point)

    def extrusion(self, max_iter: int = 100) -> KTreeBranch:
        """
        Do KNN optimization
        Return KTreeBranch that connected to lower layer KTreeLeaf
        """
        # Extrusion
        leaves = [KTreeLeaf(self.k,
                            self.vectors[\
                                int(self.vectors.shape[0]*(i)/self.k)\
                                    :int(self.vectors.shape[0]*(i+1)/self.k), :])\
                                        for i in range(self.k)]
        # KNN on the extrusion parts
        for i in range(max_iter):
            isChanged = False
            for j in leaves:
                if j.update_center_point():
                    # If one of the center point changed, continue iteration
                    isChanged = True
            if isChanged == False:
                break
            for j in leaves:
                for vector in j.vectors:
                    min_distance = 0x3f3f3f3f
                    flag = -1
                    for index, leaf in enumerate(leaves):
                        # Compare each vector to center of each node
                        distance = leaf.distance_to_center_point(vector)
                        if distance < min_distance:
                            min_distance = distance
                            flag = index
                    leaves[flag].append_new(vector)
            for j in leaves:
                j.finish_append()
        branches = [KTreeBranch(child_leaf = leaves[i], center_point=leaves[i].center_point) for i in range(self.k)]
        return branches