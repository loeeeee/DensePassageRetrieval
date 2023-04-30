import logging
import math
import numpy as np
from tqdm import tqdm

class KTree:
    def __init__(self, all_vectors: list,
                 K: int = 2, 
                 max_depth: int = 5, 
                 max_leaves: int = 2000) -> None:
        self.all_vectors = all_vectors
        self.K = K
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.root = KTreeBranchNode()
        
    def construct(self) -> None:
        """
        The construct will return nothing
        """
        """
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                        @@@@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@                           @@@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@   @@@@@@@@@@@@@@@@@@@@@   (@@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@   @@@@@@@@@@@@@@@@@@@@@    @@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@   @@@@@@@BRANCH@@@@@@@@    @@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@   @@@@@@@@@@@@@@@@@@@@@.   @@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@    @@@@@@@@@@@@@@@@@@@@.   @@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@   ,@@@@@@@@@@@@@@@@@@@@    @@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@.                           @@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    @@@@@@@    @@@@@@@@@@@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@     @@@@@@@@@@@   @@@@@@@@@@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@@@@@@@@@@@@@    @@@@@@@@@@@@@@@   @@@@@@@@@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@@@@@@@@@@     @@@@@@@@@@@@@@@@@    @@@@@@@@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@@@@@@@@    %@@@@@@@@@@@@@@@@@@@@,   @@@@@@@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@@@@@@    @@@@@@@@@@@@@@@@@@@@@@@@@   @@@@@@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@@@     @@@@@@@@@@@@@@@@@@@@@@@@@@@@   @@@@@@@@@@@@@@@@@
        @@@@@@@@@@@@@@@    @@@@@@@@@@@@@@@@@@@@@@@                     /@@@@@@
        @@@@@@@@@@@@@@@  @@@@@@@@@@@@@@@@@@@@@@@@@    @@@@@@@@@@@         &@@@
        @@@                        @@@@@@@@@@@@@@@   #@@@@@@@@@@@@@@@@@    @@@
        @@@@    @@@@@@@@@@@@@@@@   @@@@@@@@@@@@@@@#   @@@@@@@@@@@@@@@@@   @@@@
        @@@@   @@@@@@@@@@@@@@@@@   @@@@@@@@@@@@@@@@   @@@BRANCH@@@@@@@@   @@@@
        @@@@    @@@@BRANCH@@@@@@@   @@@@@@@@@@@@@@@@   @@@@@@@@@@@@@@@@   @@@@
        @@@@@   @@@@@@@@@@@@@@@%   @@@@@@@@@@@@@@@@   @@@@@@@@@@@@@@@@@   @@@@
        @@@@   (@@@@@@@@@@@@@@%    @@@@@@@@@@@@@@@@    @@@@               @@@@
        @@@@*                        @@@@@@@@@@@@@@@            @@@@@@@   @@@@
        @@@@@@@     & @@@@@@@@@   @@@@@@@@@@@@@@@@@@@  @@@@@@   @@@@@@@@@@@@@@
        @@@@@@@@@@@@   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@   #@@@@@@@@@@@@@
        @@@@@@@@@@@@   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    @@@@@@@@@@@@@
        @@@@@@@@@@@@   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%   @@@@@@@@@@@@@
        @@@@@@@@@@@@   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@&   @@@@@@@@@@@@@
        @@@@@@@@@@@@   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    @@@@@@@@@@@@@
        @@@@@@@@@@@@   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    @@@@@@@@@@@@@
        @@@@@@@@@@@@   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                       @
        @@@@@@@  @                 @@@@@@@@@@@@@@@@@   ,@@@@@@@@@@@@@@@@@@#   
        @@@          @@@@@@@@@@@@   @@@@@@@@@@@@@@@@,   @@@@@@@@@@@@@@@@@@   @
        @@@   @@@@@@@@@@@@@@@@@@@   @@@@@@@@@@@@@@@@@   @@@@@@@@@@@@@@@@@@   @
        @@@*   @@@@@@@@@@@@@@@@@@   @@@@@@@@@@@@@@@@@   @@@@@@@@@@@@@@@@@@   @
        @@@@   @@@@@@LEAF@@@@@@@@   @@@@@@@@@@@@@@@@@   @@@@@@LEAF@@@@@@@@   @
        @@@@   @@@@@@@@@@@@@@@@@@   @@@@@@@@@@@@@@@@@   @@@@@@@@@@@@@@@@@    @
        @@@@   @@@@@@@@@@@@@@@@@@   @@@@@@@@@@@@@@@@@.   @@@@@@@@@@@@@@@@   @@
        @@@@   @@@@@@@@@@@@@@@@@@   @@@@@@@@@@@@@@@@@@                       @
        @@@@@      @@@@@@@@@@@@@(   @@@@@@@@@@@@@@@@@@     #@@@@@@@@@@@@    @@
        @@@@@                      @@@@@@@@@@@@@@@@@@@   @@@@@@@@@@@@@@@   @@@
        @@@@@@@@@@@@@@@@@@@@@@@@  .@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        """
        leaf_nodes = [KTreeLeafNode(child_node = self.all_vectors)]
        branch_nodes = [self.root]
        for i in range(self.max_depth):
            temp_leaf_nodes = []
            temp_branch_nodes = []
            # KNN
            for j in leaf_nodes:
                knn = KNearestNeighbor(j, K=self.K)
                temp_leaf_nodes.extend(knn.construct())

            # Tree building
            # Create branch for next layer
            temp_branch_nodes = [j.to_branch() for j in temp_leaf_nodes]
            # Connect branches
            branch_cnt = 0
            for j in branch_nodes:
                j.extend(temp_branch_nodes[branch_cnt:branch_cnt+self.K])
                branch_cnt += self.K
            # Update current branch
            branch_nodes = temp_branch_nodes
            # Update current leaf
            leaf_nodes = temp_leaf_nodes
        # Connect the leaf to the branch
        branch_cnt = 0
        for j in branch_nodes:
            j.extend(leaf_nodes[branch_cnt:branch_cnt+self.K])
            branch_cnt += self.K
        return
    
    def search(self, query) -> list:
        current_node = self.root
        while(isinstance(current_node, KTreeBranchNode)):
            # Find the closet branch
            min_distance = 0x3f3f3f3f
            flag = None
            for branch in current_node:
                distance = np.linalg.norm(branch.center_point - query)
                if min_distance > distance:
                    min_distance = distance
                    flag = branch
            current_node = flag
        # Calculate the distance between the query and rest of the leaves in this branch
        distances_and_vector = []
        for leaf in current_node:
            distance = np.linalg.norm(leaf - query)
            distances_and_vector.append((distance, leaf))
        distances_and_vector = sorted(distances_and_vector, key = lambda x: x[0])
        return distances_and_vector
    
    def dump(self, path) -> None:
        return
    
    def load(self, load) -> None:
        return


class KTreeBranchNode:
    def __init__(self, center_point: tuple = (0,0), child_node: list = []) -> None:
        """
        Simplified node that does not store any actual document vector, but only search index vector
        """
        self.center_point = center_point
        self.child_node = child_node
        self._iter_index = -1

    def __repr__(self) -> str:
        return self.center_point
    
    def __iter__(self):
        self._iter_index = -1
        while self._iter_index < len(self.child_node) - 1:
            self._iter_index += 1
            yield self.child_node[self._iter_index]
        raise StopIteration
    
    def append(self, child_node) -> None:
        self.child_node.append(child_node)

    def extend(self, list_of_child_nodes: list) -> None:
        self.child_node.extend(list_of_child_nodes)


class KTreeLeafNode:
    def __init__(self, child_node: list = [], shape: tuple = (768,)) -> None:
        """
        center_point is the average for all the child nodes in the tree.
        child_node may include KTreeLeaf or KTreeNode, but cannot contain both.
        """
        self.shape = shape
        self.center_point = np.empty(shape)
        self.child_node_ng = child_node
        self.child_node = []
        
    def __len__(self):
        return len(self.child_node)

    def update_center_point(self) -> bool:
        """
        Return if the point has been updated
        """
        self.child_node = self.child_node_ng.copy()
        print(self.child_node)
        # Reset for the next iteration
        self.child_node_ng = []
        old_center_point = np.copy(self.center_point)
        self.center_point = np.mean(np.array(self.child_node), axis=0)
        if isinstance(old_center_point, type(None)):
            #print(old_center_point)
            return True
        elif np.round(old_center_point - self.center_point, decimals=8).all():
            #print(np.round(old_center_point - self.center_point, decimals=8).tolist())
            return False
        else:
            #print(np.round(old_center_point - self.center_point, decimals=8).tolist())
            #print("Hello")
            return True

    def append(self, new_node):
        self.child_node_ng.append(new_node)
        
    def to_branch(self):
        return KTreeBranchNode(self.center_point)
    
    def split(self, K: int):
        result = [KTreeLeafNode(self.child_node\
                                       [int(len(self)/K*i)\
                                        :int(len(self)/K*(i+1))])\
                                              for i in range(K)]
        return result


class KNearestNeighbor:
    def __init__(self, k_tree_node: KTreeLeafNode, K: int = 2) -> None:
        self.K = K
        self.logger = logging.getLogger("utils")
        self.result_nodes: list[KTreeLeafNode] = k_tree_node.split(K)
        #print(f"Node content: {self.result_nodes[0].child_node}")
        for i in self.result_nodes:
            i.update_center_point()

    def construct(self, max_iter: int = 100) -> list[KTreeLeafNode]:
        """
        Return a list of KTreeLeafNode
        """
        print("Construct is called.")
        iter_cnt = 0
        print(f"Node content: {self.result_nodes[0].child_node}")
        while(iter_cnt <= max_iter):
            isUpdated: bool = False
            """
            Stop if it converges
            """
            self.logger.info(f"Iteration: {iter_cnt}")
            #bar = tqdm(total= sum([len(node) for node in self.result_nodes]))
            print(f"Result node length: {len(self.result_nodes)}")
            for node in self.result_nodes:
                # Iter through each node
                vector_cnt = 0
                print(f"Child node length: {len(node)}")
                for vector in node.child_node:
                    vector_cnt += 1
                    # Iter through each vector
                    vector = np.array(vector)
                    min_distance = 0x3f3f3f3f
                    cnt = 0
                    flag = -1
                    for result_node in self.result_nodes:
                        # Compare each vector to center of each node
                        distance = np.linalg.norm(vector-result_node.center_point)
                        if distance < min_distance:
                            min_distance = distance
                            flag = cnt
                        cnt += 1
                    # Update to the closest node
                    node[flag].append(vector)
                    #bar.update()
            # Update the center point
            for node in self.result_nodes:
                isUpdated += node.update_center_point()
                #print(isUpdated)
            if bool(isUpdated) == False:
                break
            iter_cnt += 1
        
        if iter_cnt >= max_iter:
            self.logger.info(f"Max iteration ({max_iter}) limit reached.")
        elif isUpdated == False:
            self.logger.info(f"Local minimum reached.")

        print("Finished")
        return self.result_nodes
 
