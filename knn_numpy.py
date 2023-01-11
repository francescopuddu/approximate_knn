import numpy as np


class KnnNode:
    def __init__(self, point, parent, idx):
        self.point = point
        self.parent = parent
        self.idx = idx
        self.l = None           # <=hyperplane
        self.r = None           # > hyperplane
        self.hyperplane = None
        self.n_children = 0

    def add(self, new_point, tree_pointers, idx):
        if self.n_children == 2:                        
            if np.dot(self.hyperplane, new_point) > 0:
                self.r.add(new_point, tree_pointers, idx)
            else:
                self.l.add(new_point, tree_pointers, idx)
        elif self.n_children == 0:                      
            self.l = KnnNode(new_point, self, idx)
            tree_pointers[idx] = self.l
            self.n_children += 1
        else:                                           
            self.hyperplane = self.separating_hyperplane(self.l.point, new_point)
            if np.dot(self.hyperplane, self.l.point) > 0:
                self.r = self.l
                self.l = KnnNode(new_point, self, idx)
                tree_pointers[idx] = self.l
            else:
                self.r = KnnNode(new_point, self, idx)
                tree_pointers[idx] = self.r
            self.n_children += 1

    def separating_hyperplane(self, a, b):
        orthogonal_vector = a - b
        middle_point = (a + b) / 2
        known_term = - np.dot(orthogonal_vector, middle_point)
        orthogonal_vector[-1] = known_term
        return orthogonal_vector

    def knn(self, k, origin_point, partial_nn, nn_size, origin):
        first_child = self.l
        second_child = self.r
        if self.hyperplane is not None and (np.dot(self.hyperplane, origin_point) > 0):
            first_child = self.r
            second_child = self.l

        if first_child is not None and origin != "child":
            nn_size = first_child.knn(k, origin_point, partial_nn, nn_size, "parent")
            if nn_size >= k:
                return nn_size

        if self.point is not None:                                  # will only be None in root
            partial_nn.append(self.idx)
            nn_size += 1
            if nn_size >= k:
                return nn_size

        if second_child is not None:
            nn_size = second_child.knn(k, origin_point, partial_nn, nn_size, "parent")
            if nn_size >= k:
                return nn_size

        if origin != "parent":
            if self.parent is not None:
                nn_size = self.parent.knn(k, origin_point, partial_nn, nn_size, "child")
                return nn_size
            else:
                print("The entire tree does not contain", k, "points to build knn")
                return nn_size
        else:
            return nn_size

class KnnTree:
    def __init__(self):
        self.root = KnnNode(None, None, None)

    def build(self, data_arr, tree_pointers):
        index_arr = np.array(list(range(data_arr.shape[0])))
        np.random.shuffle(index_arr)     
        for idx in index_arr:
            self.root.add(data_arr[idx], tree_pointers, idx)

    def knn(self, tree_pointers, k):
        knn_matrix = []
        for idx, node in enumerate(tree_pointers):
            neighbourhood = []
            node.knn(k, node.point, neighbourhood, 0, "caller")
            knn_matrix.append(neighbourhood)
        return np.array(knn_matrix)