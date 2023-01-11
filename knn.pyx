import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from cython.operator cimport dereference as deref


cdef struct KnnNode:
    double* point
    KnnNode* parent
    int idx
    KnnNode* l
    KnnNode* r
    double* hyperplane
    int n_children


cdef KnnNode* node_init(double* point, KnnNode* parent, int idx, KnnNode* all_nodes):
    cdef KnnNode* new_node = &all_nodes[idx]
    deref(new_node).point = point
    deref(new_node).parent = parent
    deref(new_node).idx = idx
    deref(new_node).l = NULL
    deref(new_node).r = NULL
    deref(new_node).hyperplane = NULL
    deref(new_node).n_children = 0
    return new_node

cdef double dot_prod(double* a, double* b, int size):
    cdef int i
    cdef double res = 0
    for i in range(size):
        res += a[i]*b[i]
    return res

cdef double* separating_hyperplane(double* a, double* b, int n_dims, double* all_hyperplanes, int idx, int n_points):
    cdef double* orthogonal_vector = &all_hyperplanes[idx*(n_dims+1)]   #array of size n_dims + 1
    cdef double* middle_point = <double*> malloc((n_dims+1) * sizeof(double))   #array of size n_dims + 1
    cdef int i
    cdef double known_term = 0
    
    for i in range(n_dims + 1):
        orthogonal_vector[i] = a[i] - b[i]
        middle_point[i] = (a[i] + b[i])/2
        known_term -= orthogonal_vector[i]*middle_point[i]

    orthogonal_vector[n_dims] = known_term
    free(middle_point)

    return orthogonal_vector

cdef void node_add(KnnNode* self_ptr, double* new_point, KnnNode* tree_pointers[], int idx, int n_dims, KnnNode* all_nodes, double* all_hyperplanes, int n_points):
    cdef KnnNode self_copy = deref(self_ptr)
    cdef KnnNode left_child
    cdef double* left_point
    if self_copy.n_children == 2:                        

        if dot_prod(self_copy.hyperplane, new_point, n_dims+1) > 0:
            node_add((self_ptr[0]).r, new_point, tree_pointers, idx, n_dims, all_nodes, all_hyperplanes, n_points)
        else:
            node_add((self_ptr[0]).l, new_point, tree_pointers, idx, n_dims, all_nodes, all_hyperplanes, n_points)

    elif self_copy.n_children == 0:                      

        (self_ptr[0]).l = node_init(new_point, self_ptr, idx, all_nodes)

        tree_pointers[idx] = (self_ptr[0]).l
        (self_ptr[0]).n_children += 1

    else:                                           

        left_child = deref((self_ptr[0]).l)

        left_point = left_child.point
    
        (self_ptr[0]).hyperplane = separating_hyperplane(left_point, new_point, n_dims, all_hyperplanes, (self_ptr[0]).idx, n_points)

        if dot_prod((self_ptr[0]).hyperplane, left_point, n_dims + 1) > 0:
            (self_ptr[0]).r = (self_ptr[0]).l
            (self_ptr[0]).l = node_init(new_point, self_ptr, idx, all_nodes)
            tree_pointers[idx] = (self_ptr[0]).l
        else:
            (self_ptr[0]).r = node_init(new_point, self_ptr, idx, all_nodes)
            tree_pointers[idx] = (self_ptr[0]).r
        (self_ptr[0]).n_children += 1


cdef int node_knn(KnnNode self, int k, double* origin_point, int* partial_nn, int nn_size, int origin, int n_dims):
    # origin:
    # 0 = parent
    # 1 = child
    # 2 = caller
    cdef KnnNode* first_child = self.l
    cdef KnnNode* second_child = self.r
    if self.hyperplane != NULL and (dot_prod(self.hyperplane, origin_point, n_dims+1) > 0):
        first_child = self.r
        second_child = self.l

    # try descending the correct way
    if first_child != NULL and origin != 1:
        nn_size = node_knn(deref(first_child), k, origin_point, partial_nn, nn_size, 0, n_dims)
        if nn_size >= k:
            return nn_size

    # after going the correct way, pick up this point's node
    if self.point != NULL:
        partial_nn[nn_size] = self.idx
        nn_size += 1
        if nn_size >= k:
            return nn_size

    # after picking up correct way and own point, descend the wrong way
    if second_child != NULL:
        nn_size = node_knn(deref(second_child), k, origin_point, partial_nn, nn_size, 0, n_dims)
        if nn_size >= k:
            return nn_size

    # after checking entire subtree, if the neighbourhood is not yet full, go up the tree
    if origin != 0:
        if self.parent != NULL:
            nn_size = node_knn(deref(self.parent), k, origin_point, partial_nn, nn_size, 1, n_dims)
        else:
            print("The entire tree does not contain", k, "points to build knn")
    return nn_size


cdef struct KnnTree:
    KnnNode* root

cdef KnnTree tree_init(KnnNode* all_nodes, int n_points):
    cdef KnnNode* root_node = node_init(NULL, NULL, n_points, all_nodes)
    return KnnTree(root_node)

cdef void tree_build(KnnTree* self_tree_ptr, double[:, :] data_view, KnnNode* tree_pointers[], int n_points, int n_dims, KnnNode* all_nodes, double* all_hyperplanes):
    cdef np.ndarray index_arr = np.array(list(range(n_points)), dtype=np.int32)
    np.random.shuffle(index_arr)     
    cdef int[:] index_arr_view = index_arr
    cdef int i, idx
    cdef double* new_point
    for i in range(n_points):
        idx = index_arr_view[i]
        new_point = &data_view[idx][0]

        node_add((self_tree_ptr[0]).root, new_point, tree_pointers, idx, n_dims, all_nodes, all_hyperplanes, n_points)


cdef void tree_knn(KnnTree tree, KnnNode** tree_pointers, int k, int* knn_matrix, int n_dims, int n_points, double[:, :] data_view):
    cdef int i
    for i in range(n_points):
        node_knn(deref(tree.root), k, &data_view[i][0], &knn_matrix[i*k], 0, 2, n_dims)


cdef struct Wrapper:
    double* arr

cdef void numpy_to_c(np_data):
    cdef double [:, :] narr_view = np_data
    cdef Wrapper w = Wrapper(&narr_view[0][0])

cdef void print_knn_matrix(int* data, int k, int n_points):
    cdef int i, j
    for i in range(n_points):
        for j in range(k):
            print(data[i*k + j],)
        print()


cpdef np.ndarray tree_and_knn(np.ndarray data, int n_dims, int n_points, int k):
    # adapting input data
    cdef homogeneous_data = np.pad(data, ((0, 0), (0, 1)), mode='constant', constant_values=1)

    # creating data structures
    cdef KnnNode* node_array = <KnnNode*> malloc((n_points+1) * sizeof(KnnNode))        
    cdef double* hyperplane_array = <double*> malloc((n_points+1) * (n_dims+1) * sizeof(double))
    cdef KnnNode** tree_pointers = <KnnNode**> malloc(n_points * sizeof(KnnNode*))  
    cdef KnnTree tree = tree_init(node_array, n_points)
    cdef int* neighbourhood_array = <int*> malloc(n_points*k*sizeof(int))

    # tree operations
    tree_build(&tree, homogeneous_data_view, tree_pointers, n_points, n_dims, node_array, hyperplane_array)
    tree_knn(tree, tree_pointers, k, neighbourhood_array, n_dims, n_points, homogeneous_data_view)

    # back to numpy
    cdef int[:] knn_matrix_view
    knn_matrix_view = <int[:n_points*k]> neighbourhood_array
    cdef np.ndarray knn_matrix_np = np.empty(n_points*k, dtype=int)
    knn_matrix_np[...] = knn_matrix_view[...]
    knn_matrix_np = knn_matrix_np.reshape((n_points, k))


    free(node_array)
    free(hyperplane_array)
    free(tree_pointers)
    free(neighbourhood_array)

    return knn_matrix_np