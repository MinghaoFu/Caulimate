import numpy as np
import itertools
import networkx as nx

from Caulimate.Utils.Tools import check_array

def supper_set(Bs):
    '''
        Compute the superset of a list of matrices
    '''
    return np.where(np.sum(Bs, axis=0) > 0, 1, 0)

def is_dag(B):
    """Check whether B corresponds to a DAG.

    Args:
        B (numpy.ndarray): [d, d] binary or weighted matrix.
    """
    return nx.is_directed_acyclic_graph(nx.DiGraph(B))

def threshold_till_dag(B):
    """Remove the edges with smallest absolute weight until a DAG is obtained.

    Args:
        B (numpy.ndarray): [d, d] weighted matrix.

    Returns:
        numpy.ndarray: [d, d] weighted matrix of DAG.
        float: Minimum threshold to obtain DAG.
    """
    B = check_array(B)
    B = np.copy(B)
    if is_dag(B):
        return B, 0

    # Get the indices with non-zero weight
    nonzero_indices = np.where(B != 0)
    # Each element in the list is a tuple (weight, j, i)
    weight_indices_ls = list(zip(B[nonzero_indices],
                                 nonzero_indices[0],
                                 nonzero_indices[1]))
    # Sort based on absolute weight
    sorted_weight_indices_ls = sorted(weight_indices_ls, key=lambda tup: abs(tup[0]))

    for weight, j, i in sorted_weight_indices_ls:
        if is_dag(B):
            # A DAG is found
            break

        # Remove edge with smallest absolute weight
        B[j, i] = 0
        dag_thres = abs(weight)

    return B, dag_thres

def decycle_till_dag(B):
    """Remove the edges in the shortest cycle with the smallest absolute weight until a DAG is obtained.

    Args:
        B (numpy.ndarray): [d, d] weighted matrix.

    Returns:
        numpy.ndarray: [d, d] weighted matrix of DAG.
        float: Minimum threshold to obtain DAG.
    """
    # Convert the numpy matrix to a directed graph
    G = nx.DiGraph(B)

    # Initialize the minimum threshold to infinity
    min_threshold = float('inf')
    
    # Function to find the shortest cycle in terms of number of nodes
    def find_shortest_cycle(graph):
        try:
            cycles = list(nx.simple_cycles(graph))
            # Find the cycle with the minimum length
            shortest = min(cycles, key=len)
            return shortest
        except:
            return None

    # Repeat until there are no more cycles
    while True:
        # Find the shortest cycle
        cycle = find_shortest_cycle(G)
        if not cycle:
            break
        
        # Find the edge with the minimum absolute weight in the shortest cycle
        min_edge = None
        min_weight = float('inf')
        for u, v in zip(cycle, cycle[1:] + [cycle[0]]):
            weight = abs(G[u][v]['weight'])
            if weight < min_weight:
                min_weight = weight
                min_edge = (u, v)
        
        # Update the minimum threshold
        min_threshold = min(min_threshold, min_weight)

        # Remove the minimum weight edge from the graph
        G.remove_edge(*min_edge)

    # Convert the graph back to a numpy matrix
    dag_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()), dtype=float)

    return dag_matrix

def postprocess_dag(B, graph_thres=0.):
    """Post-process estimated solution:
        (1) Thresholding.
        (2) Remove the edges with smallest absolute weight until a DAG
            is obtained.

    Args:
        B (numpy.ndarray): [d, d] weighted matrix.
        graph_thres (float): Threshold for weighted matrix. Default: 0.3.

    Returns:
        numpy.ndarray: [d, d] weighted matrix of DAG.
    """
    B = np.copy(B)

    B[np.abs(B) <= graph_thres] = 0    # Thresholding
    B, _ = threshold_till_dag(B)

    return B

def is_pseudo_invertible(matrix):
    """
    Check if a given n x m matrix is pseudo-invertible.

    Parameters:
    - matrix (np.ndarray): An n x m matrix.

    Returns:
    - bool: True if the matrix is pseudo-invertible, False otherwise.
    """
    # Compute the singular value decomposition (SVD) of the matrix
    U, S, Vh = np.linalg.svd(matrix)

    # A matrix is pseudo-invertible if it has no zero singular values
    # Since singular values are non-negative, we check if they are all non-zero
    return np.all(S > np.finfo(float).eps)

def eudistance_mask(coords, max_eud):
    coords = check_array(coords)
    # Calculate the squared differences along each dimension
    diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    # Sum the squared differences to get squared Euclidean distances
    sq_distances = np.sum(diffs ** 2, axis=2)
    # Generate the mask: 1 if the squared distance is less than or equal to the squared threshold, 0 otherwise
    mask = (sq_distances <= max_eud ** 2).astype(int)
    np.fill_diagonal(mask, 0)

    return mask

def corrupt_dag(adj_matrix, n, m, seed=42):
    """Randomly set n elements to 0 and add m zero-elements with a random value

    Args:
        adj_matrix (_type_): _description_
        n (_type_): _description_
        m (_type_): _description_

    Returns:
        _type_: _description_
    """
    size = adj_matrix.shape[0]
    non_diag_indices = np.where(~np.eye(size, dtype=bool))  # Indices of non-diagonal elements
    # Add m zero-elements with a random value
    zero_indices = np.where(adj_matrix[non_diag_indices] == 0)[0]
    rs = np.random.RandomState(seed)
    indices_to_fill = rs.choice(zero_indices, m, replace=False)
    for index in indices_to_fill:
        adj_matrix[non_diag_indices[0][index], non_diag_indices[1][index]] = np.random.randint(1, 10)

    # Set n elements to 0
    non_zero_indices = np.nonzero(adj_matrix[non_diag_indices])[0]
    indices_to_zero = rs.choice(non_zero_indices, n, replace=False)
    adj_matrix[non_diag_indices[0][indices_to_zero], non_diag_indices[1][indices_to_zero]] = 0

    # Check if the graph is still a DAG
    # If not, undo the last modification and try again
    # This part is left as an exercise for the reader

    return adj_matrix