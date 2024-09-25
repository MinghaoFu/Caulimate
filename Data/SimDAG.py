import numpy as np
import networkx as nx
import igraph as ig
from einops import repeat

from Caulimate.Utils.GraphUtils import threshold_till_dag

from Caulimate.Utils.Tools import bin_mat

def simulate_time_vary_weight(
                            B_bin, 
                            num, 
                            func, 
                            t_period,
                            B_ranges, 
                            t_start=0, 
                            seed=42):
    """
    bijt = _bij + f(t)
    """
    #B_bin = bin_mat(np.load('/home/minghao.fu/workspace/LatentTimeVaryingCausalDiscovery/results/2024-04-01_23-15-12/synthetic_5_5_20240401-231512/epoch_1000/ground_truth.npy'))[0]
    dim1, dim2 = B_bin.shape
    B = simulate_weight(B_bin, B_ranges, seed)
    t_ids = np.linspace(t_start, t_start + num, num=num)
    t_factor = np.linspace(0.5, 1.5, num) * np.array([func(t_idx * 2 * np.pi / t_period) for t_idx in t_ids]) + 0.5 * np.array([func(t_idx * 2 * np.pi / 12) for t_idx in t_ids])
    Bs = repeat(t_factor, 'i -> i j k', j=dim1, k=dim2) * B_bin + B
    return Bs, B

def simulate_tree_adj_mat(input_dim, output_dim, seed=1, i_inv=True):
    # Z -> X, tree structure, for L @ C
    rs = np.random.RandomState(seed)
    C = np.zeros([input_dim, output_dim])
    for i in range(output_dim):
        C[rs.randint(0, input_dim), i] = 1
    
    return C


def simulate_sparse_dag(d, p, seed, i_inv=True):
    def _simulate_sparse_dag(d, p, seed):
        G = nx.gnp_random_graph(d, p, directed=True, seed=seed)
        # while not nx.is_directed_acyclic_graph(G):
        #     G = nx.gnp_random_graph(d, p, directed=True, seed=seed)
        B_bin = nx.adjacency_matrix(G).toarray()
        B_bin, _ = threshold_till_dag(B_bin)
    
    while True:
        B_bin = _simulate_sparse_dag(d, p, seed)
        if not i_inv or np.linalg.det(np.eye(d) - B_bin) != 0:
            break
    return B_bin

def simulate_random_dag(d, degree, graph_type, seed):
    """Simulate random DAG.

    Args:
        d (int): Number of nodes.
        degree (int): Degree of graph.
        graph_type ('ER' or 'SF'): Type of graph.
        rs (numpy.random.RandomState): Random number generator.
            Default: np.random.RandomState(1).

    Returns:
        numpy.ndarray: [d, d] binary adjacency matrix of DAG.
    """
    rs = np.random.RandomState(seed)
    
    def _random_permutation(B_bin):
        # np.random.permutation permutes first axis only
        P = rs.permutation(np.eye(B_bin.shape[0]))
        return P.T @ B_bin @ P
    
    def simulate_er_dag(d, degree, rs=np.random.RandomState(1)):
        """Simulate ER DAG using NetworkX package.

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """
        def _get_acyclic_graph(B_und):
            return np.tril(B_und, k=-1)

        def _graph_to_adjmat(G):
            return nx.to_numpy_array(G)

        p = float(degree) / (d - 1)
        G_und = nx.generators.erdos_renyi_graph(n=d, p=p, seed=rs)
        B_und_bin = _graph_to_adjmat(G_und)    # Undirected
        B_bin = _get_acyclic_graph(B_und_bin)
        return B_bin

    def simulate_sf_dag(d, degree):
        """Simulate ER DAG using igraph package.

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """
        def _graph_to_adjmat(G):
            return np.array(G.get_adjacency().data)

        m = int(round(degree / 2))
        # igraph does not allow passing RandomState object
        G = ig.Graph.Barabasi(n=d, m=m, directed=True)
        B_bin = np.array(G.get_adjacency().data)
        return B_bin

    if graph_type == 'ER':
        B_bin = simulate_er_dag(d, degree, rs)
    elif graph_type == 'SF':
        B_bin = simulate_sf_dag(d, degree)
    else:
        raise ValueError("Unknown graph type.")
    return _random_permutation(B_bin)

def simulate_weight(B_bin, B_ranges, seed):
    """Simulate the weights of B_bin.

    Args:
        B_bin (numpy.ndarray): [d, d] binary adjacency matrix of DAG.
        B_ranges (tuple): Disjoint weight ranges.
        rs (numpy.random.RandomState): Random number generator.
            Default: np.random.RandomState(1).

    Returns:
        numpy.ndarray: [d, d] weighted adjacency matrix of DAG.
    """
    rs = np.random.RandomState(seed)
    B = np.zeros(B_bin.shape)
    S = rs.randint(len(B_ranges), size=B.shape)  # Which range
    for i, (low, high) in enumerate(B_ranges):
        U = rs.uniform(low=low, high=high, size=B.shape)
        B += B_bin * (S == i) * U
    return B