import numpy as np
import scipy.stats as stats
from .Tools import check_array
import networkx as nx

from Caulimate.Utils.GraphUtils import is_dag
from Caulimate.Utils.Tools import bin_mat

def correlation(x, y, method='Pearson'):
    """Evaluate correlation
     Args:
         x: data to be sorted
         y: target data
     Returns:
         corr_sort: correlation matrix between x and y (after sorting)
         sort_idx: sorting index
         x_sort: x after sorting
         method: correlation method ('Pearson' or 'Spearman')
     """

    # print("Calculating correlation...")

    x = x.copy()
    y = y.copy()
    dim = x.shape[0]

    # Calculate correlation -----------------------------------
    if method=='Pearson':
        corr = np.corrcoef(y, x)
        corr = corr[0:dim,dim:]
    elif method=='Spearman':
        corr, pvalue = stats.spearmanr(y.T, x.T)
        corr = corr[0:dim, dim:]

    # Sort ----------------------------------------------------
    munk = Munkres()
    indexes = munk.compute(-np.absolute(corr))

    sort_idx = np.zeros(dim)
    x_sort = np.zeros(x.shape)
    for i in range(dim):
        sort_idx[i] = indexes[i][1]
        x_sort[i,:] = x[indexes[i][1],:]

    # Re-calculate correlation --------------------------------
    if method=='Pearson':
        corr_sort = np.corrcoef(y, x_sort)
        corr_sort = corr_sort[0:dim,dim:]
    elif method=='Spearman':
        corr_sort, pvalue = stats.spearmanr(y.T, x_sort.T)
        corr_sort = corr_sort[0:dim, dim:]

    return corr_sort, sort_idx, x_sort


def mean_correlation_coefficient(data1, data2):
    """
    Calculate the mean correlation coefficient between corresponding rows of two 2D arrays.
    The input arrays must have the same shape (batch, dim).

    Parameters:
    - data1: First array-like of shape (batch, dim).
    - data2: Second array-like of shape (batch, dim), correlated against data1.

    Returns:
    - mean_corr (float): The mean correlation coefficient across all rows.
    """
    # Ensure the input arrays have the same shape
    data1 = check_array(data1)
    data2 = check_array(data2)
    if data1.shape != data2.shape:
        raise ValueError("Input arrays must have the same shape.")

    batch_size = data1.shape[0]
    
    # Center the data
    data1_centered = data1 - np.mean(data1, axis=1, keepdims=True)
    data2_centered = data2 - np.mean(data2, axis=1, keepdims=True)

    # Compute the covariance between the corresponding rows
    covariance = np.sum(data1_centered * data2_centered, axis=1)

    # Compute the standard deviations of the rows
    std_data1 = np.sqrt(np.sum(data1_centered ** 2, axis=1))
    std_data2 = np.sqrt(np.sum(data2_centered ** 2, axis=1))

    # Compute correlation coefficients for each row
    correlation_coeffs = covariance / (std_data1 * std_data2)

    # Compute the mean correlation coefficient
    mean_corr = np.mean(correlation_coeffs)

    return mean_corr

def rank_correlation_coefficient(data1, data2):
    """
    Calculate the mean Spearman's rank correlation coefficient between corresponding rows
    of two 2D arrays. The input arrays must have the same shape (batch, dim).

    Parameters:
    - data1: First array-like of shape (batch, dim).
    - data2: Second array-like of shape (batch, dim), correlated against data1.

    Returns:
    - mean_rank_corr (float): The mean Spearman's rank correlation coefficient across all rows.
    """
    data1 = check_array(data1)
    data2 = check_array(data2)
    if data1.shape != data2.shape:
        raise ValueError("Input arrays must have the same shape.")

    rank_corrs = []

    for i in range(data1.shape[0]):
        # Calculate Spearman's rank correlation for each pair of rows
        corr, _ = scipy.stats.spearmanr(data1[i], data2[i])
        rank_corrs.append(corr)

    # Compute the mean of the correlation coefficients
    mean_rank_corr = np.mean(rank_corrs)

    return mean_rank_corr


# count graph accuracy
def count_graph_accuracy(B_bin_true, B_bin_est, check_input=False):
    # TODO F1 score
    """Compute various accuracy metrics for B_bin_est.

    true positive = predicted association exists in condition in correct direction.
    reverse = predicted association exists in condition in opposite direction.
    false positive = predicted association does not exist in condition.

    Args:
        B_bin_true (np.ndarray): [d, d] binary adjacency matrix of ground truth. Consists of {0, 1}.
        B_bin_est (np.ndarray): [d, d] estimated binary matrix. Consists of {0, 1, -1}, 
            where -1 indicates undirected edge in CPDAG.

    Returns:
        fdr: (reverse + false positive) / prediction positive.
        tpr: (true positive) / condition positive.
        fpr: (reverse + false positive) / condition negative.
        shd: undirected extra + undirected missing + reverse.
        normalized shd: shd / total possible edges
        pred_size: prediction positive.

    Code modified from:
        https://github.com/xunzheng/notears/blob/master/notears/utils.py
    """
    if check_input:
        if (B_bin_est == -1).any():  # CPDAG
            if not ((B_bin_est == 0) | (B_bin_est == 1) | (B_bin_est == -1)).all():
                raise ValueError("B_bin_est should take value in {0, 1, -1}.")
            if ((B_bin_est == -1) & (B_bin_est.T == -1)).any():
                raise ValueError("Undirected edge should only appear once.")
        else:  # dag
            if not ((B_bin_est == 0) | (B_bin_est == 1)).all():
                raise ValueError("B_bin_est should take value in {0, 1}.")
            if not is_dag(B_bin_est):
                raise ValueError("B_bin_est should be a DAG.")
    d = B_bin_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_bin_est == -1)
    pred = np.flatnonzero(B_bin_est == 1)
    cond = np.flatnonzero(B_bin_true)
    cond_reversed = np.flatnonzero(B_bin_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_bin_est + B_bin_est.T))
    cond_lower = np.flatnonzero(np.tril(B_bin_true + B_bin_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'noramlized shd': shd / (d ** 2 - d), pred_size: pred_size}


def is_markov_equivalent(B1, B2):
    '''
        Judge whether two matrices have the same v-structures and skeletons. a->b<-c
        row -> col
    '''
    def find_v_structures(adj_matrix):
        G = nx.DiGraph(adj_matrix)
        v_structures = []

        for node in G.nodes():
            parents = list(G.predecessors(node))
            for pair in itertools.combinations(parents, 2):
                if not G.has_edge(pair[0], pair[1]) and not G.has_edge(pair[1], pair[0]):
                    v_structures.append((pair[0], node, pair[1]))

        return v_structures
    
    def find_skeleton(B, v_structures):
        inds = []
        for v in v_structures:
            inds.extend([(v[0], v[1]), (v[1], v[0]), (v[2], v[1]), (v[1], v[2])])
        inds = np.array(inds)
        B[inds[:, 0], inds[:, 1]] = 0
        
        return np.logical_or(B, B.T).astype(int)
    
    B1 = bin_mat(B1)
    B2 = bin_mat(B2)
    v_structures1 = find_v_structures(B1)
    v_structures2 = find_v_structures(B2)
    sk1 = find_skeleton(B1, v_structures1)
    sk2 = find_skeleton(B2, v_structures1)

    return set(v_structures1) == set(v_structures2) and sk1 == sk2
    


if __name__ == "__main__":
    # Example usage:
    # Create random data tensors with the same shape (batch, dim)
    batch, dim = 100, 10
    data1 = torch.randn(batch, dim)
    data2 = torch.randn(batch, dim)

    # Calculate the mean correlation coefficient
    mean_corr_coef = mean_correlation_coefficient(data1, data2)
    print(mean_corr_coef)
