import numpy as np

def covariance(x):
    mean = np.mean(x, axis=-1, keepdims=True)
    n = np.shape(x)[1] - 1
    m = x - mean

    return (m.dot(m.T))/n

def center_and_norm(x, axis=-1):
    """Centers and norms x **in place**

    Parameters
    -----------
    x: ndarray
        Array with an axis of observations (statistical units) measured on
        random variables.
    axis: int, optional
        Axis along which the mean and variance are calculated.
    """
    x = np.rollaxis(x, axis)
    x -= x.mean(axis=0)
    x /= x.std(axis=0)
    
    return x