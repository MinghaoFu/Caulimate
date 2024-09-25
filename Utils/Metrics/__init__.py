import torch
import torch.nn as nn
import numpy as np

from Caulimate.Utils.Tools import check_array


def MAPE(y_true, y_pred):
    y_true, y_pred = check_array(y_true), check_array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def MAE(y_true, y_pred):
    y_true, y_pred = check_array(y_true), check_array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def MSE(y_true, y_pred):
    y_true, y_pred = check_array(y_true), check_array(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def RMSE(y_true, y_pred):
    return np.sqrt(MSE(y_true, y_pred))

    
def R_squared(y_true, y_pred):
    """
    Calculate the R-squared (coefficient of determination) for a set of actual and predicted values.

    Parameters:
    y_true (array-like): Array of actual values.
    y_pred (array-like): Array of predicted values.

    Returns:
    float: R-squared value.
    """
    y_true, y_pred = check_array(y_true), check_array(y_pred)
    
    # Calculate the mean of actual values
    y_mean = np.mean(y_true)
    
    # Total sum of squares (SST)
    sst = np.sum((y_true - y_mean) ** 2)
    
    # Residual sum of squares (SSR)
    ssr = np.sum((y_true - y_pred) ** 2)
    
    # R-squared
    r_squared = 1 - (ssr / sst)
    
    return r_squared