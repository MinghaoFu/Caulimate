import numpy as np
import pandas as pd
import os
import warnings
import torch
import itertools
import pytest
import shutil
import matplotlib.pyplot as plt
import yaml

from torchsummary import summary
from sklearn.linear_model import LinearRegression, Lasso
from torch.utils.data import Dataset
from sklearn.decomposition import PCA, FastICA, fastica
from sklearn.decomposition._fastica import _gs_decorrelation
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import assert_allclose
from scipy import linalg, stats
from tqdm import tqdm
from time import time
from lingam.utils import make_dot

def load_yaml(filename, type='class'):
    """
    Load and print YAML config files
    """
    with open(filename, 'r') as stream:
        file = yaml.safe_load(stream)
        if type == 'class':
            return dict_to_class(**file)
        elif type == 'dict':
            return file

def makedir(path, remove_exist=False):
    if remove_exist and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def check_tensor(data, dtype=None, device=None, astype=None):
    if not torch.is_tensor(data):
        if isinstance(data, np.ndarray):
            data = torch.tensor(data)
        elif isinstance(data, (list, tuple)):
            data = torch.tensor(np.array(data))
        else:
            raise ValueError("Unsupported data type. Please provide a list, NumPy array, or PyTorch tensor.")
    if astype is not None:
        return data.type_as(astype)
    
    if dtype is None:
        dtype = data.dtype
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return data.to(device, dtype=dtype)

def check_array(data):
    """
    Convert any input data to a NumPy array.

    Args:
        data: Input data of any type, including tensors.

    Returns:
        numpy.ndarray: NumPy array representation of the input data.
    """
    if isinstance(data, np.ndarray):
        return data

    if isinstance(data, torch.Tensor):
        return data.cpu().detach().numpy()

    if isinstance(data, list) or isinstance(data, tuple):
        return np.array(data)

    if isinstance(data, int) or isinstance(data, float):
        return np.array([data])

    if isinstance(data, dict):
        return np.array(list(data.values()))

    try:
        return np.array(data)
    except Exception as e:
        raise ValueError("Unable to convert the input data to a NumPy array.") from e

def dict_to_class(**dict):
    class _dict_to_class:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    return _dict_to_class(**dict)

def linear_regression_initialize(x, distance, ) -> np.array:
    x = check_array(x)
    model = LinearRegression()
    n, d = x.shape
    B_init = np.zeros((d, d))
    for i in range(d):
        start = max(i - distance, 0)
        end = min(i + distance + 1, d)

        model.fit(np.concatenate((x[:, start : i], x[:, i + 1 : end]), axis=1), x[:, i])
        B_init[i][start : i] = model.coef_[ : min(i, distance)]#np.pad(np.insert(model.coef_, min(i, distance), 0.), (start, d - end), 'constant')
        B_init[i][i + 1 : end] = model.coef_[min(i, distance) : ]
    
    return B_init

def model_info(model, input_shape):
    summary(model, input_shape)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"--- Total Parameters: {total_params}, Trainable Parameters: {trainable_params}")
    
    if torch.cuda.is_available():
        input_data = check_tensor(torch.randn([1] + list(input_shape)))
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            _ = model(input_data)
        end.record()
        
        torch.cuda.synchronize()
        # Calculate the elapsed time
        elapsed_time = start.elapsed_time(end)
        print(f"--- Elapsed time: {elapsed_time} ms")
    else:
        print("--- CUDA is not available. Please run this on a CUDA-enabled environment.")
    
    return total_params, trainable_params, elapsed_time
    
def save_DAG(n_plots, save_dir, epoch, Bs_pred, Bs_gt=None, graph_thres=0.1, add_value=True, ):
    print(f'--- Save DAG predictions...')
    save_epoch_dir = os.path.join(save_dir, f'epoch_{epoch}')
    makedir(save_epoch_dir)
    
    Bs_pred = check_array(Bs_pred) 
    Bs_pred[:, np.mean(np.abs(Bs_pred), axis=0) < graph_thres] = 0
    if Bs_gt is None:
        Bs_gt = Bs_pred
    else:
        Bs_gt = check_array(Bs_gt)
        
    row_indices, col_indices = np.nonzero(Bs_gt[0])
    edge_values = Bs_gt[:, row_indices, col_indices]
    values = []
    values_true =[]
    for _ in range(len(edge_values)):
        values.append([])
        values_true.append([])
    for k in range(n_plots):
        for idx, (i, j) in enumerate(zip(row_indices, col_indices)):
            values[idx].append(Bs_pred[k][i, j])
            values_true[idx].append(Bs_gt[k][i, j])
    
    for idx, (i, j) in enumerate(zip(row_indices, col_indices)):
        plt.plot(values[idx], label='Pred' + str(idx))
        plt.plot(values_true[idx], label = 'True' + str(idx))
        plt.legend() 
        plt.savefig(os.path.join(save_epoch_dir, f'({i}, {j})_trend.png'), format='png')
        plt.show()
        plt.clf()
        
    time_idx = np.random.randint(0, n_plots) # plot one time index
    plot_solutions([Bs_gt[time_idx], Bs_pred[time_idx]], ['B_gt', 'B_est'], os.path.join(save_epoch_dir, f'DAG_{time_idx}.png'), add_value=add_value)
    np.save(os.path.join(save_epoch_dir, 'prediction.npy'), np.round(Bs_pred, 4))
    np.save(os.path.join(save_epoch_dir, 'ground_truth.npy'), np.round(Bs_gt, 4))
    
def make_dots(arr: np.array, labels, save_path, name):
    if len(arr.shape) > 2:
        for i in arr.shape[0]:
            dot = make_dot(arr[i])
            dot.format = 'png'
            dot.render(os.path.join(save_path, f'{name}_{i}'))
    elif len(arr.shape) == 2:
        dot = make_dot(arr, labels=labels)
        dot.format = 'png'
        dot.render(os.path.join(save_path, name))
        os.remove(os.path.join(save_path, name)) # remove digraph
        
# def plot_solution(B_true, B_est, M_true, M_est, save_name=None, add_value=False, logger=None):
#     """Checkpointing after the training ends.

#     Args:
#         B_true (numpy.ndarray): [d, d] weighted matrix of ground truth.
#         B_est (numpy.ndarray): [d, d] estimated weighted matrix.
#         B_processed (numpy.ndarray): [d, d] post-processed weighted matrix.
#         save_name (str or None): Filename to solve the plot. Set to None
#             to disable. Default: None.
#     """
#     # Define a function to add values to the plot
#     def add_values_to_plot(ax, matrix):
#         for (i, j), val in np.ndenumerate(matrix):
#             if np.abs(val) > 0.1:
#                 ax.text(j, i, f'{val:.1f}', ha='center', va='center', color='black')
            
#     fig, axes = plt.subplots(figsize=(10, 4), ncols=4)

#     # Plot ground truth
#     im = axes[0].imshow(B_true, cmap='RdBu', interpolation='none', vmin=-2.25, vmax=2.25)
#     axes[0].set_title("B_gt", fontsize=13)
#     axes[0].tick_params(labelsize=13)
#     if add_value:
#         add_values_to_plot(axes[0], B_true)

#     # Plot estimated solution
#     im = axes[1].imshow(B_est, cmap='RdBu', interpolation='none', vmin=-2.25, vmax=2.25)
#     axes[1].set_title("B_est", fontsize=13)
#     axes[1].set_yticklabels([])    # Remove yticks
#     axes[1].tick_params(labelsize=13)
#     if add_value:
#         add_values_to_plot(axes[1], B_est)

#     # Plot post-processed solution
#     im = axes[2].imshow(M_true, cmap='RdBu', interpolation='none', vmin=-2.25, vmax=2.25)
#     axes[2].set_title("M_gt", fontsize=13)
#     axes[2].set_yticklabels([])    # Remove yticks
#     axes[2].tick_params(labelsize=13)
#     if add_value:
#         add_values_to_plot(axes[2], M_true)

#     im = axes[3].imshow(M_est, cmap='RdBu', interpolation='none', vmin=-2.25, vmax=2.25)
#     axes[3].set_title("M_est", fontsize=13)
#     axes[3].set_yticklabels([])    # Remove yticks
#     axes[3].tick_params(labelsize=13)
#     if add_value:
#         add_values_to_plot(axes[3], M_est)
        
#     # Adjust space between subplots and add colorbar
#     fig.subplots_adjust(wspace=0.1)
#     im_ratio = 4 / 10
#     cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.05*im_ratio, pad=0.035)
#     cbar.ax.tick_params(labelsize=13)

#     # Save or display the figure
#     if save_name is not None:
#         fig.savefig(save_name, bbox_inches='tight')
#     else:
#         plt.show()

#     # Return the figure
#     return fig

def plot_solutions(mats, names, save_name=None, add_value=False, logger=None):
    """Checkpointing after the training ends.

    Args:
        B_true (numpy.ndarray): [d, d] weighted matrix of ground truth.
        B_est (numpy.ndarray): [d, d] estimated weighted matrix.
        B_processed (numpy.ndarray): [d, d] post-processed weighted matrix.
        save_name (str or None): Filename to solve the plot. Set to None
            to disable. Default: None.
    """
    # Define a function to add values to the plot
    def add_values_to_plot(ax, matrix):
        for (i, j), val in np.ndenumerate(matrix):
            if np.abs(val) > 0.1:
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', color='black')
    
    n_figs = len(mats)
    fig, axes = plt.subplots(figsize=(10, n_figs), ncols=n_figs)

    for i in range(n_figs):
        im = axes[i].imshow(mats[i], cmap='RdBu', interpolation='none', vmin=-2.25, vmax=2.25)
        axes[i].set_title(names[i], fontsize=13)
        if i != 0:
            axes[i].set_yticklabels([])    # Remove yticks
        axes[i].tick_params(labelsize=13)
        if add_value:
            add_values_to_plot(axes[i], mats[i])
        
    # Adjust space between subplots and add colorbar
    fig.subplots_adjust(wspace=0.1)
    im_ratio = len(mats) / 10
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.05*im_ratio, pad=0.035)
    cbar.ax.tick_params(labelsize=13)

    # Save or display the figure
    if save_name is not None:
        fig.savefig(save_name, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    # Return the figure
    return fig