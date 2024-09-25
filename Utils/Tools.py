import numpy as np
import pandas as pd
import os
import torch
import shutil
import matplotlib.pyplot as plt
import yaml
import xarray as xr
import random
import torch.nn as nn
import networkx as nx

from torchsummary import summary
from sklearn.linear_model import LinearRegression, Lasso, SGDRegressor
from torch.utils.data import Dataset
from sklearn.decomposition import PCA, FastICA, fastica
from sklearn.decomposition._fastica import _gs_decorrelation
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import assert_allclose
from scipy import linalg, stats
from tqdm import tqdm
from time import time
import os
import subprocess

def get_free_gpu():
    # Run nvidia-smi command and get the output
    nvidia_smi_output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits']
    ).decode('utf-8')

    # Parse the output
    gpu_memory_usage = []
    for line in nvidia_smi_output.strip().split('\n'):
        index, memory_used = line.split(', ')
        gpu_memory_usage.append((int(index), int(memory_used)))

    # Find the GPU with the least memory used
    best_gpu = min(gpu_memory_usage, key=lambda x: x[1])[0]
    return str(best_gpu)

def scp_transfer_file(path, user='fuminghao', remote_server='fus-MacBook-Pro.local', destination="/Users/fuminghao/Desktop/CauLimate", remove=True):
    """
    Transfer a file to a remote server using SCP.

    Args:
        path (str): Path to the file to be transferred.
        user (str): Username for the remote server.
        remote_server (str): IP address or hostname of the remote server.
        destination (str): Path to the destination on the remote server. Default is "/home".

    Returns:
        None
    """
    # Add your code here to transfer the file using SCP
    os.system(f'scp {path} {user}@{remote_server}:{destination}')
    if remove:
        os.remove(path)

def bin_mat(B: np.array):
    '''
        Binarize a (batched) numpy matrix
    '''
    return np.where(check_array(B) != 0, 1, 0)

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def save_log(log_file_path: str, log: str):
    print(log)
    with open(log_file_path, 'a') as file:  
        file.write(log)

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

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

def check_tensor(data, dtype=None, astype=None, device=None):
    if not torch.is_tensor(data):
        if isinstance(data, np.ndarray):
            data = torch.tensor(data)
        elif isinstance(data, (list, tuple)):
            data = torch.tensor(np.array(data))
        elif isinstance(data, xr.DataArray):
            data = torch.from_numpy(data.fillna(0).to_numpy())
        else:
            raise ValueError("Unsupported data type. Please provide a list, NumPy array, or PyTorch tensor.")
    
    if astype is not None:
        return data.type_as(astype)
    
    if dtype is None:
        dtype = data.dtype
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return data.to(device, dtype=dtype)

def check_array(data, dtype=None):
    """
    Convert any input data to a NumPy array.

    Args:
        data: Input data of any type, including tensors.

    Returns:
        numpy.ndarray: NumPy array representation (copy) of the input data.
    """
    if isinstance(data, np.ndarray):
        data = data.copy()

    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()

    if isinstance(data, list) or isinstance(data, tuple):
        data = np.array(data)

    if isinstance(data, int) or isinstance(data, float):
        data = np.array([data])

    if isinstance(data, dict):
        data = np.array(list(data.values()))
    
    if dtype is not None:
        data = data.astype(dtype)

    return data


def center_and_norm(x, axis=0):
    """Centers and norms x **in place**

    Parameters
    -----------
    x: ndarray
        Array with an axis of observations (statistical units) measured on
        random variables.
    axis: int, optional
        Axis along which the mean and variance are calculated.
    """
    x = check_array(x)
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
    
    return (x - means) / stds

def whiten_data(X):
    """_summary_

    Args:
        X (_type_): (n_samples, dim)

    Returns:
        _type_: _description_
    """
    X = check_array(X)
    # Remove the mean
    X_mean = X.mean(axis=0)
    X_demeaned = X - X_mean
    
    # Compute the covariance of the mean-removed data
    covariance_matrix = np.cov(X_demeaned, rowvar=False)
    
    # Eigenvalue decomposition of the covariance matrix
    eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
    
    # Compute the whitening matrix
    whitening_matrix = np.dot(eigen_vectors, np.diag(1.0 / np.sqrt(eigen_values)))
    
    # Transform the data using the whitening matrix
    X_whitened = np.dot(X_demeaned, whitening_matrix)
    
    return X_whitened, X_mean, np.mean(X_whitened, axis=0), np.var(X_whitened, axis=0)

def dict_to_class(**dict):
    class _dict_to_class:
        def __init__(self, **entries):
            self.__dict__.update(entries)
            self.entries = entries
        
        def __str__(self):
            return str(self.entries)

    return _dict_to_class(**dict)

def lin_reg_discover(X1, X2, mask=None, n_iter=2000, tol=1e-2):
    '''
    X1 = X2B
    '''
    print("--- Linear regression initialization")
    X1 = check_tensor(X1, dtype=torch.float32)
    X2 = check_tensor(X2, dtype=torch.float32)
    dim1 = X1.shape[-1]
    dim2 = X2.shape[-1]
    mask = check_tensor(mask, dtype=torch.int64) if mask is not None else check_tensor(torch.ones(dim2, dim1))
    B = check_tensor(torch.zeros((dim2, dim1)))
    for i in tqdm(range(dim1)):
        act_inds = torch.where(mask[:, i] == 1)[0]
        model = nn.Linear(mask[:, i].sum(), 1, bias=False).to(X1.device)
        x_features= X2[:, act_inds]
        y_target = X1[:, i:i + 1]
    
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        
        # Train the model (Here we should have a loop for epochs, but for simplicity, we just do it once)
        optimizer.zero_grad()
        outputs = model(x_features)
        loss = criterion(outputs, y_target)
        loss.backward()
        optimizer.step()
        for _ in range(n_iter):
            optimizer.zero_grad()  # Clear gradients for the next train
            outputs = model(x_features)  # Forward pass: Compute predicted y by passing x to the model
            loss = criterion(outputs, y_target)  # Compute loss
            loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
            optimizer.step()  # Update model parameters
            if loss < tol:
                break

        # Update the B_init matrix with the learned coefficients
        with torch.no_grad():
            B[act_inds, i] = model.weight
    
    return B

def lin_reg_init(x, mask=None, n_iter=2000, thres=0.1, tol=1e-2, lr=1e-3) -> torch.Tensor:
    '''
    X = XB
    '''
    print("--- Linear regression initialization")
    x = check_tensor(x, dtype=torch.float32)
    n, dim = x.shape
    mask = check_tensor(mask, dtype=torch.int64) if mask is not None else check_tensor(torch.ones(dim, dim) - torch.eye(dim))
    B_init = check_tensor(torch.zeros((dim, dim)))
    for i in tqdm(range(dim)):
        l_act_inds = torch.where(mask[:i, i] == 1)[0]
        r_act_inds = torch.where(mask[i+1:, i] == 1)[0] + i + 1
        model = nn.Linear(int(mask[:, i].sum()), 1, bias=False).to(x.device)
        x_features = torch.cat((x[:,l_act_inds], x[:,r_act_inds]), dim=1)
        y_target = x[:, i:i + 1]
    
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        
        # Train the model (Here we should have a loop for epochs, but for simplicity, we just do it once)
        optimizer.zero_grad()
        outputs = model(x_features)
        loss = criterion(outputs, y_target)
        loss.backward()
        optimizer.step()
        for _ in range(n_iter):
            optimizer.zero_grad()  # Clear gradients for the next train
            outputs = model(x_features)  # Forward pass: Compute predicted y by passing x to the model
            loss = criterion(outputs, y_target)  # Compute loss
            loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
            optimizer.step()  # Update model parameters
            if loss < tol:
                break

        with torch.no_grad():
            B_init[l_act_inds, i] = model.weight[:, :len(l_act_inds)]
            B_init[r_act_inds, i] = model.weight[:, len(l_act_inds):]

    B_init_arr = check_array(B_init)
    B_init_arr[np.abs(B_init_arr) < thres] = 0
    reg_mask = bin_mat(B_init_arr) 

    print(f"--- Linear regression init loss: {loss}")
    
    return B_init_arr, reg_mask

def large_scale_linear_regression_initialize(x, max_iter=1000, mask=None) -> np.array:
    '''
        X = BX
    '''
    print("--- Linear regression initialization")
    x = check_array(x)
    model = SGDRegressor(penalty='l1', alpha=0.1, max_iter=max_iter, tol=1e-3)
    n, dim = x.shape
    B_init = np.zeros((dim, dim))
    for i in tqdm(range(dim)):
        if mask is not None:
            l_act_inds = np.where(mask[:, :i] == 1)[0]
            r_act_inds = np.where(mask[:, i:] == 1)[0] + i

            model.fit(np.concatenate((x[:, l_act_inds], x[:, r_act_inds]), axis=1), x[:, i])
            B_init[:l_act_inds, i] = model.coef_[:len(l_act_inds)]
            B_init[i+1:, i] = model.coef_[len(l_act_inds):]
        else:
            model.fit(np.concatenate((x[:, i], x[:, i+1:]), axis=1), x[:, i])
            B_init[:i, i] = model.coef_[:i]
            B_init[i+1:, i] = model.coef_[i:]
        # B_init[i][:i] = model.coef_[:i]#np.pad(np.insert(model.coef_, min(i, distance), 0.), (start, d - end), 'constant')
        # B_init[i][i + 1:] = model.coef_[i: ]

    return B_init

def linear_regression_initialize(x, distance=None, ) -> np.array:
    x = check_array(x)
    model = LinearRegression()
    n, d = x.shape
    B_init = np.zeros((d, d))
    if distance == None:
        distance = d - 1
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
    

def coord_to_index(coord, shape):
    """
    Convert coordinate to index.

    Args:
        coord (tuple): Coordinate.
        shape (tuple): Shape of the array.

    Returns:
        int: Index.
    """
    return np.ravel_multi_index(coord, shape)

def index_to_coord(index, shape):
    """
    Convert index to coordinate.

    Args:
        index (int): Index.
        shape (tuple): Shape of the array.

    Returns:
        tuple: Coordinate.
    """
    return np.unravel_index(index, shape)

def recover_coordinates(ds):
    ds['lon'] = (ds['lon'] + 180) % 360 - 180
    ds = ds.sortby('lon')
    return ds
