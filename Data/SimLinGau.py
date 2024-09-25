import numpy as np
import networkx as nx
import os
import torch

from typing import Callable
from torch.utils.data import TensorDataset
from einops import repeat

from Caulimate.Data import SimCoords, SimDAG
from Caulimate.Utils.Tools import check_tensor
from Caulimate.Utils.GraphUtils import eudistance_mask

class LinGauSuff:
    def __init__(self, 
                 n: int, 
                 dim: int, 
                 degree: float, 
                 t_period: float, 
                 seed: int = 1, 
                 max_eud: int = 100,
                 B_ranges: tuple = ((-1.5, -0.5), (0.5, 1.5)), 
                 extent: list = [-180, 180, -90, 90], 
                 graph_type: str = 'ER', 
                 vary_func: Callable = None, 
                 load_path: str = None,
                 ) -> None:
        """X_t = X_t @ B_t + E_t

        Args:
            n (_type_): _description_
            noise_type (_type_): _description_
            dim (_type_): _description_
            degree (_type_): _description_
            t_period (_type_): _description_
            save_path (_type_): _description_
            seed (_type_): _description_
            max_eud (int, optional): _description_. Defaults to 100.
            B_ranges (tuple, optional): _description_. Defaults to ((-0.2, -0.05), (0.05, 0.2)).
            extent (list, optional): _description_. Defaults to [-180, 180, -90, 90].
            graph_type (str, optional): _description_. Defaults to 'ER'.
            vary_func (_type_, optional): _description_. Defaults to None.
        """
        if load_path is not None:
            data = np.load(load_path)
            self.X = data['X']
            self.Bs = data['Bs']
            self.B_bin = data['B_bin']
            self.coords = data['coords']
        else:
            self.num = n
            self.coords = SimCoords.simulate_uniform_coords(dim, extent, seed)
            self.mask = eudistance_mask(self.coords, max_eud)
            
            self.B_bin = SimDAG.simulate_random_dag(dim, degree, graph_type, seed) * self.mask
            self.Bs, self.B = SimDAG.simulate_time_vary_weight(self.B_bin, n, vary_func, t_period, B_ranges, seed=seed)
            noise = np.random.RandomState(seed).normal(scale=1.0, size=(n, dim))
            self.X = np.empty((n, dim))
            for i in range(n):
                self.X[i] = noise[i] @ np.linalg.inv(np.eye(dim) - self.Bs[i]) 

    def save(self, save_path):
        np.savez(save_path,
                 X = self.X,
                 Bs = self.Bs,
                 B_bin = self.B_bin,
                 coords = self.coords)

class LinGauNoSuff:
    def __init__(self, 
                 n, 
                 obs_dim, 
                 lat_dim, 
                 degree, 
                 t_period, 
                 seed=1, 
                 max_eud = 100,
                 B_ranges=((-0.2, -0.05), (0.05, 0.2)), 
                 extent = [-180, 180, -90, 90], 
                 graph_type='ER', 
                 vary_func=None,
                 load_path=None,
                 ) -> None:
        rs = np.random.RandomState(seed)
        if load_path is not None:
            data = np.load(load_path)
            self.X = data['X']
            self.Bs = data['Bs']
            self.Cs = data['Cs']
            self.B_bin = data['B_bin']
            self.coords = data['coords']
        else:
            self.num = n
            self.coords = SimCoords.simulate_uniform_coords(obs_dim, extent, seed)
            self.mask = eudistance_mask(self.coords, max_eud)
            
            self.B_bin = SimDAG.simulate_random_dag(obs_dim, degree, graph_type, seed) * self.mask
            self.Bs = SimDAG.simulate_time_vary_weight(self.B_bin, n, vary_func, t_period, B_ranges, seed=seed)
            self.C_bin = SimDAG.simulate_tree_adj_mat(lat_dim, obs_dim, seed)
            self.Cs = SimDAG.simulate_time_vary_weight(self.C_bin, n, vary_func, t_period, B_ranges, seed=seed)
            noise = rs.normal(scale=1.0, size=(n, obs_dim))
            self.L = rs.normal(scale=1.0, size=(n, lat_dim))
            self.X = np.empty((n, obs_dim))
            for i in range(n):
                self.X[i] = (self.L[i] @ self.Cs[i] + noise[i]) @ np.linalg.inv(np.eye(obs_dim) - self.Bs[i]) 
                        
        self.init_tensor_dataset()
            
    def init_tensor_dataset(self):
        X_tensor = check_tensor(self.X, dtype=torch.float32)
        X_tensor = X_tensor - X_tensor.mean(dim=0)
        T_tensor = check_tensor(torch.arange(X_tensor.shape[0]), dtype=torch.float32).reshape(-1, 1)
        coords_tensor = repeat(check_tensor(self.coords, dtype=torch.float32), 'j k ->  i j k', i=X_tensor.shape[0])   
        Bs_tensor = check_tensor(self.Bs, dtype=torch.float32)
        Cs_tensor = check_tensor(self.Cs, dtype=torch.float32)
        self.tensor_dataset = TensorDataset(X_tensor, T_tensor, Bs_tensor, Cs_tensor, coords_tensor)

    def save(self, save_path):
        np.savez(save_path,
                 X = self.X,
                 Bs = self.Bs,
                 Cs = self.Cs,
                 B_bin = self.B_bin,
                 coords = self.coords)
        
if __name__ == "__main__":
    # test LinGauSuff
    suff = LinGauSuff(100, 10, 3, 10, vary_func=np.cos, seed=1)
    suff.save('suff.npz')
    # test LinGauNoSuff
    nosuff = LinGauNoSuff(100, 10, 3, 3, 10, vary_func=np.cos, seed=1)
    nosuff.save('nosuff.npz')
    # test load
    suff = LinGauSuff(100, 10, 3, 10, load_path='suff.npz')
    nosuff = LinGauNoSuff(100, 10, 3, 3, 10, load_path='nosuff.npz')