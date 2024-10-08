# For climate data from 
# Prof. Dr. Jakob Runge (he/him)
# climateinformaticslab.com
 
# Chair of Climate Informatics
# Technische Universität Berlin | Electrical Engineering and Computer Science | Institute of Computer Engineering and Microelectronics
 
# Group Lead Causal Inference
# German Aerospace Center | Institute of Data Science

import pickle
import xarray as xr
import numpy as np
import glob
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import os

from Caulimate.Utils.Tools import check_tensor, check_array, center_and_norm

DATASET_DIR = os.path.join(os.getenv('DATASET_DIR'), 'CESM2')
SST_DATA_PATH = os.path.join(DATASET_DIR, 'CESM2_pacific_SST.pkl') 
SPACE_INDEX_DATA_PATH = os.path.join(DATASET_DIR, 'CESM2_pacific_space_index.pkl')
GROUP_DATA_DIR = os.path.join(DATASET_DIR, 'group_region')
DOWNSCALE_PATH = os.path.join(DATASET_DIR, 'downscaled_pacific_CESM2.txt')
DOWNSCALE_METADATA_PATH = os.path.join(DATASET_DIR, 'downscaled_metadata.pkl')

def load_data(data_path, num, d_X, distance):
    with open(data_path, 'rb') as file:
        data = pickle.load(file)
    
    data = data[:, :d_X]
    # reset args.num
    num = data.shape[0]
    
    m_true = generate_band_bin_matrix(num, d_X, distance)
    
    return data, m_true

def generate_band_bin_matrix(n, d, bandwidth):
    matrix = np.zeros((d, d))
    for i in range(d):
        for j in range(max(0, i - bandwidth), min(d, i + bandwidth + 1)):
            if i != j:
                matrix[i, j] = 1
    return np.tile(matrix, (n, 1, 1))


class CESM2_dataset(Dataset):
    def __init__(self, space_index_data_path, sst_data_path, n_slices=None):
        f = open(sst_data_path, 'rb')
        SST = pickle.load(f)
        f.close()
        # metadata file
        f = open(space_index_data_path, 'rb')
        coords = pickle.load(f).drop_dims('time')
        f.close()
        
        sst_space_index = coords['space_index'].values[coords.nnaSST]
        print('--- Number of regions in SST: {}'.format(sst_space_index.shape[0]))
        sst_space_indices = np.repeat(sst_space_index[np.newaxis, :], 6000, axis=0)
        self.xt = SST
        self.st = sst_space_indices

    def __len__(self):
        return self.xt.shape[0]

    def __getitem__(self, idx):
        return {'xt': self.xt[idx], 'st': self.st[idx]}
    
class CESM2_entire_grouped_dataset(Dataset):
    """
        Dataset for entire CESM2 pacific SST data without parallelization
    """
    def __init__(self, path, ts_len, n_domains):
        super().__init__()
        self.n_domains = 12
        self.ts_len = ts_len
        self.data = {}

        self.xr_ds = self.load_nc_data(path)
        
        xr_ds_expanded = self.xr_ds.stack(spatial=('nlat', 'nlon')).reset_coords(['lat', 'lon'], drop=False)
        grouped_means = xr_ds_expanded.groupby('group_index').mean(dim='spatial', skipna=True)

        SST_means = grouped_means['SST'].values.T
        lat_means = grouped_means['lat'].values
        lon_means = grouped_means['lon'].values
        group_index = grouped_means['group_index'].values
        
        valid_indices = ~np.isnan(SST_means[0])
        SST_means = SST_means[:, valid_indices]
        lat_means = lat_means[valid_indices]
        lon_means = lon_means[valid_indices]
        group_index = group_index[valid_indices]
        
        self.coords = np.stack([lat_means, lon_means], axis=1)
        
        self.n_samples = SST_means.shape[0]
        self.n_ts = self.n_samples - ts_len + 1 
        domain_seq = torch.arange(n_domains, dtype=torch.float32) # 12 months
        # convert SST to time-series data
        SST_processed = center_and_norm(SST_means)
        self.data['xt'] = check_tensor([SST_processed[i:i+ts_len] for i in range(SST_processed.shape[0] - ts_len + 1)], dtype=torch.float32, device=torch.device('cpu')) 
        self.data['st'] = check_tensor(self.coords, device=torch.device('cpu'))
        self.data['ct'] = domain_seq.repeat(self.n_ts // self.n_domains + 1)[:self.n_ts, np.newaxis]
        self.data['ht'] = check_tensor(np.arange(self.n_ts), dtype=torch.float32, device=torch.device('cpu'))
        
        self.group_index = group_index
        self.d_X = self.data['xt'].shape[-1]
        
        print("--- Entire CESM2 pacific SST, observation dim is {} after map-reduce".format(self.d_X))

    def load_nc_data(self, path):
        return xr.open_dataarray(path)
    
    def __len__(self):
        return self.n_ts
    
    def __getitem__(self, idx):
        return {'xt': self.data['xt'][idx], 'ct': self.data['ct'][idx], 'ht': self.data['ht'][idx], 'st': self.data['st']}
    
class CESM2_grouped_dataset(Dataset):
    def __init__(self, path, num_area, ts_len=3, n_domains=12):
        super().__init__()
        self.xr_da = self.load_nc_data(path)
        self.n_areas = int(self.xr_da.area_index.max().item() + 1)
        self.n_groups = int(self.xr_da.group_index.max().item() + 1)
        self.times, self.n_lat, self.n_lon = self.xr_da.shape
        self.group_datasets = [area_dataset(self.xr_da, i, ts_len, n_domains) for i in range(num_area)]
        
        print(f"--- Number of areas: {self.n_areas}, Number of groups: {self.n_groups}")
        
    def load_nc_data(self, path):
        return xr.open_dataarray(path)
    
    def __len__(self):
        return len(self.group_datasets)
    
    def __getitem__(self, idx):
        return self.group_datasets[idx]
    
class area_dataset(Dataset):
    def __init__(self, xr_da, area_idx, ts_len, n_domains):
        super().__init__()
        self.area_idx = area_idx
        self.n_domains = 12
        self.ts_len = ts_len
        self.data = {}

        self.area_xr_da = xr_da.copy().where(xr_da.area_index == area_idx, drop=True)
        
        area_ds = self.area_xr_da.stack(spatial=('nlat', 'nlon'))
        area_ds_expanded = area_ds.reset_coords(['lat', 'lon'], drop=False)
        grouped_means = area_ds_expanded.groupby('group_index').mean(dim='spatial', skipna=True)
        self.full_xr_ds = area_ds_expanded
        self.xr_ds = grouped_means

        SST_means = grouped_means['SST'].values.T
        lat_means = grouped_means['lat'].values
        lon_means = grouped_means['lon'].values
        group_index = grouped_means['group_index'].values
        
        valid_indices = ~np.isnan(SST_means[0])

        SST_means = SST_means[:, valid_indices]
        lat_means = lat_means[valid_indices]
        lon_means = lon_means[valid_indices]
        group_index = group_index[valid_indices]
        
        self.coords = np.stack([lat_means, lon_means], axis=1)
        
        self.n_samples = SST_means.shape[0]
        self.n_ts = self.n_samples - ts_len + 1 
        domain_seq = torch.arange(n_domains, dtype=torch.float32) # 12 months
        # convert SST to time-series data
        SST_processed = center_and_norm(SST_means)
        self.X = check_array(SST_means, dtype=np.float32)
        self.T = check_array(np.arange(self.n_samples), dtype=np.float32).reshape(-1, 1)
        
        self.data['xt'] = check_tensor([SST_processed[i:i+ts_len] for i in range(SST_processed.shape[0] - ts_len + 1)], dtype=torch.float32)
        self.data['st'] = check_tensor(self.coords)
        self.data['ct'] = domain_seq.repeat(self.n_ts // self.n_domains + 1)[:self.n_ts, np.newaxis]
        self.data['ht'] = check_tensor(np.arange(self.n_ts), dtype=torch.float32)
        
        self.group_inds = group_index
        self.d_X = self.data['xt'].shape[-1]
        
        print("--- Area {} has {} groups".format(area_idx, self.d_X))
    
    def __len__(self):
        return self.n_ts
    
    def __getitem__(self, idx):
        return {'xt': self.data['xt'][idx], 'ct': self.data['ct'][idx], 'ht': self.data['ht'][idx], 'st': self.data['st']}

class downscale_dataset(Dataset):
    def __init__(self, path, metadata_path, ts_len=3, n_domains=12):
        self.ts_len = ts_len
        self.n_domains = n_domains
        
        X = check_array(np.loadtxt(path), dtype=np.float32) - 273.15 # celsius to Kelvin
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
    
        self.X = (X - self.means) / self.stds
        
        f = open(metadata_path, 'rb')
        meta_data = pickle.load(f)
        self.coords = check_array(np.column_stack((meta_data.lat[meta_data.nnaSST].values, meta_data.lon[meta_data.nnaSST].values)), dtype=np.float32)
        self.d_X = self.X.shape[-1]
        self.n_samples = self.X.shape[0]
        self.n_ts = self.n_samples - self.ts_len + 1 
        domain_seq = torch.arange(n_domains, dtype=torch.float32) # 12 months
        self.T = check_array(np.arange(self.n_samples), dtype=np.float32).reshape(-1, 1)
    
        self.data = {}
        self.data['xt'] = check_tensor([self.X[i:i+ts_len] for i in range(self.n_ts)], dtype=torch.float32, device='cpu')
        self.data['st'] = check_tensor(self.coords, device='cpu')
        self.data['ct'] = domain_seq.repeat(self.n_ts // self.n_domains + 1)[:self.n_ts, np.newaxis]
        self.data['ht'] = check_tensor(np.arange(self.n_ts), dtype=torch.float32, device='cpu')
    
    def __len__(self):
        return self.n_ts
    
    def __getitem__(self, idx):
        return {'xt': self.data['xt'][idx], 'ct': self.data['ct'][idx], 'ht': self.data['ht'][idx], 'st': self.data['st']}
    


if __name__ == "__main__":
    pass
    
    