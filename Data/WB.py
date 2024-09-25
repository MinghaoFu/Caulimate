# For climate data from 
# Prof. Dr. Jakob Runge (he/him)
# climateinformaticslab.com
 
# Chair of Climate Informatics
# Technische Universität Berlin | Electrical Engineering and Computer Science | Institute of Computer Engineering and Microelectronics
 
# Group Lead Causal Inference
# German Aerospace Center | Institute of Data Science

import pickle
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import glob
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from Caulimate.Utils.Tools import check_tensor, check_array, center_and_norm
from Caulimate.Utils.GraphUtils import eudistance_mask

SST_DATA_PATH = '/l/users/minghao.fu/dataset/WeatherBench_data_full/temperature_850/*.nc'
V_PATH = '/l/users/minghao.fu/dataset/WeatherBench_data_full/v_component_of_wind/*.nc'
U_PATH = '/l/users/minghao.fu/dataset/WeatherBench_data_full/u_component_of_wind/*.nc'
SPACE_INDEX_DATA_PATH = "/l/users/minghao.fu/dataset/CESM2/CESM2_pacific.pkl"
GROUP_DATA_DIR = "/l/users/minghao.fu/dataset/CESM2/group_region/"

def create_adjacency_matrix(u_slice: np.array, v_slice: np.array, lon: np.array, lat: np.array):
    # TODO: adapt for non-gridded coordinates
    n_lat, n_lon = u_slice.shape
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Flatten the arrays for easier handling
    lon_flat = lon_grid.flatten()
    lat_flat = lat_grid.flatten()
    u_flat = u_slice.flatten()
    v_flat = v_slice.flatten()
    total_points = len(lon_flat)

    # Initialize the adjacency matrix
    adj_matrix = np.zeros((total_points, total_points))

    # Compute endpoints based on the u and v vectors
    endpoint_lons = lon_flat + u_flat
    endpoint_lats = lat_flat + v_flat

    # Determine the nearest grid point for each endpoint
    for i in range(total_points):
        #lon_dis = [min(a, b) for a, b in zip(np.abs(endpoint_lons[i] - lon_flat), 360 - np.abs(endpoint_lons[i] - lon_flat))]
        #lat_dis = [min(a, b) for a, b in zip(np.abs(endpoint_lats[i] - lat_flat), 180 - np.abs(endpoint_lats[i] - lat_flat))]
        distances = np.sqrt((endpoint_lons[i] - lon_flat)**2 + (endpoint_lats[i] - lat_flat)**2)
        distances[i] = np.inf
        #distances = np.sqrt([a + b for a, b in zip([pow(x, 2) for x in lon_dis], [pow(x, 2) for x in lat_dis])])
        nearest_point_index = np.argmin(distances)
        if distances[nearest_point_index] < np.inf:  # Optionally set a maximum distance threshold
            adj_matrix[i, nearest_point_index] = np.sqrt(u_flat[i]**2 + v_flat[i]**2)

    return adj_matrix

def reconstruct_uv_slices(adj_matrix, lat, lon):
    n_lat, n_lon = len(lat), len(lon)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Flatten the grids for easier indexing
    lon_flat = lon_grid.flatten()
    lat_flat = lat_grid.flatten()

    # Initialize u_slice and v_slice
    u_slice = np.zeros_like(lon_grid)
    v_slice = np.zeros_like(lat_grid)

    # Iterate through each grid point in the adjacency matrix
    for i in range(n_lat * n_lon):
        # Get the index of the maximum connection
        j = np.argmax(adj_matrix[i, :])
        if adj_matrix[i, j] > 0:  # Check if there is a connection
            # Calculate the vector components
            u_slice.flat[i] = lon_flat[j] - lon_flat[i]
            v_slice.flat[i] = lat_flat[j] - lat_flat[i]

    return u_slice, v_slice

class WeatherBench_dataset(Dataset):
    def __init__(self, path, u_path, v_path, save_path, lat_range=slice(45, 75), lon_range=slice(45, 75), level=850, ts_len=3, n_domains=12, resample_size='1D', max_eud=10):
        super().__init__()

        xr_space_ds = self.load_nc_data(path) #.coarsen(lat=2, lon=2, boundary='trim').mean()
        self.xr_ds = xr_space_ds.sel(lon=lon_range, lat=lat_range).stack(space=('lat', 'lon')).resample(time=resample_size).mean().squeeze() #.sel(lat=slice(0, 45), lon=slice(0, 90))
        self.coords = check_array([list(coords_tuple) for coords_tuple in self.xr_ds.coords['space'].values])
        self.save_path = save_path
        self.d_X = len(self.coords)
        self.n_samples = self.xr_ds.shape[0]
        self.n_ts = self.n_samples - ts_len + 1
        self.n_domains = n_domains
        self.mask = eudistance_mask(self.coords, max_eud)

        self.data = {}
        centered_data = center_and_norm(self.xr_ds.copy().values)
        self.data['x'] = centered_data
        self.data['xt'] = check_tensor([centered_data[i:i+ts_len] for i in range(centered_data.shape[0] - ts_len + 1)], dtype=torch.float32)
        self.data['st'] = check_tensor(self.coords)
        domain_seq = torch.arange(n_domains, dtype=torch.float32) # 12 months
        self.data['ct'] = check_tensor(domain_seq.repeat(self.n_ts // self.n_domains + 1)[:self.n_ts, np.newaxis])
        self.data['ht'] = check_tensor(np.arange(self.n_ts), dtype=torch.float32)

        self.wind_adj_mat = self.adj_mat_init_by_wind(u_path, v_path, lat_range, lon_range, level, resample_size) 
        # self.times, self.n_lat, self.n_lon = self.xr_ds.shape
        print(f"--- Number of areas: {len(self.coords)}")
        
    def load_nc_data(self, path):
        return xr.open_mfdataset(path, combine='by_coords').to_array() # mfdataset to xarray
    
    def adj_mat_init_by_wind(self, u_path, v_path, lat_range, lon_range, level, resample_size):
        u = xr.open_mfdataset(u_path, combine='by_coords').resample(time=resample_size).mean()
        v = xr.open_mfdataset(v_path, combine='by_coords').resample(time=resample_size).mean()
        #for time_idx in tqdm(range(self.n_samples)):
        time_idx = 100
        v_slice = v.isel(time=time_idx).sel(lat=lat_range, lon=lon_range, level=level)['v']
        u_slice = u.isel(time=time_idx).sel(lat=lat_range, lon=lon_range, level=level)['u']
        lon, lat = u_slice.coords['lon'], u_slice.coords['lat']

        plt.figure(figsize=(14, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()

        plt.quiver(lon, lat, u_slice, v_slice, scale=500, color='red', width=0.001)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(f"{self.save_path}/wind_{time_idx}.png")

        adj_mat = create_adjacency_matrix(u_slice.values, v_slice.values, lon, lat) * self.mask
        u_slice_recon, v_slice_recon = u_slice.copy(), v_slice.copy()
        u_slice_recon.values, v_slice_recon.values = reconstruct_uv_slices(adj_mat, lat, lon)

        plt.quiver(lon, lat, u_slice_recon, v_slice_recon, scale=500, color='red', width=0.001)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(f"{self.save_path}/wind_{time_idx}_init.png")

        return adj_mat / np.max(adj_mat)

    def __len__(self):
        return self.n_ts
    
    def __getitem__(self, idx):
        return {'xt': self.data['xt'][idx], 'ct': self.data['ct'][idx], 'ht': self.data['ht'][idx], 'st': self.data['st']}
    
if __name__ == "__main__":
    dataset = WeatherBench_dataset(SST_DATA_PATH)
    
    
    