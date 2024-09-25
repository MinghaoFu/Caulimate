from Caulimate.Utils.Tools import makedir
import xarray as xr

z500 = xr.open_mfdataset('/l/users/minghao.fu/dataset/WeatherBench_data_full/temperature_850/*.nc', combine='by_coords')
u = xr.open_mfdataset('/l/users/minghao.fu/dataset/WeatherBench_data_full/u_component_of_wind/*.nc', combine='by_coords')
v = xr.open_mfdataset('/l/users/minghao.fu/dataset/WeatherBench_data_full/v_component_of_wind/*.nc', combine='by_coords')

# Define the latitude and longitude boundaries
lat_min, lat_max = -20, 20   # Adjust these values to your desired latitude range
lon_min, lon_max = -60, 60   # Adjust these values to your desired longitude range

# Select the region within the defined boundary for both datasets
u_region = u.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
v_region = v.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
z500 = z500.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

makedir('/l/users/minghao.fu/dataset/WeatherBench_data_subset/', remove_exist=True)

# Save the selected regions to new NetCDF files
u_region.to_netcdf('/l/users/minghao.fu/dataset/WeatherBench_data_subset/v_component_of_wind.nc')
v_region.to_netcdf('/l/users/minghao.fu/dataset/WeatherBench_data_subset/v_component_of_wind.nc')
z500.to_netcdf('/l/users/minghao.fu/dataset/WeatherBench_data_subset/temperature_850.nc')

