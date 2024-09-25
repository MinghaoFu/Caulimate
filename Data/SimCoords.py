import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def simulate_uniform_coords(num_points, extent, seed):
    """
    Generates approximately uniformly distributed coordinates within the given extent,
    considering an equal-area projection for latitudes.
    
    Parameters:
    - num_points: The total number of coordinates to generate.
    - extent: The geographic extent as [min_longitude, max_longitude, min_latitude, max_latitude].
    
    Returns:
    - A list of tuples, where each tuple represents (longitude, latitude).
    """
    min_lon, max_lon, min_lat, max_lat = extent
    rs = np.random.RandomState(seed)
    # Generate uniformly distributed longitudes
    longitudes = rs.uniform(min_lon, max_lon, num_points)
    
    # Generate latitudes with consideration for the Earth's curvature
    lat_distribution = rs.uniform(np.sin(np.radians(min_lat)), np.sin(np.radians(max_lat)), num_points)
    latitudes = np.degrees(np.arcsin(lat_distribution))
    
    return np.stack([longitudes, latitudes], axis=1)

def simulate_grid_coords(size, extent, seed, skew=False):
    """Simulate coordinates based on the extent [] on map.

    Args:
        size (tuple): shape of simulated data points
        extent (list): [min_lon, max_lon, min_lat, max_lat]
        skew (bool, optional): Defaults to True. Skewed points around grids are generated.
    """
    rs = np.random.RandomState(seed)
    if extent[1] - extent[0] == 360:
        lons = np.linspace(extent[0], extent[1], size[1] + 1)[:-1] # Remove the last element to avoid overlapping
    else:
        lons = np.linspace(extent[0], extent[1], size[1])
    if extent[3] - extent[2] == 180:
        lats = np.linspace(extent[2], extent[3], size[0] + 1)[:-1]
    else:
        lats = np.linspace(extent[2], extent[3], size[0])

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lon_points = lon_grid.flatten()
    lat_points = lat_grid.flatten()

    if skew:
        # Apply randomness
        lon_jitter = (extent[1] - extent[0]) / size[1] / 10  # 10% of the grid cell size
        lat_jitter = (extent[3] - extent[2]) / size[0] / 10
        lon_points += rs.uniform(-lon_jitter, lon_jitter, size=lon_points.shape)
        lat_points += rs.uniform(-lat_jitter, lat_jitter, size=lat_points.shape)

    # Combine the random points into a list of tuples or any format you prefer
    return np.stack([lon_points, lat_points], axis=1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cartopy
    import cartopy.crs as ccrs
    import numpy as np

    # Initialize plot with a specific projection (PlateCarree is common for maps)
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    extent = [-180, 180, -90, 90]
    ax.set_extent(extent)  # Set the extent to match the extent used for generating points

    coords = simulate_grid_coords((10, 10), extent, skew=True)
    coords = simulate_uniform_coords(100, extent)
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
    ax.add_feature(cartopy.feature.LAND, edgecolor='black')
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
    ax.add_feature(cartopy.feature.RIVERS)

    # Unzip coordinates and plot
    lons, lats = zip(*coords)
    ax.scatter(lons, lats, color='red', s=10, transform=ccrs.Geodetic(), label='Random Points')

    # Add gridlines, labels, and title for better readability
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    plt.title('Random Coordinates within Specified Extent')
    plt.legend(loc='upper left')

    plt.show()
    plt.savefig('./SimCoords.png')


