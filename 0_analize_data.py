import xarray as xr
import netCDF4 as nc
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import pandas as pd

#opening mslp 2002 example file
filepath = '/Net/elnino/data/obs/ERA5/global/daily/mslp_era5_day_2002.nc'
ds = xr.open_dataset(filepath)
print(ds)

#plotting mslp distribution for the example above
print(ds['mslp'])
ds['mslp'].plot()
plt.show()
ds.close()

#mlsp plot of just Florida Region
ds_flor = ds.sel(latitude=slice(22, 32), longitude=slice(275,280))
#print(ds_flor['mslp'])
ds_flor['mslp'].plot()
plt.show()
ds_flor.close()


#plotting t2m distribution for t2m file of the same year
ds = xr.open_dataset('/Net/elnino/data/obs/ERA5/global/daily/t2m_era5_day_2002.nc')
print(ds['t2m'])
ds['t2m'].plot()
plt.show()
ds.close()

#t2m plot of just Florida Region
ds_flor = ds.sel(latitude=slice(22, 32), longitude=slice(275,280))
ds_flor['t2m'].plot()
plt.show()
ds_flor.close()


#plotting u200 distribution of u200 file of the same year
ds = xr.open_dataset('/Net/elnino/data/obs/ERA5/global/daily/u200_era5_day_2002.nc')
print(ds['u200'])
ds['u200'].plot()
plt.show()
ds.close()


#plotting u850 distr of the same year
ds = xr.open_dataset('/Net/elnino/data/obs/ERA5/global/daily/u850_era5_day_2002.nc')
#print(ds['u850'])
ds['u850'].plot()
plt.show()
ds.close()

#plotting v200 distr of the same year
ds = xr.open_dataset('/Net/elnino/data/obs/ERA5/global/daily/v200_era5_day_2002.nc')
#print(ds['v200'])
ds['v200'].plot()
plt.show()
ds.close()

#plotting v850 distr of the same year
ds = xr.open_dataset('/Net/elnino/data/obs/ERA5/global/daily/v850_era5_day_2002.nc')
#print(ds['v850'])
ds['v850'].plot()
plt.show()
ds.close()




#Demonstrating File Sizes
folder_path = '/Net/elnino/data/obs/ERA5/global/daily/'
nc_files = [f for f in os.listdir(folder_path) if f.endswith('.nc')]
# Create a dataset to store file sizes
sizes_dataset = xr.Dataset()
file_sizes = []
# Iterate through each NetCDF file
for nc_file in nc_files:
    file_path = os.path.join(folder_path, nc_file)
    file_size = os.path.getsize(file_path)
    sizes_dataset[nc_file] = file_size
    file_sizes.append(file_size)
#print("List of file sizes:", file_sizes)
print("How many files:", len(file_sizes))
print("Mean File size:", (np.mean(file_sizes) /  1073741824), 'GB') #1073741824 bytes in a GB
print("File Size Range:", ((np.max(file_sizes) - np.min(file_sizes)) / 1073741824), 'GB')
# Create a histogram
plt.hist(file_sizes, bins=2, color='blue', edgecolor='black')
plt.xlabel('File Size')
plt.ylabel('Frequency')
plt.title('Distribution of File Sizes')

#There are 581 files, 7 files for each variables for each of the 83 years; 83 * 7 = 581
num_files = len(nc_files)
print(f"Number of NetCDF files: {num_files}")

#Mean MSLP plot for this example file
filepath_ex = '/Net/elnino/data/obs/ERA5/global/daily/mslp_era5_day_2002.nc'
ds_ex = xr.open_dataset(filepath_ex)
#taking mean mslp and plotting for this example year
mean_mslp = ds_ex['mslp'].mean(dim=('latitude', 'longitude'))
# Convert time to a format that Matplotlib understands
mean_mslp['time'] = pd.to_datetime(mean_mslp['time'].values)
mean_mslp.plot.line(x='time')
plt.title('Mean MSLP Daily for 2002')
plt.xlabel('Time')
plt.ylabel('MSLP Measurement')
plt.show()

#Mean Precipitation for this example year
filepath_ex_tp = '/Net/elnino/data/obs/ERA5/global/daily/tp_era5_day_2002.nc'
ds_ex_tp = xr.open_dataset(filepath_ex_tp)
mean_tp = ds_ex_tp['tp'].mean(dim=('latitude', 'longitude'))
mean_tp.plot.line(x='time')
plt.title('2002 Time Series Plot of Mean Total Precipitation')
plt.xlabel('Time')
plt.ylabel('Mean Total Precipitation')
plt.show()


#World Map Projection of Precipitation for this example year using cartopy
filepath_ex_tp = '/Net/elnino/data/obs/ERA5/global/daily/tp_era5_day_2002.nc'
ds_ex_tp = xr.open_dataset(filepath_ex_tp)
projection = ccrs.PlateCarree()
fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(10, 6))

ds_ex_tp_transposed = ds_ex_tp.transpose('time', 'latitude', 'longitude')
ds_one_day = ds_ex_tp_transposed.sel(time='2002-01-01')

ds_one_day
ds_one_day['tp'].plot.pcolormesh(ax=ax, transform=projection, cmap='rainbow', x='longitude', y='latitude', add_colorbar=True)

ax.coastlines()
ax.gridlines(draw_labels=True, linestyle='--', color='black', alpha=0.5)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

plt.title('World Map Plot of mslp for Jan 1 2002')
plt.show()

#Florida Region Cropped Map Projection of Precipitation for this example year using cartopy
ds_ex_tp = xr.open_dataset(filepath_ex_tp)
ds_ex_tp = ds_ex_tp.sel(latitude=slice(22, 32), longitude=slice(265,285)) #**the proposed Florida Crop; can be modified to suit our purposes**

ds_ex_tp_transposed = ds_ex_tp.transpose('time', 'latitude', 'longitude')
ds_one_day = ds_ex_tp_transposed.sel(time='2002-01-01')

projection = ccrs.PlateCarree()
fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(10, 6))

ds_one_day['tp'].plot.pcolormesh(ax=ax, transform=projection, cmap='rainbow', x='longitude', y='latitude', add_colorbar=True)

ax.coastlines()
ax.gridlines(draw_labels=True, linestyle='--', color='black', alpha=0.5)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

plt.title('Florida Region Crop Plot of Total Precipitation for Jan 1 2002')
plt.show()