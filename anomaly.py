import xarray as xr
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import torch

data_dir = '/Net/elnino/data/obs/ERA5/global/daily'

def createFileInfoDict(data_dir):
    all_files = os.listdir(data_dir)
    file_info_dict = {}

    for filename in all_files:
        parts = filename.split("_")
        year = int(parts[-1].split(".")[0])
        variable = parts[0]

        if year not in file_info_dict:
            file_info_dict[year] = pd.DataFrame(columns=["filename", "variable"])

        new_df = pd.DataFrame({"filename": [filename], "variable": [variable]})
        file_info_dict[year] = pd.concat([file_info_dict[year], new_df], ignore_index=True)
    
    return file_info_dict

file_dict = createFileInfoDict(data_dir)

year_mean_tensors = []
years = range(1990, 2021)
for yr in years:
    datasets = [xr.open_dataset(os.path.join(data_dir, file_dict[yr].iloc[i]['filename'])) for i in range(len(file_dict[yr]))]
    merged_ds = xr.merge(datasets)
    merged_ds = merged_ds.sel(time=~((merged_ds['time.month'] == 2) & (merged_ds['time.day'] == 29)))
    
    tensor_values = [merged_ds[var].values for var in merged_ds.data_vars if var in ['tp']]
    tensor_values = np.stack(tensor_values, axis=0)
    tensor = torch.tensor(tensor_values, dtype=torch.float32)
    tensor = tensor.permute(1, 0, 2, 3)
    
    print(f"iter {yr}")
    print(tensor.shape)

    year_mean_tensors.append(tensor)

tensor_array = np.stack(year_mean_tensors, axis=0)
mean_values = np.mean(tensor_array, axis=0)
print(mean_values.shape)

mean_values = torch.tensor(mean_values)

# Save to NetCDF
mean_values_np = mean_values.numpy()
dims = ('time', 'variable', 'latitude', 'longitude')
coords = {
    'time': np.arange(mean_values_np.shape[0]),
    'variable': ['tp'],
    'latitude': merged_ds['latitude'].values,
    'longitude': merged_ds['longitude'].values
}
data_vars = {'mean_tp': (dims, mean_values_np)}

mean_ds = xr.Dataset(data_vars, coords)

# Save to NetCDF file
output_path = 'climato_tensor.nc'
mean_ds.to_netcdf(output_path)
print(f"Saved to {output_path}")
