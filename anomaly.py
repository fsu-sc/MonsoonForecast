import xarray as xr
#import netCDF4 as nc
import os
import data_loader.dataset_Block as ds 
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import torch
import model.data_loader as dl
import model.NNmodel as ECNN
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import torchvision.models.vision_transformer as ViT
import torchvision.models as models
ViT_model = models.vit_b_16(weights=None)

  

data_dir = "/Net/elnino/data/obs/ERA5/global/daily/"
year = [1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2008, 2018]
offsets =[1,2,3,4,5,6,7,8,9,10,11,12]

variables = ['tp']

train_data = ds.NetCDFDataset(data_dir,year,offsets, variables)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

tensor_to_agg = train_data.__getitem__((2008, 12))[0]
print("done")

mean_tensor = torch.mean(tensor_to_agg, dim=(2, 3))
#cumulative_tensor = torch.cumsum(tensor_to_agg, dim=(2, 3))

print(mean_tensor.shape)



print("dbg")

import xarray as xr
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import pandas as pd

def normalize_tensor(tensor):
    mean = tensor.mean(dim=(0, 1, 2), keepdim=True)
    std = tensor.std(dim=(0, 1, 2), keepdim=True)
    normalized_tensor = (tensor - mean) / std
    return normalized_tensor

def createFileInfoDict(data_dir):
#data_dir = "/Net/elnino/data/obs/ERA5/global/daily/"
    all_files = os.listdir(data_dir)
    file_info_dict = {}


    for filename in all_files:

        parts = filename.split("_")
        year = int(parts[-1].split(".")[0])
        variable = parts[0]
    
        # If the year is not already a key in the dictionary, initialize an empty DataFrame
        if year not in file_info_dict:
            file_info_dict[year] = pd.DataFrame(columns=["filename", "variable"])
    
        # Append information to the DataFrame associated with the year
        new_df = pd.DataFrame({"filename": [filename], "variable": [variable]})
        file_info_dict[year] = pd.concat([file_info_dict[year], new_df], ignore_index=True)
    
    return file_info_dict

#opening mslp 2002 example file
data_dir = '/Net/elnino/data/obs/ERA5/global/daily'
file_dict = createFileInfoDict(data_dir)
#ds = xr.open_dataset(filepath)
#print(ds)

year_cumulative_tensors = []
year_mean_tensors = []
for yr in range(1990,2021):
    datasets = [xr.open_dataset(os.path.join(data_dir, file_dict[yr].iloc[i]['filename']),)
                                                        for i in range(len(file_dict[yr]))]
    merged_ds = xr.merge(datasets)
    merged_ds = merged_ds.sel(latitude=slice(24, 33), longitude=slice(272,282))
    merged_ds = merged_ds.sel(time=~((merged_ds['time.month'] == 2) & (merged_ds['time.day'] == 29)))  
    tensor_values = [merged_ds[var].values for var in merged_ds.data_vars if var in ['tp']]
    tensor_values = np.stack(tensor_values, axis=0)
    tensor = torch.tensor(tensor_values, dtype=torch.float32)
    normalized_tensor = normalize_tensor(tensor)
    #normalized_tensor = normalized_tensor.permute(1,0,2,3)
    #self.block_dict[yr] = [normalized_tensor, onset_tensor]
    tensor = normalized_tensor.permute(1,0,2,3)
    print(f"iter {yr}")
    print(tensor.shape)

    #year_mean_tensor = torch.mean(tensor, dim=(2,3))
    #year_mean_tensors.append(year_mean_tensor)
    year_mean_tensors.append(tensor)
    #year_cumulative_tensor = torch.cumsum(tensor, dim=(0,2,3))
    #year_cumulative_tensors.append(year_cumulative_tensor)



# Assuming 'tensor_list' is your list of tensors, each containing 365 values
# Convert the list of tensors to a single numpy array
tensor_array = np.stack(year_mean_tensors, axis=0)

# Take the mean along the first axis (axis 0) to get mean values for each day
mean_values = np.mean(tensor_array, axis=0)

print(mean_values.shape)  # Should be (365,)
mean_values = torch.tensor(mean_values)

#expanded_tensor = mean_values.unsqueeze(-1).unsqueeze(-1)
#print(expanded_tensor.shape)
#sparse_tensor = expanded_tensor.expand(365, 1, 37, 41)
#print(sparse_tensor.shape)
print("done 3")
torch.save(mean_values, 'climato_tensor.pt')
print("breakpoint")
