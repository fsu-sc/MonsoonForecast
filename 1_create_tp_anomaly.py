# %%
import xarray as xr
import os
# import data_loader.dataset_Block as ds 
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import torch
# import model.data_loader as dl
# import model.NNmodel as ECNN
# import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import xarray as xr
import pandas as pd
# import torchvision.models.vision_transformer as ViT
# import torchvision.models as models

def createFileInfoDict(data_dir, variables = ['tp', 'mslp', 't2m', 'u200', 'u850', 'v200', 'v850']):
    """
    Create a dictionary with the year as the key and the filename and variable as the values.
    """
    all_files = os.listdir(data_dir)
    file_info_dict = {}

    for filename in all_files:
        parts = filename.split("_")
        year = int(parts[-1].split(".")[0])
        variable = parts[0]
        if variable in variables:
            # If the year is not already a key in the dictionary, initialize an empty DataFrame
            if year not in file_info_dict:
                file_info_dict[year] = pd.DataFrame(columns=["filename", "variable"])
    
            # Append information to the DataFrame associated with the year
            new_df = pd.DataFrame({"filename": [filename], "variable": [variable]})
            file_info_dict[year] = pd.concat([file_info_dict[year], new_df], ignore_index=True)
    
    return file_info_dict

# %%
data_dir = "/Net/elnino/data/obs/ERA5/global/daily/"
# Years available from 1940 to 2022
offsets =[1,2,3,4,5,6,7,8,9,10,11,12]
# start_year = 1990
start_year = 1979
end_year = 2022
bbox = [24.0, 32.0, 270.0, 281.0]  # For florida
# bbox = [24, 32, -87, -80]  # For florida
debug = False

# Available variables: ['tp', 'mslp', 't2m', 'u200', 'u850', 'v200', 'v850']

file_dict = createFileInfoDict(data_dir, ['tp', 't2m'])

# Initialize lists to store the cumulative and mean precipitation tensors
year_mean_tp = torch.zeros((end_year-start_year+1, 365))
# %%
# Computes the mean climatology tp tensor
for yr in range(start_year, end_year+1):
# for yr in range(start_year, start_year+5):
    print(f"Processing year {yr}")
    cur_idx = yr-start_year
    # Select the tp dataset for the current year
    year_dataset = xr.open_dataset(os.path.join(data_dir, file_dict[yr]['filename'][file_dict[yr]['variable']=='tp'].values[0]))
    # Crops the dataset to the region of interest (florida)
    year_dataset = year_dataset.sel(latitude=slice(bbox[0], bbox[1]), longitude=slice(bbox[2], bbox[3]))
    # Selects all days except leap days
    year_dataset = year_dataset.sel(time=~((year_dataset['time.month'] == 2) & (year_dataset['time.day'] == 29)))  
    # Converts the dataset to a tensor for the selected variables
    year_tp = torch.tensor(year_dataset['tp'].values, dtype=torch.float32)
    # Compute the mean tp tensor for the year
    year_mean_tp[cur_idx, :] = torch.mean(year_tp, dim=(1,2))
    # Plot the cumulative tp anomaly tensor for the year
    if debug:
        # Plot a single day of tp use cartopy to plot the map
        # fig, ax = plt.subplots(1,1, figsize=(10,8), subplot_kw={'projection': ccrs.PlateCarree()})
        # data_extent_cartopy = [bbox[2]-360, bbox[3]-360, bbox[0], bbox[1]]
        # ax.imshow(year_dataset['tp'][0,:,:].values, origin='upper', extent=data_extent_cartopy, transform=ccrs.PlateCarree())
        # ax.coastlines()
        # plt.show()
        plt.plot(year_mean_tp[cur_idx, :])
        plt.title(f"Mean tp for {yr}")
        plt.show()

# %%
# Compute the mean climatology tp tensor using a rolling mean of 10 days
year_cum_tp_anomaly = torch.zeros((end_year-start_year+1, 365)) # Main one. Cumulative tp anomaly of each year (years, 365 days)
year_rolling_mean_tp = torch.zeros((end_year-start_year+1, 365)) # Mean total precipitation of each year (365 days)
overall_mean_tp = torch.mean(year_mean_tp) # Overall mean total precipitation of all the years
climo_rolling_mean_tp = torch.zeros(365) # Mean climatology total precipitation of all the years (365 days)
roll_size = 30
start_idx = roll_size//2
end_idx = 365-roll_size//2
test_year = 1998

for cur_year in range(start_year, end_year+1):
    cur_idx = cur_year-start_year
    for i in range(365-roll_size):
        year_rolling_mean_tp[cur_idx, i] = torch.mean(year_mean_tp[cur_idx, i:i+roll_size])
        climo_rolling_mean_tp[i] += year_rolling_mean_tp[cur_idx, i]
    year_cum_tp_anomaly[cur_idx, :] = year_rolling_mean_tp[cur_idx, :] - overall_mean_tp
    # Make the cummulative sum for each year
    year_cum_tp_anomaly[cur_idx, :] = torch.cumsum(year_cum_tp_anomaly[cur_idx, :], dim=0)

climo_rolling_mean_tp = climo_rolling_mean_tp/len(range(start_year, end_year+1))
# Plot the mean climatology tp tensor
plt.plot(range(start_idx, end_idx), year_cum_tp_anomaly[test_year-start_year, start_idx:end_idx], label='anomaly')
plt.plot(range(start_idx, end_idx), climo_rolling_mean_tp[start_idx:end_idx], label='climo')
plt.plot(range(start_idx, end_idx), year_rolling_mean_tp[test_year-start_year, start_idx:end_idx], label='year mean')
# Plot the overall mean tp tensor as a dashed horizontal line
plt.axhline(overall_mean_tp, color='red', linestyle='--', label='overall mean')
plt.title(f"Mean climatology tp for year {test_year}")
plt.legend()
plt.savefig("tp_anomaly.png")
plt.show()

# %%
# Assuming 'tensor_list' is your list of tensors, each containing 365 values
# Save the year_cum_tp_anomaly tensor
torch.save(year_cum_tp_anomaly, 'climato_tensor.pt')