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
max_offset = 12
# start_year = 1990
start_year = 1989
# end_year = 2022
end_year = 2000

bbox = [24.0, 32.0, 270.0, 281.0]  # For florida
# bbox = [24, 32, -87, -80]  # For florida
debug = True
# Available variables: ['tp', 'mslp', 't2m', 'u200', 'u850', 'v200', 'v850']
fields = ['tp', 't2m']
tot_fields = len(fields)

file_dict = createFileInfoDict(data_dir, fields)
# Initialize lists to store the cumulative and mean precipitation tensors
year_spatial_mean = torch.zeros((tot_fields, end_year-start_year+1, 365))

# %% Computes the mean climatology tp tensor
# THIS PART IS CORRECT
for yr in range(start_year, end_year+1):
    for i, cur_field in enumerate(fields):
        print(f"Processing year {yr}")
        year_idx = yr-start_year
        # Select the tp dataset for the current year
        file_name = file_dict[yr]['filename'][file_dict[yr]['variable']==cur_field].values[0]
        year_dataset = xr.open_dataset(os.path.join(data_dir, file_dict[yr]['filename'][file_dict[yr]['variable']==cur_field].values[0]))
        # Crops the dataset to the region of interest (florida)
        year_dataset = year_dataset.sel(latitude=slice(bbox[0], bbox[1]), longitude=slice(bbox[2], bbox[3]))
        # Selects all days except leap days
        year_dataset = year_dataset.sel(time=~((year_dataset['time.month'] == 2) & (year_dataset['time.day'] == 29)))  
        # Converts the dataset to a tensor for the selected variables
        year_data = torch.tensor(year_dataset[cur_field].values, dtype=torch.float32)
        # Compute the spatial mean tp tensor for the year
        year_spatial_mean[i, year_idx, :] = torch.mean(year_data, dim=(1,2))
        # Plot the cumulative tp anomaly tensor for the year
        if debug:
            # Plot a single day of tp use cartopy to plot the map
            # fig, ax = plt.subplots(1,1, figsize=(10,8), subplot_kw={'projection': ccrs.PlateCarree()})
            # data_extent_cartopy = [bbox[2]-360, bbox[3]-360, bbox[0], bbox[1]]
            # ax.imshow(year_dataset['tp'][0,:,:].values, origin='upper', extent=data_extent_cartopy, transform=ccrs.PlateCarree())
            # ax.coastlines()
            # plt.show()
            plt.scatter(range(365), year_spatial_mean[i, year_idx, :])
            plt.title(f"Mean {cur_field} for {yr}")
            plt.show()

# %%
# Initialize tensors to store different types of anomalies and means
# Shape: (number of fields, number of years, days in year)
year_cum_anomaly = torch.zeros((tot_fields, end_year-start_year+1, 365)) # Stores cumulative anomaly for each field/year/day
year_anomaly = torch.zeros((tot_fields, end_year-start_year+1, 365)) # (correct) Stores daily anomaly for each field/year/day
year_rolling_mean = torch.zeros((tot_fields, end_year-start_year+1, 365)) # (correct) Stores rolling mean for each field/year/day
climo_rolling_mean_by_year = torch.zeros((tot_fields, 365)) # (correct) Stores climatological rolling mean for each field/day

# Calculate overall mean across all years for each field
# Shape: (number of fields,)
overall_mean = np.array([torch.mean(year_spatial_mean[i,:,:]) for i in range(tot_fields)])  # Correct

# Rolling window parameters
roll_size = 14# Size of rolling window in days

# %%
# Loop through each year and field to calculate anomalies
for cur_year in range(start_year, end_year+1):
    for cur_field in range(tot_fields):
        # Get index relative to start year
        year_idx = cur_year-start_year
        
        # Calculate rolling mean for each window position
        for i in range(365-roll_size):
            # Calculate mean over rolling window for current year/field
            year_rolling_mean[cur_field, year_idx, i] = torch.mean(year_spatial_mean[cur_field, year_idx, i:i+roll_size])

        # Calculate anomaly by subtracting overall mean from rolling mean
        year_anomaly[cur_field, year_idx, :] = year_rolling_mean[cur_field, year_idx, :] - overall_mean[cur_field]
        # Calculate cumulative anomaly by taking cumulative sum of daily anomalies
        # Calculate cumulative anomaly by taking cumulative sum of daily anomalies
        # Only accumulate up to 365-roll_size since rolling means are only valid for that range
        year_cum_anomaly[cur_field, year_idx, :365-roll_size] = torch.cumsum(year_anomaly[cur_field, year_idx, :365-roll_size], dim=0)
        # Set remaining days to 0 since we don't have valid rolling means for them
        year_cum_anomaly[cur_field, year_idx, 365-roll_size:] = 0

# %%
# Calculate climatological rolling mean
climo_rolling_mean = year_rolling_mean.sum(axis=1)/(end_year-start_year+1)

# %%

# Make two plot 
# for cur_year in range(start_year, end_year+1):
for cur_year in range(start_year, start_year+3):
    cur_year_idx = cur_year-start_year # Year to use for testing/visualization
    fig, axs = plt.subplots(2,2, figsize=(10,8))
    for i, cur_field in enumerate(fields):
        # Plot the mean climatology tp tensor
        # axs[1,i].plot(range(365 - roll_size), year_cum_anomaly[i, cur_year_idx, :365 - roll_size], label='cum anomaly')
        # axs[1,i].plot(range(365 - roll_size), year_anomaly[i, cur_year_idx, :365 - roll_size], label='anomaly')
        axs[1,i].set_title(f"Cumulative anomaly {cur_field} for year {cur_year}")
        axs[1,i].legend()

        axs[0,i].plot(range(365 - roll_size), climo_rolling_mean[i, :365 - roll_size], label='climo')
        axs[0,i].plot(range(365 - roll_size), year_rolling_mean[i, cur_year_idx, :365 - roll_size], label='year roll mean')
        # Plot the overall mean tp tensor as a dashed horizontal line
        axs[0,i].axhline(overall_mean[i], color='red', linestyle='--', label='overall mean')
        axs[0,i].set_title(f"Mean climatology,  {cur_field} for year {cur_year}")
        axs[0,i].legend()
    plt.legend()
    plt.savefig(f"tp_anomaly_{cur_year}.png")
    plt.show()

# %%
# Assuming 'tensor_list' is your list of tensors, each containing 365 values
# Save the year_cum_tp_anomaly tensor
torch.save(year_cum_tp_anomaly, 'climato_tensor.pt')