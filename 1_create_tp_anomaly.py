# %%
import xarray as xr
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import xarray as xr
import pandas as pd
# %% 
# What this code generates is a tensor with the shape (number of fields, number of years, days in year)
# It contains the cumulative anomaly of tp for each year and day of the year

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
output_imgs_folder = "/Net/work/ozavala/CODE/MonsoonForecast/imgs/"
# Years available from 1940 to 2022
# start_year = 1990
start_year = 1979
# end_year = 2000
end_year = 2022

# Rolling window parameters
roll_size = 14# Size of rolling window in days
 
# bbox_positive = [25.0, 30.0, 276.0, 280.0]  # From image in trello
# bbox = [bbox_positive[0], bbox_positive[1], bbox_positive[2]-360, bbox_positive[3]-360]
bbox_positive = [24.0, 32.0, 270.0, 281.0]  # For florida
# bbox_positive = [24, 32, -87, -80]  # For florida
debug = False
# Available variables: ['tp', 'mslp', 't2m', 'u200', 'u850', 'v200', 'v850']
# Explain each variable:
# tp: total precipitation
# mslp: mean sea level pressure
# t2m: 2m temperature
# u200: 200hPa zonal wind
# u850: 850hPa zonal wind
# v200: 200hPa meridional wind
# v850: 850hPa meridional wind
fields = ['tp', 't2m', 'u200']
tot_fields = len(fields)

file_dict = createFileInfoDict(data_dir, fields)
# Initialize lists to store the cumulative and mean precipitation tensors
year_spatial_mean = torch.zeros((tot_fields, end_year-start_year+1, 365))

# %% Getting the land mask from AVISO dataset
aviso_file_name = "//unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM/1993-01.nc"
aviso_dataset = xr.open_dataset(aviso_file_name)

# %% Computes the mean climatology tp tensor
first_time = True
for yr in range(start_year, end_year+1):
    year_data = []
    for i, cur_field in enumerate(fields):
        print(f"Processing year {yr} for {cur_field}")
        year_idx = yr-start_year
        # Select the tp dataset for the current year
        file_name = file_dict[yr]['filename'][file_dict[yr]['variable']==cur_field].values[0]
        year_dataset = xr.open_dataset(os.path.join(data_dir, file_dict[yr]['filename'][file_dict[yr]['variable']==cur_field].values[0]))
        # Crops the dataset to the region of interest (florida)
        year_dataset = year_dataset.sel(latitude=slice(bbox_positive[0], bbox_positive[1]), longitude=slice(bbox_positive[2], bbox_positive[3]))
        # Selects all days except leap days
        year_dataset = year_dataset.sel(time=~((year_dataset['time.month'] == 2) & (year_dataset['time.day'] == 29)))  
        # Converts the dataset to a tensor for the selected variables
        year_data.append(torch.tensor(year_dataset[cur_field].values, dtype=torch.float32))

        if first_time:
            print("Resampling aviso mask to match year_dataset grid resolution")
            # Resample aviso mask to match year_dataset grid resolution
            aviso_dataset_crop = aviso_dataset.interp(latitude=year_dataset.latitude, longitude=(year_dataset.longitude-360))
            aviso_mask = np.isnan(aviso_dataset_crop['adt'][0,:,:])
            # Plot the mask
            fig, axs = plt.subplots(1,1, figsize=(10,8))
            axs.imshow(aviso_mask.values, origin='lower')
            plt.show()
            first_time = False

        # Apply the land mask to the data
        for j in range(year_data[i].shape[0]):
            year_data[i][j,:,:] *= aviso_mask.values
            # Set 0 values to nan
            year_data[i][j,:,:] = torch.where(year_data[i][j,:,:] == 0, torch.nan, year_data[i][j,:,:])

        # Compute the spatial mean tp tensor for the year
        year_spatial_mean[i, year_idx, :] = torch.nanmean(year_data[i], dim=(1,2))

    # Plot the mean field value over the full domain for all the fields
    if debug:
        # Two figures, one with the full data and the other with the mean applied
        fig, axs = plt.subplots(2,tot_fields, figsize=(25,10))
        for i in range(tot_fields):
            day_of_year = 10
            im = axs[0,i].imshow(year_data[i][day_of_year,:,:], origin='lower') # Plotting mean field value over the full domain
            plt.colorbar(im, ax=axs[0,i], shrink=0.5)
            axs[0,i].set_title(f"Full {fields[i]} for {yr} day {day_of_year}")
            axs[1,i].scatter(range(365), year_spatial_mean[i, year_idx, :])
            axs[1,i].set_title(f"Mean {fields[i]} for {yr} ")
        plt.show()

# %%
# Initialize tensors to store different types of anomalies and means
# Shape: (number of fields, number of years, days in year)
year_cum_anomaly = torch.zeros((tot_fields, end_year-start_year+1, 365)) # (correct) Stores cumulative anomaly for each field/year/day
year_anomaly = torch.zeros((tot_fields, end_year-start_year+1, 365)) # (correct) Stores daily anomaly for each field/year/day
year_rolling_mean = torch.zeros((tot_fields, end_year-start_year+1, 365)) # (correct) Stores rolling mean for each field/year/day
climo_rolling_mean_by_year = torch.zeros((tot_fields, 365)) # (correct) Stores climatological rolling mean for each field/day

# Calculate overall mean across all years for each field
# Shape: (number of fields,)
overall_mean = np.array([torch.nanmean(year_spatial_mean[i,:,:]) for i in range(tot_fields)])  # Correct


# Loop through each year and field to calculate anomalies
for cur_year in range(start_year, end_year+1):
    for cur_field in range(tot_fields):
        # Get index relative to start year
        year_idx = cur_year-start_year
        
        # Calculate rolling mean for each window position
        for i in range(365-roll_size):
            # Calculate mean over rolling window for current year/field
            year_rolling_mean[cur_field, year_idx, i] = torch.nanmean(year_spatial_mean[cur_field, year_idx, i:i+roll_size])

        # Calculate anomaly by subtracting overall mean from rolling mean
        year_anomaly[cur_field, year_idx, :] = year_rolling_mean[cur_field, year_idx, :] - overall_mean[cur_field]

# Calculate climatological rolling mean
climo_rolling_mean = year_rolling_mean.sum(axis=1)/(end_year-start_year+1)
# Calculate cumulative anomaly by taking cumulative sum of daily anomalies
year_cum_anomaly = year_anomaly.cumsum(axis=2)

# Normalize all the anomalies to mean zero and std 1
year_cum_anomaly_normalized = (year_cum_anomaly - torch.mean(year_cum_anomaly, dim=2, keepdim=True))/torch.std(year_cum_anomaly, dim=2, keepdim=True)
print("Done!")

# Assuming 'tensor_list' is your list of tensors, each containing 365 values
# %% Save the year_cum_tp_anomaly tensor with shape (number of fields, number of years, days in year)
output_folder = "/Net/work/ozavala/CODE/MonsoonForecast/"
torch.save(year_cum_anomaly, os.path.join(output_folder, 'year_cum_tp_anomaly.pt'))
torch.save(year_cum_anomaly_normalized, os.path.join(output_folder, 'year_cum_tp_anomaly_normalized.pt'))

# %%

onset_mask_file_name = "/Net/work/ozavala/CODE/MonsoonForecast/onset_pen_FL.csv"
onset_mask_df = pd.read_csv(onset_mask_file_name, names=['Year', 'OnsetDay'])
# anomaly_file = os.path.join(output_folder, 'year_cum_tp_anomaly_normalized.pt')
anomaly_file = os.path.join(output_folder, 'year_cum_tp_anomaly.pt')
year_cum_anomaly_test = torch.load(anomaly_file)

# Make two plot 
for cur_year in range(start_year, end_year+1):
    print(f"Processing year {cur_year}")
    # Read the onset day for the current year
    onset_day = onset_mask_df.loc[onset_mask_df['Year'] == cur_year, 'OnsetDay'].values[0]
    cur_year_idx = cur_year-start_year # Year to use for testing/visualization
    fig, axs = plt.subplots(3,tot_fields, figsize=(5*tot_fields,13))
    for i, cur_field in enumerate(fields):
        # Plot the mean climatology tp tensor
        axs[2,i].plot(range(365 - roll_size), year_cum_anomaly_test[i, cur_year_idx, :365 - roll_size], label='cum anomaly', color='green')
        axs[2,i].set_title(f"Cumulative anomaly {cur_field} for year {cur_year}")
        # Scatter the onset day
        axs[2,i].scatter(onset_day, year_cum_anomaly_test[i, cur_year_idx, onset_day], color='red', label=f'onset day: {onset_day}')
        axs[2,i].legend()
        # Plot the anomaly
        axs[1,i].plot(range(365 - roll_size), year_anomaly[i, cur_year_idx, :365 - roll_size], label='anomaly')
        axs[1,i].legend()

        axs[0,i].plot(range(365 - roll_size), climo_rolling_mean[i, :365 - roll_size], label='climo')
        axs[0,i].plot(range(365 - roll_size), year_rolling_mean[i, cur_year_idx, :365 - roll_size], label='year roll mean')
        # Plot the overall mean tp tensor as a dashed horizontal line
        axs[0,i].scatter(onset_day, overall_mean[i], color='green', label='onset day')
        axs[0,i].axhline(overall_mean[i], color='red', linestyle='--', label='overall mean')
        axs[0,i].set_title(f"Mean climatology,  {cur_field} for year {cur_year}")
        axs[0,i].legend()
    plt.legend()
    plt.savefig(os.path.join(output_imgs_folder, f"tp_anomaly_{cur_year}.png"))
    plt.show()

# %% 
# Print mean and STD of the onset day
print(f"Mean onset day: {onset_mask_df['OnsetDay'].mean()}")
print(f"STD onset day: {onset_mask_df['OnsetDay'].std()}")
