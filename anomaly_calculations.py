# %% Basic imports and helper functions
import xarray as xr
import cupy_xarray
import os
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from os.path import join
from glob import glob

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

def get_file_list(data_dir, variables = ['tp', 'mslp', 't2m', 'u200', 'u850', 'v200', 'v850']):
    """
    Get a list of all the files in the data directory for the given variables.
    """
    databank = {}
    for variable in variables:
        # parse year from filename
        aux = sorted(glob(join(data_dir, f'{variable}_era5_day_*.nc')))
        # iterate over the files and parse the year
        
        for file in aux:
            year = int(file.split('_')[-1].split('.')[0])
            databank[f"{variable}_{year}"] = file
    return databank

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
# file_dict = createFileInfoDict(data_dir, ['tp'])  Old way
target_variable = 'tp'
databank = get_file_list(data_dir, [target_variable])

# %% Initialize  a cumulative tp anomaly cupy tensor
year_cum_tp_anomaly = cp.zeros((end_year-start_year+1, 365))
for ii, yr in enumerate(range(start_year, end_year+1)):
    print(f"Processing year {yr}")
    # Select the tp dataset for the current year
    try:
        year_dataset = xr.open_dataset(databank[f'{target_variable}_{yr}'])
    except:
        print(f"No {target_variable} data for {yr}")
        continue
    ## Subset the dataset to the region of interest (Florida)
    year_dataset = year_dataset.sel(latitude=slice(bbox[0], bbox[1]), longitude=slice(bbox[2], bbox[3]))
    ## Select all days except leap days
    year_dataset = year_dataset.sel(time=~((year_dataset['time.month'] == 2) & (year_dataset['time.day'] == 29)))
    ## Computer the mean for the year
    year_mean_tp = cp.mean(year_dataset[target_variable], axis=(1,2))
    year_cum_tp_anomaly[ii] = cp.asarray(year_mean_tp.data)
if debug:
    fig, ax = plt.subplots()
    ax.plot(year_cum_tp_anomaly.T.get())
    ax.set_title(f"{target_variable} cumulative anomaly in Florida for {start_year}-{end_year}")
    ax.grid()
    plt.show()

# %% get the daily mean tp
global_mean_tp = cp.mean(year_cum_tp_anomaly)
if debug:
    print(f'global_mean_tp.shape: {global_mean_tp.shape}')
    print(f'global_mean_tp: {global_mean_tp}')
# %% Rolling mean to daily means
roll_size = 30
year_rolling_mean_tp = cp.zeros((end_year-start_year+1, 365))
yearly_anomaly = cp.zeros((end_year-start_year+1, 365))
for ii, yr in enumerate(range(start_year, end_year+1)):
    aux_ds = xr.Dataset({'tp': (['time'], year_cum_tp_anomaly[ii].get())})
    rolling_mean_tp = aux_ds['tp'].rolling(time=roll_size, center=True).mean()
    year_rolling_mean_tp[ii, :] = cp.asarray(rolling_mean_tp.data)
    yearly_anomaly[ii, :] = year_rolling_mean_tp[ii, :] - global_mean_tp 
if debug:
    fig, ax = plt.subplots(2,1, figsize=(10,5))
    ax[0].plot(year_rolling_mean_tp.T.get())
    ax[0].axhline(global_mean_tp.get(), color='red', linestyle='--', label='global mean')
    ax[0].set_title(f"{target_variable} rolling mean anomaly in Florida for {start_year}-{end_year}")
    ax[0].grid()
    ax[1].plot(yearly_anomaly.T.get())
    ax[1].axhline(0, color='k', linestyle='--', label='zero')
    ax[1].set_title(f"{target_variable} yearly anomaly in Florida for {start_year}-{end_year}")
    ax[1].grid()
    plt.show()
# %% Create netcdf file
ds = xr.Dataset({'yearly_anomaly': (['year', 'day'], yearly_anomaly.get()),
                 'daily_means_series': (['year', 'day'], year_cum_tp_anomaly.get())}, 
                coords={'year': range(start_year, end_year+1), 'day': range(365)})
ds.attrs['description'] = f"Yearly {target_variable} anomaly in Florida for {start_year}-{end_year} computed from daily anomalies"
ds.attrs['climatology_average'] = global_mean_tp.get()
ds.to_netcdf(f'{target_variable}_yearly_anomaly_florida_{start_year}-{end_year}.nc', format='NETCDF4')

# %% Create yearly plots
#for yr in range(start_year, end_year+1):
#    fig, ax = plt.subplots()
#    ax.plot(yearly_anomaly[yr-start_year, :].get())
#    ax.set_title(f"{target_variable} yearly anomaly in Florida for {yr}")
#    ax.grid()
#    plt.show()