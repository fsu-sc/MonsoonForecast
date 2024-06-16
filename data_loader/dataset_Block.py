import os
import xarray as xr
import pandas as pd
import random 
import matplotlib.pyplot as plt
from torchvision.transforms import Resize, Compose, ToTensor
from torchvision.transforms.functional import to_pil_image, to_tensor
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from base import BaseDataLoader
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np

from datetime import datetime, timedelta
onset_mask_df = pd.read_csv("/unity/f2/aoleksy/MonsoonForecast/onset_pen_FL.csv", names=['Year','OnsetDay'])

def day_of_year_to_date(year, day_of_year):
    # Check if the year is a leap year
    is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    
    # Determine the number of days in February based on whether it's a leap year
    february_days = 29 if is_leap_year else 28
    
    # Define the number of days in each month
    days_in_month = [31, february_days, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    # Iterate through the months to find the month and day corresponding to the day of the year
    month = 1
    while day_of_year > days_in_month[month - 1]:
        day_of_year -= days_in_month[month - 1]
        month += 1
    
    # Return the date as a datetime object
    return f"{year}-{month:02d}-{day_of_year:02d}"

def date_to_day_of_year(date):
    # Split the date into year, month, and day
    year, month, day = map(int, date.split('-'))
    
    # Check if the year is a leap year
    is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    
    # Define the number of days in February based on whether it's a leap year
    february_days = 29 if is_leap_year else 28
    
    # Define the number of days in each month
    days_in_month = [31, february_days, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    # Calculate the day of the year
    day_of_year = sum(days_in_month[:month - 1]) + day
    
    return day_of_year

def date_subtract(day, num):
    given_date = datetime.strptime(day, "%Y-%m-%d")
    result_date = given_date - timedelta(days= num)
    result_date_str = result_date.strftime("%Y-%m-%d")
    return result_date_str

#----




# def normalize_tensor(tensor):
#     mean = tensor.mean(dim=(0, 1, 2), keepdim=True)
#     std = tensor.std(dim=(0, 1, 2), keepdim=True)
#     normalized_tensor = (tensor - mean) / std
#     return normalized_tensor

class NetCDFDataset(Dataset):
    
    def __init__(self, data_dir, variables = ['tp', 'mslp', 't2m', 'u200', 'u850', 'v200', 'v850']):
        self.data_dir = data_dir
        self.load_year_offsets(range(1990,2000), range(1,25))
        self.file_info_dict = {}
        self.file_dict = self.createFileInfoDict()  # Create a dictionary of file information while loading
        self.onset_mask_df = pd.read_csv("data/onset_pen_FL.csv", names=['Year', 'OnsetDay'])
        self.climato_tensor = torch.load('data/climato_tensor.pt')
        self.latitude_slice = slice(24, 33)  
        self.longitude_slice = slice(272,282)  
        self.variables = variables
        self.patch_size = (224, 224)  # Patch size for spatial tokenization
        self.x = []
        self.y = []
        self.var_means = None
        self.var_stds = None
        self.samples = []
        self.samples_dict = {}
        self.xy_dict = {}
        self.block_dict = {}
        
        
        if 'tp' in self.variables:
            self.tp_idx = self.variables.index('tp')

        self.load_data()
    
    def load_year_offsets(self, years, offsets):
        self.years = years
        self.offsets = offsets
        
        
       
    def createFileInfoDict(self):
    #data_dir = "/Net/elnino/data/obs/ERA5/global/daily/"
        all_files = os.listdir(self.data_dir)

        for filename in all_files:

            parts = filename.split("_")
            year = int(parts[-1].split(".")[0])
            variable = parts[0]
        
            # If the year is not already a key in the dictionary, initialize an empty DataFrame
            if year not in self.file_info_dict:
                self.file_info_dict[year] = pd.DataFrame(columns=["filename", "variable"])
        
            # Append information to the DataFrame associated with the year
            new_df = pd.DataFrame({"filename": [filename], "variable": [variable]})
            self.file_info_dict[year] = pd.concat([self.file_info_dict[year], new_df], ignore_index=True)
        
        return self.file_info_dict
    def loadERA5Data(self, year):
        filenames = self.file_info_dict.get(year, [])
        
        # if not filenames:
        #     print("No data available for the selected year.")
        #     return None
        
        ds_list = []
        for filename in filenames:
            ds = xr.open_dataset(os.path.join(self.data_dir, filename), engine='netcdf4')
            ds_list.append(ds)

        # Concatenate all datasets along the time dimension
        ds_year = xr.concat(ds_list, dim='time')
        
        return ds_year
    
    def plot_tp_vs_wind_speed(self, year):
        # Load ERA5 data for the desired year(s)
        ds = self.loadERA5Data(year)
        if ds is None:
            return
        
        # Calculate wind speed
        wind_speed = np.sqrt(ds['u200']**2 + ds['v200']**2)

        # Plot tp against wind speed
        plt.figure(figsize=(10, 6))
        plt.scatter(wind_speed.values.flatten(), ds['tp'].values.flatten(), alpha=0.5)
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Total Precipitation (mm)')
        plt.title(f'Total Precipitation vs Wind Speed ({year})')
        plt.grid(True)
        plt.show()

    


    def load_data(self):
        for yr in self.years:
            onset = onset_mask_df.loc[onset_mask_df['Year'] == yr, 'OnsetDay'].iloc[0]
            onset_tensor = torch.tensor(onset, dtype=torch.float32)
            msk_date = day_of_year_to_date(yr, onset)
            min_start_date = date_subtract(msk_date, (max(self.offsets) + 15))
            #trying different window lengths: 16 days
            max_end_date = date_subtract(msk_date, min(self.offsets))
            time_slice = slice(min_start_date, max_end_date) #+1?
            datasets = [xr.open_dataset(os.path.join(self.data_dir, self.file_dict[yr].iloc[i]['filename']),)
                                                    for i in range(len(self.file_dict[yr]))]
            merged_ds = xr.merge(datasets)
            # merged_ds = merged_ds.sel(time=time_slice, latitude=self.latitude_slice, longitude=self.longitude_slice)
            merged_ds = merged_ds.sel(time=time_slice)
            # tensor_values = [merged_ds[var].values for var in merged_ds.data_vars if var in self.variables]
            stacked_tensors = []
            for var_name in self.variables:
                var_data = merged_ds[var_name].values
                var_mean = np.nanmean(var_data)
                var_std = np.nanstd(var_data)
                var_normalized = (var_data - var_mean) / var_std
                var_tensor = torch.tensor(var_normalized, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
                # var_tensor = torch.tensor(var_data, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
                stacked_tensors.append(var_tensor)
            
            # tensor_values = np.stack(stacked_tensors[0], axis=0)
            # tensor = torch.tensor(tensor_values, dtype=torch.float32)
            # normalized_tensor = normalize_tensor(tensor)
            normalized_tensor = stacked_tensors[0].permute(1,0,2,3)
            tp_tensor = stacked_tensors[0][:, self.tp_idx, :, :]
            climate_slice = slice((onset - max(self.offsets) - 15), (onset - min(self.offsets) + 1))
            anom_tensor = tp_tensor - self.climato_tensor[climate_slice, :, :, :]
            normalized_tensor = torch.cat((normalized_tensor, anom_tensor), dim = 1)
            # self.block_dict[yr] = [normalized_tensor, onset_tensor]
            # Should be one number
            sum_anomaly_by_batch = torch.sum(normalized_tensor[:,1,:,:], dim=(1, 2))
            anom_mean = torch.mean(sum_anomaly_by_batch)
            anom_sd = torch.std(sum_anomaly_by_batch)
            sum_anomaly_by_batch = (sum_anomaly_by_batch - anom_mean) / anom_sd
            # Should be one number
            sum_by_batch = torch.sum(normalized_tensor[:,0,:,:], dim=(1, 2))
            new_tensor = torch.stack((sum_by_batch, sum_anomaly_by_batch), dim=-1)
            self.block_dict[yr] = [new_tensor, onset_tensor]    
            
            # The input should be two numbers
            # self.block_dict[yr] = [normalized_tensor, onset_tensor]
    # def load_data(self):
    #     for yr in self.years:
    #         onset = onset_mask_df.loc[onset_mask_df['Year'] == yr, 'OnsetDay'].iloc[0]
    #         onset_tensor = torch.tensor(onset, dtype=torch.float32)
    #         msk_date = day_of_year_to_date(yr, onset)
    #         min_start_date = date_subtract(msk_date, (max(self.offsets) + 15))
    #         max_end_date = date_subtract(msk_date, min(self.offsets))
    #         time_slice = slice(min_start_date, max_end_date)

    #         datasets = [xr.open_dataset(os.path.join(self.data_dir, self.file_dict[yr].iloc[i]['filename']))
    #                     for i in range(len(self.file_dict[yr]))]
    #         merged_ds = xr.merge(datasets)
    #         merged_ds = merged_ds.sel(time=time_slice)

    #         # Normalize and stack variables along the channel dimension
    #         stacked_tensors = []
    #         for var_name in self.variables:
    #             var_data = merged_ds[var_name].values
    #             var_mean = np.nanmean(var_data)
    #             var_std = np.nanstd(var_data)
    #             var_normalized = (var_data - var_mean) / var_std
    #             var_tensor = torch.tensor(var_normalized, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
    #             stacked_tensors.append(var_tensor)

    #         # Concatenate along the channel dimension
    #         normalized_tensor = torch.cat(stacked_tensors, dim=0)
    #         normalized_tensor = normalized_tensor.permute(1, 0, 2, 3)
    #         # Slice tp_tensor for tp_idx
    #         tp_tensor = normalized_tensor[:, self.tp_idx, :, :]

    #         # Compute anomaly tensor
    #         climate_slice = slice((onset - max(self.offsets) - 15), (onset - min(self.offsets) + 1))
    #         anom_tensor = tp_tensor - self.climato_tensor[climate_slice, 0, :, :]
    #         new_tensor = tp_tensor - self.climato_tensor[climate_slice, 0, :, :]
    #         # Prepare normalized tensor

    #         normalized_tensor = torch.cat((normalized_tensor, new_tensor.unsqueeze(1)), dim=1)

    #         self.block_dict[yr] = [normalized_tensor, onset_tensor]


    def __len__(self):
        return len(self.block_dict)

    def __getitem__(self, index):
        # Determine the year and offset based on the index
        print("IM HERE!!!!!!!!!!!")
        yr = self.years[index // len(self.offsets)]
        off = self.offsets[index % len(self.offsets)]
        # yr, off = yr_off_tuple

        off_tensor = torch.tensor(off, dtype=torch.float32)
        shape = self.block_dict[yr][0].shape[0]
        slice_b = shape - off + 1
        slice_a = slice_b - 16
        #sum_tp_tensor = torch.sum(self.block_dict[yr][0][slice_a:slice_b,0], dim =0)
        sum_anom_tensor = torch.sum(self.block_dict[yr][0][slice_a:slice_b,1], dim =0)
        return sum_anom_tensor , off_tensor


    def shuffle(self):
        random.shuffle(self.samples)
    def batch(self, batch_size):
        for i in range(0, len(self.samples), batch_size):
            batch = self.samples[i:i + batch_size]
            x_batch, y_batch = zip(*batch)  # Unzip the batch into x and y
            yield torch.stack(x_batch), torch.stack(y_batch)
            

#data_dir = "/Net/elnino/data/obs/ERA5/global/daily/"
#year = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]
#offsets = range(1,25)

#ds_indices =  [(yr, off) for yr in year for off in offsets]

#print("Landmark before dataset init")        
#ds = NetCDFDataset(data_dir, year, offsets)
#print("Landmark before getitem call")
#print("Sample get item output: ", ds.__getitem__(2002, 5))

#ds_sampler = SubsetRandomSampler(ds_indices)
#ds_loader = DataLoader(ds, batch_size=8, sampler=ds_sampler)


#count = 0
#for inputs, targets in ds_loader:
#        count+=1
#print("Count: ", count)
#for i in range(1, 25, 2):
#    ds.__getitem__(2002, i)
#    print(i)


#print("nothing")

class DefaultDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = NetCDFDataset(self.data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)