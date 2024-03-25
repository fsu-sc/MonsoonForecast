import os
import xarray as xr
import pandas as pd

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

#---

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

import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np


def normalize_tensor(tensor):
    mean = tensor.mean(dim=(0, 1, 2), keepdim=True)
    std = tensor.std(dim=(0, 1, 2), keepdim=True)
    normalized_tensor = (tensor - mean) / std
    return normalized_tensor

class NetCDFDataset(Dataset):
    def __init__(self, file_dict, data_dir, 
                 years, offsets, 
                 variables = ['tp', 'mslp', 't2m', 'u200', 'u850', 'v200', 'v850'],):
        self.file_dict = file_dict
        self.data_dir = data_dir
        self.latitude_slice = slice(24, 33)  
        self.longitude_slice = slice(272,282)  
        self.variables = variables
        self.x = []
        self.y = []
        self.xy_dict = {}
        for yr in years:
            for off in offsets:
                onset = onset_mask_df.loc[onset_mask_df['Year'] == yr, 'OnsetDay'].iloc[0]
                msk_date = day_of_year_to_date(yr, onset)
                end_date = date_subtract(msk_date, off)
                start_date = date_subtract(end_date, 5)

                time_slice = slice(start_date, end_date)
            
                datasets = [xr.open_dataset(os.path.join(self.data_dir, self.file_dict[yr].iloc[i]['filename']),)
                                                    for i in range(len(self.file_dict[yr]))]
                merged_ds = xr.merge(datasets)
                merged_ds = merged_ds.sel(time=time_slice, latitude=self.latitude_slice, longitude=self.longitude_slice)
                
                # Extract values from the xarray dataset and concatenate them along a new dimension
                tensor_values = [merged_ds[var].values for var in merged_ds.data_vars if var in self.variables]
                tensor_values = np.stack(tensor_values, axis=0)
                                        #change axis to 0
            
            
                # Convert the values to a PyTorch tensor
                tensor = torch.tensor(tensor_values, dtype=torch.float32)
            
                # Normalize the tensor
                normalized_tensor = normalize_tensor(tensor)
               
                #print("Norm Tens Shape:", normalized_tensor.shape)
            
                x_sing = normalized_tensor
                y_sing = onset
                xy_sing = [normalized_tensor, onset]
                self.x.append(x_sing)
                self.y.append(y_sing)
                self.xy_dict[(yr, off)] = xy_sing
                

    def __len__(self):
        return len(self.x)

    def __getitem__(self, yr_offset_pair):
        year, offset = yr_offset_pair
        return self.xy_dict[(year, offset)]