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
    def __init__(self, file_dict, data_dir):
        self.file_dict = file_dict
        self.data_dir = data_dir
   
    def __len__(self):
        return len(self.file_dict)

    def __getitem__(self, year, offset):
        onset = onset_mask_df.loc[onset_mask_df['Year'] == year, 'OnsetDay'].iloc[0]
        msk_date = day_of_year_to_date(year, onset)
        end_date = date_subtract(msk_date, offset)
        start_date = date_subtract(end_date, 5)

        time_slice = slice(start_date, end_date)
        latitude_slice = slice(24, 33)  
        longitude_slice = slice(272,282)  
     
        datasets = [xr.open_dataset(os.path.join(self.data_dir, self.file_dict[year].iloc[i]['filename']),)
                                            for i in range(len(self.file_dict[year]))]
        merged_ds = xr.merge(datasets)
        merged_ds = merged_ds.sel(time=time_slice, latitude=latitude_slice, longitude=longitude_slice)
        
        # Concatenate all variables into one tensor
        variables = ['tp', 'mslp', 't2m', 'u200', 'u850', 'v200', 'v850']  # Specify the variable names in your dataset
        
        # Extract values from the xarray dataset and concatenate them along a new dimension
        tensor_values = [merged_ds[var].values for var in merged_ds.data_vars]
        tensor_values = np.stack(tensor_values, axis=-1)
    
    
        # Convert the values to a PyTorch tensor
        tensor = torch.tensor(tensor_values, dtype=torch.float32)
    
        # Normalize the tensor
        normalized_tensor = normalize_tensor(tensor)
        #print("Norm Tens Shape:", normalized_tensor.shape)
    
        return [normalized_tensor, onset]
        # Append the onset mask date as a value to the tensor
        #onset_mask_tensor = torch.tensor(onset, dtype=torch.int)
        #print("Onset Tens Shape:", onset_mask_tensor.shape)
        #print(onset_mask_tensor)
        #print("Shape 3:", torch.tensor([onset_mask_tensor, normalized_tensor]).shape)
        #tensor_with_mask = torch.cat([onset_mask_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0), normalized_tensor], dim=0)
    
       #return tensor_with_mask
    
    #def get_batch():

        #return batch