import os
import xarray as xr
import pandas as pd
import random 
from torchvision.transforms import Resize, Compose, ToTensor
from torchvision.transforms.functional import to_pil_image, to_tensor
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


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
    
    def __init__(self, data_dir, 
                 years, offsets, 
                 variables = ['tp', 'mslp', 't2m', 'u200', 'u850', 'v200', 'v850'],eval=False):
        self.data_dir = data_dir
        self.file_dict = self.createFileInfoDict()  # Create a dictionary of file information while loading
        self.offsets = offsets
        self.years = years
        self.onset_mask_df = pd.read_csv("/unity/f2/aoleksy/MonsoonForecast/onset_pen_FL.csv", names=['Year', 'OnsetDay'])
        self.climato_tensor = torch.load('/unity/f2/aoleksy/MonsoonForecast2/MonsoonForecast/climato_tensor.pt')
        self.latitude_slice = slice(24, 33)  
        self.longitude_slice = slice(272,282)  
        self.variables = variables
        self.patch_size = (224, 224)  # Patch size for spatial tokenization
        self.x = []
        self.y = []
        self.samples = []
        self.samples_dict = {}
        self.xy_dict = {}
        self.block_dict = {}

        if 'tp' in self.variables:
            self.tp_idx = self.variables.index('tp')

        self.load_data = self.load_data()
        
       
    def createFileInfoDict(self):
    #data_dir = "/Net/elnino/data/obs/ERA5/global/daily/"
        all_files = os.listdir(self.data_dir)
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
            merged_ds = merged_ds.sel(time=time_slice, latitude=self.latitude_slice, longitude=self.longitude_slice)
            tensor_values = [merged_ds[var].values for var in merged_ds.data_vars if var in self.variables]
            tensor_values = np.stack(tensor_values, axis=0)
            tensor = torch.tensor(tensor_values, dtype=torch.float32)
            normalized_tensor = normalize_tensor(tensor)
            normalized_tensor = normalized_tensor.permute(1,0,2,3)
            tp_tensor = tensor[:, self.tp_idx, :, :]
            climate_slice = slice((onset - max(self.offsets) - 15), (onset - min(self.offsets) + 1))
            anom_tensor = tp_tensor - self.climato_tensor[climate_slice, :, :, :]
            normalized_tensor = torch.cat((normalized_tensor, anom_tensor), dim = 1)
            self.block_dict[yr] = [normalized_tensor, onset_tensor]
            #tensor = tensor.permute(1,0,2,3)
            #self.block_dict[yr] = [tensor, onset_tensor]

    def __len__(self):
        return len(self.block_dict)

    def __getitem__(self, yr_off_tuple): #idx is offset
        yr, off = yr_off_tuple
        off_tensor = torch.tensor(off, dtype=torch.float32)
        shape = self.block_dict[yr][0].shape[0]
        slice_b = shape - off + 1
        slice_a = slice_b - 16
        #return (self.block_dict[yr][0][slice_a:slice_b, :, :, :], self.block_dict[yr][1])
        return (self.block_dict[yr][0][slice_a:slice_b, :, :, :], off_tensor)
        # return self.samples[index]
        # normalized_tensor, onset = self.samples[index]
        # return normalized_tensor, onset

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