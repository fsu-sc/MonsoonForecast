import os
import xarray as xr
import pandas as pd
import random 
import matplotlib.pyplot as plt
from torchvision.transforms import Resize, Compose, ToTensor
from torchvision.transforms.functional import to_pil_image, to_tensor
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
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

class NetCDFDataset(Dataset):
    
    def __init__(self, max_offset = 12, lead_time = 10, start_year = 1979):
        print("Reading data....")
        self.onset_mask_df = pd.read_csv("onset_pen_FL.csv", names=['Year', 'OnsetDay'])
        self.climato_tensor = torch.load('climato_tensor.pt')
        self.total_years = self.climato_tensor.shape[0]
        self.max_offset = max_offset
        self.lead_time = lead_time
        print("Done!")
        # Generate a dataset that contains all the years and adjusted offsets
        self.X = np.zeros((self.total_years*self.max_offset, self.lead_time))
        self.Y = np.zeros(self.total_years*self.max_offset)
        for yr in range(self.total_years):
            onset_day = onset_mask_df.loc[onset_mask_df['Year'] == yr + start_year, 'OnsetDay'].values[0]
            for offset in range(self.max_offset):
                idx = yr*self.max_offset + offset
                self.Y[idx] = offset
                self.X[idx,:] = self.climato_tensor[yr][onset_day - self.lead_time - offset - 1: onset_day - offset - 1]

        # Plot the data for debugging purposes
        yr = 0
        plt.scatter(np.mean(self.X, axis = 1), self.Y, c=self.Y, cmap='viridis')
        plt.xlabel("Mean accumulated anomaly of lead times")
        plt.ylabel("Days to onset")
        plt.savefig("scatter.png")
        plt.show()
        exit()
       
    def __len__(self):
        return len(self.block_dict)

    def __getitem__(self, yr_off_tuple):
        # yr_off_tuple is a tuple of (year, offset)
        yr, off = yr_off_tuple

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
            


# Main
if __name__ == "__main__":
    offsets = 12
    ds = NetCDFDataset(max_offset = offsets)