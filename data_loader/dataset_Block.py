# %%
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.transforms import Resize, Compose, ToTensor
from torchvision.transforms.functional import to_pil_image, to_tensor
#from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
from os.path import join

from datetime import datetime, timedelta

class NetCDFDataset(Dataset):
    """
    A PyTorch Dataset for loading and preprocessing climate data for monsoon onset prediction.
    
    This dataset processes NetCDF climate data and onset dates to create input-output pairs
    for training a model to predict the number of days until monsoon onset.

    Parameters
    ----------
    max_offset : int, default=12
        Maximum number of days before onset to consider for prediction
    lead_time : int, default=10 
        Number of days of climate data to use as input for each prediction
    start_year : int, default=1979
        First year of data to include
    end_year : int, default=2022
        Last year of data to include
    onset_mask_file_name : str, optional
        Path to CSV file containing onset dates for each year
    climato_tensor_file_name : str, optional
        Path to .pt file containing climate tensor data
    fields : list of str, default=['tp', 't2m']
        Climate variables to use as input features
    num_examples_per_year : int, default=10
        Number of training examples to generate per year
    random_offset : bool, default=True
        If True, randomly sample offsets before onset. If False, use sequential offsets.

    The dataset generates input tensors X with shape (N, lead_time*num_fields + 1) where:
    - N = total_years * num_examples_per_year
    - Each sample contains lead_time days of data for each field concatenated
    - The final column contains the day of year

    The target tensor Y has shape (N,) containing the number of days until onset.

    Input features are normalized by subtracting the minimum value for each field/sample.
    """
    
    def __init__(self, max_offset = 12, lead_time = 10, start_year = 1979, end_year = 2022,
                 onset_mask_file_name = None, climato_tensor_file_name = None, 
                 fields = ['tp', 't2m'], num_examples_per_year = 10, random_offset = True):


        print("Reading data....")
        self.onset_mask_df = pd.read_csv(onset_mask_file_name, names=['Year', 'OnsetDay'])
        self.climato_tensor = torch.load(climato_tensor_file_name, weights_only=False)
        tot_fields = len(fields)

        # Subset the climato_tensor to the range of requrested years (it assumes the climato tensor starts at 1979)
        # Climate tensor has shape (fields, total years, days in year)
        self.climato_tensor = self.climato_tensor[:, start_year-1979:end_year-1979+1, :]
        # Subset the onset mask to the range of requested years
        self.onset_mask_df = self.onset_mask_df[self.onset_mask_df['Year'].isin(range(start_year, end_year+1))]
        self.total_years = end_year - start_year + 1
        self.max_offset = max_offset
        self.lead_time = lead_time
        print("Done!")
        # Generate a dataset that contains all the years and adjusted offsets
        self.X = np.zeros((self.total_years*num_examples_per_year, self.lead_time*tot_fields*2+1)) # +1 for the day of the year
        self.Y = np.zeros(self.total_years*num_examples_per_year)
        for yr in range(self.total_years):
            onset_day = self.onset_mask_df.loc[self.onset_mask_df['Year'] == yr + start_year, 'OnsetDay'].values[0]

            # Generate num_examples_per_year examples for each year randomly from 0 to max_offset
            if random_offset:
                offsets = np.random.choice(range(self.max_offset), num_examples_per_year, replace=False)
            else:
                offsets = np.arange(num_examples_per_year)
            print(f"Offsets for year {yr}: {offsets}")
            for i, offset in enumerate(offsets):
                # Each example will have a lead time of X days and the number of fields all concatenated
                # This means that in case we have 3 fields and 10 lead times the first
                # 30 inputs will be the 10 lead times of the three fields concatenated
                idx = yr*num_examples_per_year + i
                self.Y[idx] = offset
                # For each example we will remove the minimum value
                each_field_inputs = []
                each_field_inputs_deriv = []
                for field in range(tot_fields):
                    min_val = self.climato_tensor[field][yr][onset_day - self.lead_time - offset - 1: onset_day - offset - 1] .min()
                    max_val = self.climato_tensor[field][yr][onset_day - self.lead_time - offset - 1: onset_day - offset - 1] .max()
                    range_val = max_val - min_val
                    # Normal input values
                    cur_values = (self.climato_tensor[field][yr][onset_day - self.lead_time - offset - 1: onset_day - offset - 1])
                    each_field_inputs.append(cur_values)
                    # Derivative of current values (fill the first value with 0)
                    derv_values = np.diff(cur_values)
                    derv_values = np.insert(derv_values, 0, 0)
                    each_field_inputs_deriv.append(derv_values)

                self.X[idx,0:self.lead_time*tot_fields] = np.concatenate(each_field_inputs)
                self.X[idx,self.lead_time*tot_fields:self.lead_time*tot_fields*2] = np.concatenate(each_field_inputs_deriv)
                # Compute the derivative 
                self.X[idx,-1] = onset_day - offset

        print(f"X shape: {self.X.shape} Y shape: {self.Y.shape}")
       
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)

# %% Main
if __name__ == "__main__":

    onset_mask_file_name = "/Net/work/ozavala/CODE/MonsoonForecast/onset_pen_FL.csv"
    # climato_tensor_file_name = "/Net/work/ozavala/CODE/MonsoonForecast/year_cum_tp_anomaly.pt"
    climato_tensor_file_name = "/Net/work/ozavala/CODE/MonsoonForecast/year_cum_tp_anomaly_normalized.pt"

    output_imgs_folder = "/Net/work/ozavala/CODE/MonsoonForecast/imgs/"

    max_offset = 30 # How many offsets to use, from 0 to max_offset
    lead_time = 20 # How many input days to use
    start_year = 1979
    end_year = 2017
    fields = ['tp', 't2m', 'u200']
    num_examples_per_year = 10
    random_offset = True
    ds = NetCDFDataset(max_offset = max_offset, lead_time = lead_time, start_year = start_year, end_year = end_year, 
                       onset_mask_file_name = onset_mask_file_name, 
                       climato_tensor_file_name = climato_tensor_file_name, 
                       fields = fields, num_examples_per_year = num_examples_per_year,
                       random_offset = random_offset)

    # %%
    x, y = ds[:]
    print(f"Total number of samples: {len(ds)} from ({start_year} - {end_year}) = {ds.total_years}*{num_examples_per_year}(offsets)")
    print(f"Shape of x: {x.shape} (total years* offset, lead_time) Shape of y: {y.shape} ({ds.total_years}*{num_examples_per_year})")

    # Define vmin and vmax for the scatter plot for each field
    vmin = {'tp': -.00005, 't2m': 0, 'u200': 0}
    vmax = {'tp': .00005, 't2m': 1.1, 'u200': 1.1}

    # Plot the data for debugging purpose
    num_years = ds.total_years # Number of years to plot
    # Make a figure wit 1 row and len(fields) columns
    # for cur_year in range(num_years):
    for cur_year in range(3):
        fig, axs = plt.subplots(3, len(fields), figsize=(7*len(fields),10))
        # Define the indices for the current year
        start_idx = cur_year*num_examples_per_year
        end_idx = (cur_year+1)*num_examples_per_year
        # Get the true offset for the current year
        true_offset = y[start_idx:end_idx]
        input_curr_day = x[start_idx:end_idx,-1]
        
        tot_fields = len(fields)

        # Plot the input values
        for cur_field, field in enumerate(fields):
            offsets = y[start_idx:end_idx].numpy().astype(int)
            # Sort the data by offset
            sort_idx = np.argsort(offsets)
            offsets = offsets[sort_idx]
            x_sorted = x[start_idx:end_idx][sort_idx]
            
            print(f"Offsets for year {cur_year}: {offsets}")
            axs[0, cur_field].scatter(x_sorted[:,cur_field*lead_time:(cur_field+1)*lead_time].mean(axis = 1), 
                                    offsets)
            # axs[0, cur_field].set_xlim(vmin[field], vmax[field])
            axs[0, cur_field].set_xlabel(f"Mean accumulated anomaly of lead times for {field}")
            axs[0, cur_field].set_ylabel("Days to onset")
            # Plot inputs and outputs
            for i, cur_offset in enumerate(offsets[:5]):
                axs[1, cur_field].scatter(range(lead_time), 
                                          x_sorted[i,cur_field*lead_time:(cur_field+1)*lead_time],
                                          label = f"{cur_offset}")
            # axs[1, cur_field].set_ylim(vmin[field], vmax[field])
            axs[1, cur_field].legend()
            # Set axis labels for the last plot
            axs[1, cur_field].set_xlabel("Offset")
            axs[1, cur_field].set_ylabel(f"Anomaly of {field}")
            # Plot the derivative
            for i, cur_offset in enumerate(offsets[:4]):
                axs[2, cur_field].scatter(range(lead_time), 
                                        x_sorted[i,tot_fields*lead_time+cur_field*lead_time:tot_fields*lead_time+(cur_field+1)*lead_time],
                                        label = f"{cur_offset}")
            axs[2, cur_field].legend()
            axs[2, cur_field].set_xlabel("Offset")
            axs[2, cur_field].set_ylabel(f"Derivative of {field}")


        plt.suptitle(f"Onset vs mean accumulated anomaly of lead times for {start_year + cur_year} \n (current offset: {input_curr_day}) \n (target offset: {true_offset})")
        plt.savefig(join(output_imgs_folder, f"scatter_{start_year + cur_year}.png"))
        plt.show()
# %%
