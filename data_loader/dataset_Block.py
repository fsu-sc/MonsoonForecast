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
    Dataset class for the NetCDF data.
    The climato tensor should be a tensor of shape (fields, total years, 365) where each year is a 365 day long vector.
    The total shape of the input tensor should be (fields*total years*max_offset, lead_time) and the target should be a tensor of shape (total years*max_offset)
    """
    
    def __init__(self, max_offset = 12, lead_time = 10, start_year = 1979, end_year = 2022,
                 onset_mask_file_name = None, climato_tensor_file_name = None, fields = ['tp', 't2m']):
        print("Reading data....")
        self.onset_mask_df = pd.read_csv(onset_mask_file_name, names=['Year', 'OnsetDay'])
        self.climato_tensor = torch.load(climato_tensor_file_name, weights_only=False)

        # Subset the climato_tensor to the range of requrested years (it assumes the climato tensor starts at 1979)
        self.climato_tensor = self.climato_tensor[:, start_year-1979:end_year-1979+1, :]
        # Subset the onset mask to the range of requested years
        self.onset_mask_df = self.onset_mask_df[self.onset_mask_df['Year'].isin(range(start_year, end_year+1))]
        self.total_years = end_year - start_year + 1
        self.max_offset = max_offset
        self.lead_time = lead_time
        print("Done!")
        # Generate a dataset that contains all the years and adjusted offsets
        self.X = np.zeros((self.total_years*self.max_offset*len(fields), self.lead_time))
        self.Y = np.zeros(self.total_years*self.max_offset)
        for yr in range(self.total_years):
            onset_day = self.onset_mask_df.loc[self.onset_mask_df['Year'] == yr + start_year, 'OnsetDay'].values[0]

            for offset in range(self.max_offset):
                idx = yr*self.max_offset + offset
                self.Y[idx] = offset
                self.X[idx,:-1] = np.diff(self.climato_tensor[yr][onset_day - self.lead_time - offset - 1: onset_day - offset - 1])
                #self.X[idx,:] = self.climato_tensor[yr][onset_day - self.lead_time - offset - 1: onset_day - offset - 1]
       
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)

# %% Main
if __name__ == "__main__":

    onset_mask_file_name = "/Net/work/ozavala/CODE/MonsoonForecast/onset_pen_FL.csv"
    climato_tensor_file_name = "/Net/work/ozavala/CODE/MonsoonForecast/climato_tensor.pt"

    max_offset = 12
    start_year = 1979
    end_year = 2017
    lead_time = 10
    fields = ['tp', 't2m']
    ds = NetCDFDataset(max_offset = max_offset, lead_time = lead_time, start_year = start_year, end_year = end_year, 
                       onset_mask_file_name = onset_mask_file_name, 
                       climato_tensor_file_name = climato_tensor_file_name)

    # %%

    x, y = ds[:]
    print(f"Total number of samples: {len(ds)} from ({start_year} - {end_year}) = {ds.total_years}*{max_offset}(offsets)")
    print(f"Shape of x: {x.shape} (total years* offset, lead_time) Shape of y: {y.shape} ({ds.total_years}*{max_offset})")

    # Plot the data for debugging purposes
    num_years = 3# Number of years to plot
    plt.scatter(np.mean(x[0:num_years*max_offset,:].numpy(), axis = 1), y[0:num_years*max_offset].numpy(), 
                c=np.repeat(np.arange(num_years), max_offset), cmap='rainbow')
    plt.xlabel("Mean accumulated anomaly of lead times")
    plt.ylabel("Days to onset")
    plt.savefig("scatter.png")
    plt.show()

# %%
