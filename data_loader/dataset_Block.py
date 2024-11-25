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
root_folder = "/Net/work/ozavala/CODE/MonsoonForecast/"
local_folder = "/home/jv24b/github/MonsoonForecast"
onset_mask_df = pd.read_csv(join(root_folder, "onset_pen_FL.csv"), names=['Year','OnsetDay'])

class NetCDFDataset(Dataset):
    
    def __init__(self, max_offset = 12, lead_time = 10, start_year = 1979, end_year = 2022):
        print("Reading data....")
        self.onset_mask_df = pd.read_csv(join(root_folder, "onset_pen_FL.csv"), names=['Year', 'OnsetDay'])
        self.climato_tensor = torch.load(join(local_folder, "climato_tensor.pt"), weights_only=False)
        # Subset the climato_tensor to the range of years
        self.climato_tensor = self.climato_tensor[start_year-1979:end_year-1979+1, :]
        self.onset_mask_df = self.onset_mask_df[self.onset_mask_df['Year'].isin(range(start_year, end_year+1))]
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
                self.X[idx,:-1] = np.diff(self.climato_tensor[yr][onset_day - self.lead_time - offset - 1: onset_day - offset - 1])
                #self.X[idx,:] = self.climato_tensor[yr][onset_day - self.lead_time - offset - 1: onset_day - offset - 1]
       
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)

# %% Main
if __name__ == "__main__":
    offsets = 12
    ds = NetCDFDataset(max_offset = offsets, start_year = 1979, end_year = 2017)

    # %%

    x, y = ds[:]
    print(f"Total number of samples: {len(ds)} Shape of x: {x.shape} Shape of y: {y.shape}")

    # Plot the data for debugging purposes
    #plt.scatter(np.mean(x, axis = 1), y, 
                #c=np.repeat(np.arange(ds.total_years), offsets), cmap='rainbow')
    #plt.xlabel("Mean accumulated anomaly of lead times")
    #plt.ylabel("Days to onset")
    #plt.savefig("scatter.png")
    #plt.show()
# %%
