import xarray as xr
#import netCDF4 as nc
import os
import data_loader.dataset as ds 
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import torch
import model.data_loader as dl
import model.NNmodel as ECNN
import pandas as pd
# onset_mask_df = pd.read_csv("/unity/f2/aoleksy/MonsoonForecast/onset_pen_FL.csv", names=['Year','OnsetDay'])
# filepath = "/Net/elnino/data/obs/ERA5/global/daily/"
# dataset =  dl.NetCDFDataset(filepath,onset_mask_df)
# print(dataset.file_keys())
  
data_dir = "/Net/elnino/data/obs/ERA5/global/daily/"
year = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010]
offsets =[1,2,3,4,5,6,7,8,9,10,11,12]
# file_dic = ds.createFileInfoDict(data_dir)
variables = ['tp', 'mslp', 't2m', 'u200', 'u850', 'v200', 'v850']
test_year = [1990,1991,1992,1993,1994,1995,1996,1997,1998,1999]
train_data = ds.NetCDFDataset(data_dir,year,offsets)
test_data = ds.NetCDFDataset(data_dir,test_year,offsets)

# tensors = []
# val_tensors = []
# for year in range(2001,2010):
#     for offst in range(1,10):
#         tensors.append(dataset.__getitem__(year, offst,var=3))

# for year in range(1990,1999):
#     for offst in range(1,10):
#         val_tensors.append(dataset.__getitem__(year, offst,var=3))

# print("Tensor loading complete")

# def get_dimensions(lst):
#     if isinstance(lst, list):
#         return [len(lst)] + get_dimensions(lst[0])
#     else:
#         return []
    
    
    
# Given 'tensors', a list of 200 lists, each containing a tensor and a number
# Initialize empty lists for tensors and numbers
# x_train = []
# y_train = []

# x_val = []
# y_val = []

# # Iterate over each inner list
# for tensor, number in tensors:
#     # Append the tensor to x_train
#     x_train.append(tensor)
#     # Append the number to y_train
#     y_train.append(float(number))

# # Iterate over each inner list
# for tensor, number in val_tensors:
#     # Append the tensor to x_train
#     x_val.append(tensor)
#     # Append the number to y_train
#     y_val.append(float(number))

# # Convert lists to tensors if needed
# x_train = torch.stack(x_train)
# y_train = torch.tensor(y_train)
# x_val = torch.stack(x_val)
# y_val = torch.tensor(y_val)

# print("x_train shape:", x_train.shape)
# print("y train shape:", y_train.shape)
# print("y_train[2] type:", y_train[2].type)
# print(type(y_train[2].item()))


model = ECNN.EnhancedCNN(in_channels=len(variables))
# model = MnistModel()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Assuming 'model' is defined somewhere above in your code
# Wrap the model with DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define your loss function
criterion = nn.MSELoss()

# Define your optimizer
optimizer = optim.Adam(model.parameters(), lr=0.05)

# Number of epochs
epochs = 400
mc_samples = 100
batch_size = 32

# Initialize lists to store the predicted outputs and true values for later analysis
predicted_outputs = []
true_values = []

# Create DataLoader for training data
# train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# # Create DataLoader for validation data
# val_dataset = TensorDataset(x_val, y_val)
val_loader = DataLoader(test_data, batch_size=batch_size)
train_losses = []
val_losses = []
# Training loop
for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    # For each batch in the training data
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Initialize a tensor to store the outputs for each Monte Carlo sample
        all_outputs = torch.zeros((mc_samples, len(inputs)), device=device)

       # Monte Carlo sampling
        for i in range(mc_samples):
            outputs = model(inputs).squeeze()
            all_outputs[i] = outputs
        
        # Compute mean prediction and loss
        outputs = model(inputs).squeeze()
        # mean_prediction = all_outputs.mean(dim=0)
        loss = criterion(outputs, targets)
        
        # Zero gradients, perform a backward pass, and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    # Print average training loss per epoch
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}]- Train Loss: {(running_loss / len(train_loader)):.5f}', end=', ')

    # Validation
    if (epoch + 1) % 10 == 0:
        model.eval()  # Set model to evaluation mode
        val_running_loss = 0.0

        with torch.no_grad():  # No need to track gradients for validation
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Initialize a tensor to store the outputs for each Monte Carlo sample
                all_val_outputs = torch.zeros((mc_samples, len(inputs)), device=device)

                # Monte Carlo sampling for validation
                for i in range(mc_samples):
                    val_outputs = model(inputs).squeeze()
                    all_val_outputs[i] = val_outputs

                # Compute mean prediction and loss for validation
                mean_val_prediction = all_val_outputs.mean(dim=0)
                val_loss = criterion(mean_val_prediction, targets)
                val_running_loss += val_loss.item()
        val_losses.append(val_running_loss / len(val_loader))
        # Print average validation loss per epoch
        print(f'Validation Loss: {(val_running_loss / len(val_loader)):.5f}')
        # print()

# At this point, 'predicted_outputs' and 'true_values' contain the predictions and true values for the training set

# Plotting the training and validation loss
plt.figure()
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()