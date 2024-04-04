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
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
# import model.ViTmodel as ViT
import torchvision.models.vision_transformer as ViT
import torchvision.models as models
ViT_model = models.vit_b_16(weights=None)
# onset_mask_df = pd.read_csv("/unity/f2/aoleksy/MonsoonForecast/onset_pen_FL.csv", names=['Year','OnsetDay'])
# filepath = "/Net/elnino/data/obs/ERA5/global/daily/"
# dataset =  dl.NetCDFDataset(filepath,onset_mask_df)
# print(dataset.file_keys())
  
data_dir = "/Net/elnino/data/obs/ERA5/global/daily/"
year = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]
offsets =[1,2,3,4,5,6,7,8,9,10,11,12]
# file_dic = ds.createFileInfoDict(data_dir)
variables = ['tp', 'mslp', 't2m', 'u200', 'u850', 'v200', 'v850']
# variables = variables[0:]
test_year = [1990,1991,1992,1993,1994,1995,1996,1997,1998,1999]
train_data = ds.NetCDFDataset(data_dir,year,offsets)

test_data = ds.NetCDFDataset(data_dir,test_year,offsets)
# val_data = ds.NetCDFDataset(data_dir,[2],[12])

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

#-------------ViT Model----------------
# import torch
# import torch.nn as nn

# # Define the Mean Squared Error (MSE) loss function
# criterion = nn.MSELoss()
# batch_size = 32
# num_epochs= 100
# optimizer = optim.Adam(ViT_model.parameters(), lr=0.05)
# # Training loop
# for epoch in range(num_epochs):
#     # Shuffle the dataset if needed
#     train_data.shuffle()
    
#     # Iterate over batches directly from your custom dataset
#     for x_batch, y_batch in train_data.batch(batch_size):
#         # Reshape data for input into ViT model
#         batch_size, num_channels, depth, height, width = x_batch.shape
#         x_reshaped = x_batch.view(batch_size * depth, num_channels, height, width)
        
#         # Normalize data
#         # x_normalized = ds.normalize_tensor(x_reshaped)
        
#         # Forward pass through ViT model
#         outputs = ViT_model(x_reshaped)
        
#         # Compute loss and perform backpropagation
#         loss = criterion(outputs, y_batch)  # Calculate MSE loss
#         loss.backward()
#         optimizer.step()

#         # Print loss or other metrics
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# # Optionally, save the trained model
# torch.save(ViT_model.state_dict(), 'vit_model.pth')




model = ECNN.EnhancedCNN(in_channels=6)
writer = SummaryWriter('logs')

# model = MnistModel()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Assuming 'model' is defined somewhere above in your code
# Wrap the model with DataParallel if multiple GPUs are availableif torch.cuda.device_count() > 1:
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
epochs = 1000
mc_samples = 10
batch_size = 8
patience = 25  # Number of epochs to wait for improvement
min_delta = 0.001  # Minimum change in validation loss to qualify as improvement
best_val_loss = float('inf')
epochs_no_improve = 0
# Initialize lists to store the predicted outputs and true values for later analysis
predicted_outputs = []
true_values = []

# Create DataLoader for training data
# train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

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
    count = 0
    for inputs, targets in train_loader:
        count+=1
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero gradients, perform a forward pass with dropout enabled, and compute loss
        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1)  # Dropout enabled during training
        loss = criterion(outputs, targets)
        
        # Perform Monte Carlo sampling during training
        # for _ in range(mc_samples):
        #     outputs = model(inputs)  # Dropout enabled during training
        #     loss += criterion(outputs, targets)
        
        # Backpropagation and weight update
        # loss /= (mc_samples + 1)  # Average loss over Monte Carlo samples
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(count)
    train_losses.append(running_loss / len(train_loader))

    # Print average training loss per epoch
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}]- Train Loss: {(running_loss / len(train_loader)):.5f}', end=', ')
        writer.add_scalar('Loss/Train', running_loss / len(train_loader), epoch)
    # Validation
    if (epoch + 1) % 10 == 0:
        model.eval()  # Set model to evaluation mode
        val_running_loss = 0.0

        with torch.no_grad():  # No need to track gradients for validation
            for inputs, targets in val_loader:
                # targets = targets.unsqueeze(1)
                # Add an extra dimension for the model
                inputs, targets = inputs.to(device), targets.to(device)

                # Compute model predictions
                val_outputs = model(inputs).squeeze()
                
                # Compute loss for this batch
                val_loss = criterion(val_outputs, targets)
                val_running_loss += val_loss.item()
       
        val_losses.append(val_running_loss / len(val_loader))
        if val_loss + min_delta < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
                print('Early stopping!')
                break
        # Print average validation loss per epoch
        print(f'Validation Loss: {(val_running_loss / len(val_loader)):.5f}')
        writer.add_scalar('Loss/Validation', val_running_loss / len(val_loader), epoch)
        # print()
writer.close()

# Save the model state dictionary along with other necessary information
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_losses,
    'val_loss': val_losses
}, 'monsoon_offset_trainer.pth')

# At this point, 'predicted_outputs' and 'true_values' contain the predictions and true values for the training set

# Plotting the training and validation loss
# plt.figure()
# plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
# plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()

#------Model-Evaluation-------
# model.eval()
# test_loader = DataLoader(val_data, batch_size=batch_size)
# test_loss = 0.0

# with torch.no_grad():
#     for inputs, targets in test_loader:
#         inputs, targets = inputs.to(device), targets.to(device)
#         outputs = model(inputs).squeeze()
#         test_loss += criterion(outputs, targets).item()
        
#         # Extracting and printing predicted values for each batch
#         predicted_values = outputs.tolist()  # Convert tensor to list
#         print("Predicted Values:", predicted_values)
        
# # Calculate the average test loss over all batches
# average_test_loss = test_loss / len(test_loader.dataset)
# print("Average Test Loss:", average_test_loss)
