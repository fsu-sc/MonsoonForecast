import xarray as xr
#import netCDF4 as nc
import os
from datetime import datetime
import data_loader.dataset as ds 
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import torch
import model.data_loader as dl
import model.NNmodel as ECNN
import torch.optim as optim
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.ViTmodel import VitTransformer
import pandas as pd
# import model.ViTmodel as ViT
import torchvision.models.vision_transformer as ViT
import torchvision.models as models
import time





from data_loader import dataset_Block as dB


data_dir = "/Net/elnino/data/obs/ERA5/global/daily/"
year = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]
year = year[:10]
offsets =[i for i in range(1,40)]


variables = ['tp', 'mslp', 't2m', 'u200', 'u850', 'v200', 'v850']
variables = variables[:1]
channels = len(variables)
test_year = [1990,1991,1992,1993,1994,1995,1996,1997,1998,1999]


test_year = test_year[:4]

# Loading data from ERA5 dataset, using the NetCDFDataset class
train_data = dB.NetCDFDataset(data_dir,year,offsets,variables=variables)
test_data = dB.NetCDFDataset(data_dir,test_year,offsets,variables=variables)
# model = ECNN.EnhancedCNN(in_channels=16)
model = ECNN.DenseModel()
writer = SummaryWriter(f'logs/run{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')



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
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Number of epochs
epochs = 10000
mc_samples = 10
batch_size = 32
patience = 25  # Number of epochs to wait for improvement
min_delta = 0.001  # Minimum change in validation loss to qualify as improvement
best_val_loss = float('inf')
learning_rate = 0.001
epochs_no_improve = 0
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
    count = 0
    for inputs, targets in train_loader:
        count+=1
        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the average values for the full batch and plot
        # sum_by_batch = torch.sum(inputs[:,:,0,:,:], dim=(1, 2, 3))
        # # Scatter this sum vs the target
        # plt.scatter(targets.cpu().numpy(),sum_by_batch.cpu().numpy())
        # plt.xlabel('Sum of input values')
        # plt.ylabel('Target values')
        # plt.title('Sum of input values vs target values')
        # plt.savefig('sum_vs_target.png')
        # plt.show()

        # plt.close()
        # sum_anomaly_by_batch = torch.sum(inputs[:,:,1,:,:], dim=(1, 2, 3))
        # plt.scatter(targets.cpu().numpy(),sum_anomaly_by_batch.cpu().numpy())
        # plt.xlabel('Sum of input values')
        # plt.ylabel('Target values')
        # plt.title('Sum of input values vs target values')
        # plt.savefig('sumanomaly_vs_target.png')
        # plt.show()
        
        # Zero gradients, perform a forward pass with dropout enabled, and compute loss
        optimizer.zero_grad()
        # TODO just make a dense network with 2 input and 10 neurons and 4 hidden layers with BN in between. Output linear
        outputs = model(inputs).squeeze(1)  # Dropout enabled during training
        loss = criterion(outputs, targets)
        # plt.imshow(inputs[0][0][0][:][:].cpu().numpy(), cmap='jet')  # Example of a heatmap
        # # plt.colorbar(label='Temperature')
        # plt.xlabel('X Axis')
        # plt.ylabel('Y Axis')
        # plt.title('Temperature Data')
        # plt.show()
        # plt.savefig('temperature_data.png')
        # plt.close()



        # Perform Monte Carlo sampling during training
        # for _ in range(mc_samples):
        #     outputs = model(inputs)  # Dropout enabled during training
        #     loss += criterion(outputs, targets)
        
        # Backpropagation and weight update
        # loss /= (mc_samples + 1)  # Average loss over Monte Carlo samples
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
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
                # targets = targets.
                # Add an extra dimension for the model
                inputs, targets = inputs.to(device), targets.to(device)
                # plt.imshow(inputs[0][1][0][0], cmap='jet')  # Example of a heatmap
                # plt.colorbar(label='Temperature')
                # plt.xlabel('X Axis')
                # plt.ylabel('Y Axis')
                # plt.title('Temperature Data')
                # plt.show()
                # plt.savefig('temperature_data.png')
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
folder_path = "./trained_models"
os.makedirs(folder_path, exist_ok=True)
file_path = os.path.join(folder_path, f"monsoon_offset_trainer_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth")
# Save the model state dictionary along with other necessary information
filename = f"monsoon_offset_trainer_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"

# Save the model, optimizer, and other necessary information
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_losses,
    'val_loss': val_losses
}, file_path)
