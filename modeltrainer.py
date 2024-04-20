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

# ViT_model = models.ViTTransformer(weights=None)

# onset_mask_df = pd.read_csv("/unity/f2/aoleksy/MonsoonForecast/onset_pen_FL.csv", names=['Year','OnsetDay'])
# filepath = "/Net/elnino/data/obs/ERA5/global/daily/"
# dataset =  dl.NetCDFDataset(filepath,onset_mask_df)
# print(dataset.file_keys())

data_dir = "/Net/elnino/data/obs/ERA5/global/daily/"
year = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]
year = year[:10]
offsets =[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
# year = year[:10]
# file_dic = ds.createFileInfoDict(data_dir)
variables = ['tp', 'mslp', 't2m', 'u200', 'u850', 'v200', 'v850']

variables = variables[:1]
channels = len(variables)
test_year = [1990,1991,1992,1993,1994,1995,1996,1997,1998,1999]

# test_year = [2002,2004]
test_year = test_year[:2]
train_data = ds.NetCDFDataset(data_dir,year,offsets,variables=variables)

test_data = ds.NetCDFDataset(data_dir,test_year,offsets,variables=variables)
# val_data = ds.NetCDFDataset(data_dir,[2],[12])
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



#-------------ViT Model----------------
# import torch
# import torch.nn as nn
model = VitTransformer(
    num_encoders=6,  # Example value
    latent_size=768,  # Example value
    device=device,
    num_heads=12,  # Example value
    num_classes=1,  # Example value, adjust based on your task
    dropout=0.1,  # Example value
    patch_size=16,  # Example value, ensure it's suitable for your data # Since D=7 in your data
    n_channels=6  # Since C=6 in your data
).to(device)
# # Define the Mean Squared Error (MSE) loss function
criterion = nn.MSELoss()
batch_size = 32
num_epochs= 100
criterion = nn.CrossEntropyLoss()  # Adjust based on your task
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Example parameters
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# # Create DataLoader for validation data
# val_dataset = TensorDataset(x_val, y_val)
val_loader = DataLoader(test_data, batch_size=batch_size)
# Example training loop
for epoch in range(num_epochs):  # num_epochs should be defined
    for batch in train_loader:
        inputs, labels = batch  # Adjust based on how your data is structured
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# # Optionally, save the trained model
# torch.save(ViT_model.state_dict(), 'vit_model.pth')
model = ECNN.EnhancedCNN(in_channels=channels)
# checkpoint = torch.load('trained_models/monsoon_offset_trainer_2024-04-05_13-50-39.pth')

# If the model was trained with DataParallel, remove 'module.' from the keys
# new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}

# Load the model state dictionary
# model.load_state_dict(new_state_dict)

model.eval()
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch  # Adjust based on your data's structure
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        # Calculate and print your test metr

writer = SummaryWriter(f'logs/run{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

# model = MnistModel()


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
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Number of epochs
epochs = 10000
mc_samples = 10
batch_size = 32
patience = 25  # Number of epochs to wait for improvement
min_delta = 0.001  # Minimum change in validation loss to qualify as improvement
best_val_loss = float('inf')
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
