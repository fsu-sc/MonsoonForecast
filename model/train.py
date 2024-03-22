import torch.optim as optim
import torch.nn as nn
import torch

import data_loader as dl
import model as mod

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split

file_info_dict = dl.createFileInfoDict("/Net/elnino/data/obs/ERA5/global/daily/")

dataset = dl.NetCDFDataset(file_info_dict, "/Net/elnino/data/obs/ERA5/global/daily/")

# Define the range of years for training and validation
train_years = range(2000, 2021)
val_years = range(1993, 2000)

# Define the range of offsets
offsets = range(31, 41)  #updated based on Dr. Misra suggestion

# Create a list of tuples containing year and offset indices
train_indices =  [(year, offset) for year in train_years for offset in offsets]
val_indices = [(year, offset) for year in val_years for offset in offsets]


# Create samplers for training and validation sets
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)


# Create data loaders for training and validation sets
train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)


#-----------------------------

model = mod.EnhancedCNN()

# Define your loss function (Mean Squared Error for regression)
criterion = nn.MSELoss()

# Define your optimizer (Adam optimizer with learning rate 0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs
epochs = 100
mc_samples = 10

# Lists to store training and validation losses
train_losses = []
val_losses = []

# Training loop
for epoch in range(epochs):
    # Set model to training mode
    model.train()
    
    # Initialize lists to store predicted outputs and true values for training
    train_predicted_outputs = []
    train_true_values = []
    
    # Iterate over batches in the training data
    for inputs, targets in train_loader:
        all_outputs = torch.zeros((mc_samples, len(inputs)))
        outputs = model.outputs(inputs)

        # Append predicted outputs and true values for training
        train_predicted_outputs.extend(outputs.tolist())
        train_true_values.extend(targets.tolist())  
    
        # Compute the loss
        loss = criterion(outputs.squeeze(), targets.float())
        
        # Zero gradients, backward pass, and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Compute training loss
    train_loss = criterion(torch.tensor(train_predicted_outputs), torch.tensor(train_true_values))
    train_losses.append(train_loss.item())

    # Print training loss
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {train_loss.item()}')
        
    # Validation
    if (epoch + 1) % 10 == 0:
        # Set model to evaluation mode
        model.eval()
        
        # Initialize lists to store predicted outputs and true values for validation
        val_predicted_outputs = []
        val_true_values = []

        # Iterate over batches in the validation data
        for val_inputs, val_targets in val_loader:
            outputs = model(val_inputs)
            
            # Append predicted outputs and true values for validation
            val_predicted_outputs.extend(outputs.tolist())
            val_true_values.extend(val_targets.tolist())

        # Compute validation loss
        val_loss = criterion(torch.tensor(val_predicted_outputs), torch.tensor(val_true_values))
        val_losses.append(val_loss.item())
        
        # Print validation loss
        print(f'Validation Epoch [{epoch + 1}/{epochs}], Loss: {val_loss.item()}')