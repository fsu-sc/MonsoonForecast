from model import NNmodel as nModel
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from data_loader import dataset_Block as dB
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np

data_dir = "/Net/elnino/data/obs/ERA5/global/daily/"
train_year = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009]
test_year = [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]
offsets = range(1,13) # (1,13)

train_ds_indices =  [(yr, off) for yr in train_year for off in offsets]
test_ds_indices = [(yr, off) for yr in test_year for off in offsets]

train_data = dB.NetCDFDataset(data_dir,train_year,offsets,variables=["tp"])
val_data = dB.NetCDFDataset(data_dir,test_year,offsets,variables=["tp"])

     
train_ds_sampler = SubsetRandomSampler(train_ds_indices)
test_ds_sampler = SubsetRandomSampler(test_ds_indices)
train_loader = DataLoader(train_data, batch_size=32, sampler=train_ds_sampler)
val_loader = DataLoader(val_data, batch_size=32, sampler=test_ds_sampler)


model = nModel.NeuralNetwork()
criterion = nn.CrossEntropyLoss()  # Since targets are integers, use CrossEntropyLoss
criterion2 = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize lists to store training and validation losses
train_losses = []
val_losses = []

all_val_pairs = []

# Training loop
num_epochs = 250
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for inputs, targets in train_loader:
        targets = targets - 1
        optimizer.zero_grad()
        outputs = model(inputs)

        #print("Outputs shape:", outputs.shape)
        #print("Targets shape:", targets.shape)

        loss = criterion(outputs, targets.squeeze().long())  # Squeeze to remove extra dim, cast targets to long
        loss.backward()
        optimizer.step()

    # Store training loss
    train_losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        # Validation phase
        model.eval()
        val_loss_epoch = 0.0  # Accumulator for epoch's validation loss
        with torch.no_grad():
            all_predictions = []
            all_targets = []
            for inputs, targets in val_loader:
                targets = targets - 1
                outputs = model(inputs)
                
                #targets = targets.unsqueeze(1).expand(-1, outputs.shape[1])

                #print("Outputs shape:", outputs.shape)
                #print("Targets shape:", targets.shape)
                val_loss_batch = criterion(outputs, targets.squeeze().long())
                val_loss_epoch += val_loss_batch.item()  # Accumulate validation loss per batch

                # Store predictions and targets for later analysis
                predictions = torch.argmax(outputs, dim=1) + 1  # Convert class indices back to original target values
                all_predictions.extend(predictions.tolist())
                all_targets.extend(targets.tolist())

        # Calculate average validation loss for the epoch and store it
        val_loss_epoch /= len(val_loader)
        val_losses.append(val_loss_epoch)

    # Print the training and validation losses every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_loss_epoch:.4f}')
    all_val_pairs.append((all_predictions, all_targets))

# Combine predictions and targets into pairs
prediction_target_pairs = list(zip(all_predictions, all_targets))

# Print the first few prediction-target pairs
print("Prediction-Target Pairs:")
print("Length: ", len(prediction_target_pairs))
for i in range(5):
    print(prediction_target_pairs[i])


# Plot loss vs epoch
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.grid(True)
plt.show()



val_mses = []
for i in range(len(all_val_pairs)):
    # Extract the numbers from the pairs
    numbers1 = all_val_pairs[i][0]
    numbers2 = all_val_pairs[i][1]

    # Convert the numbers to numpy arrays for easy computation
    numbers1 = np.array(numbers1)
    numbers2 = np.array(numbers2)

    # Calculate the squared differences
    squared_diff = (numbers1 - numbers2) ** 2

    # Compute the mean squared error
    mse = np.mean(squared_diff)
    val_mses.append(mse)

    indices = range(1, len(val_mses) + 1)


# Plot the values with a line plot
plt.plot(indices, val_mses, color='blue', linestyle='-')

# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Val MSE Loss vs. Epoch')

# Display the plot
plt.grid(True)
plt.show()