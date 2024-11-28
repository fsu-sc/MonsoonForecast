# %%
from model import NNmodel as nModel
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from data_loader import dataset_Block as dB
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

# %% Model setup

max_offset = 12
lead_time = 10
data = dB.NetCDFDataset(max_offset = max_offset, lead_time = lead_time, start_year = 1979, end_year = 2017)
test_data = dB.NetCDFDataset(max_offset = max_offset, lead_time = lead_time, start_year = 2018, end_year = 2022)
#data_tensors = tuple(t.float() for t in data.tensors)
#dataset = Dataset(data_tensors)
train_dataset, val_dataset = random_split(data, [0.8, 0.2])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
model = nModel.DenseModel(input_size=lead_time, hidden_size=100, num_layers=5, dropout_rate=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion2 = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize lists to store training and validation losses
train_losses = []
val_losses = []

all_val_pairs = []

# %% Training loop
num_epochs = 50
for epoch in range(num_epochs):
    # Set model to training mode
    model.train()
    train_loss = 0
    # Iterate over batches in the training data
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        targets = targets.unsqueeze(-1)
        
        # Compute the loss
        optimizer.zero_grad()
        loss = criterion2(outputs, targets)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)  # Multiply by batch size
    
    # Calculate average training loss
    train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.6f}')
    
    # Validation
    model.eval()
    val_loss = 0
    val_predictions = []
    val_targets_all = []
    
    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            val_outputs = model(val_inputs)
            val_targets = val_targets.unsqueeze(-1)
            
            # Store predictions and targets
            all_val_pairs.extend(list(zip(val_outputs.squeeze().tolist(), val_targets.squeeze().tolist())))
            
            # Compute validation loss
            batch_loss = criterion2(val_outputs, val_targets)
            val_loss += batch_loss.item() * val_inputs.size(0)
        
        # Calculate average validation loss
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f'Validation Epoch [{epoch + 1}/{num_epochs}], Loss: {val_loss:.6f}')

# %% Test
model.eval()
test_loss = 0
with torch.no_grad():
    for test_inputs, test_targets in test_loader:
        test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
        test_outputs = model(test_inputs)
        test_targets = test_targets.unsqueeze(-1)
        test_loss += criterion2(test_outputs, test_targets).item()
    print(f"Test Loss: {test_loss/len(test_loader)}")
    

        
# %%
# Combine predictions and targets into pairs
prediction_target_pairs = all_val_pairs
#prediction_target_pairs = list(zip(train_predicted_outputs, train_true_values))

# Print the first few prediction-target pairs
print("Prediction-Target Pairs:")
print("Length: ", len(prediction_target_pairs))
for i in range(5):
    print(prediction_target_pairs[i])


# Plot loss vs epoch
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
# plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.grid(True)
plt.savefig("loss_jv24b.png")
#plt.show()



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
plt.savefig("val_mses_jv24b.png")
#plt.show()
