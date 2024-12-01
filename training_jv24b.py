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
from os.path import join
# %% Model setup
max_offset = 60 # How far in the future to predict 
lead_time = 20  # How many input days to use
num_examples_per_year = 50
# fields = ['tp', 't2m', 'u200']
fields = ['tp', 't2m', 'u200']
tot_fields = len(fields)
include_gradient = False

onset_mask_file_name = "/Net/work/ozavala/CODE/MonsoonForecast/onset_pen_FL.csv"
climato_tensor_file_name = "/Net/work/ozavala/CODE/MonsoonForecast/year_cum_tp_anomaly.pt"
imgs_folder = "/Net/work/ozavala/CODE/MonsoonForecast/imgs/"

start_year = 1979
end_year = 2017
test_start_year = 2018
test_end_year = 2022

random_offset = True
data = dB.NetCDFDataset(max_offset = max_offset, lead_time = lead_time, start_year = start_year, end_year = end_year, 
                    onset_mask_file_name = onset_mask_file_name, 
                    climato_tensor_file_name = climato_tensor_file_name, fields = fields, 
                    num_examples_per_year = num_examples_per_year, random_offset = random_offset, include_gradient = include_gradient)
random_offset = False
test_data = dB.NetCDFDataset(max_offset = max_offset, lead_time = lead_time, start_year = test_start_year, end_year = test_end_year, 
                    onset_mask_file_name = onset_mask_file_name, 
                    climato_tensor_file_name = climato_tensor_file_name, fields = fields, 
                    num_examples_per_year = max_offset, random_offset = random_offset, include_gradient = include_gradient)

# Calculate split indices based on years
train_percentage = 0.90
val_percentage = 1 - train_percentage
total_samples = len(data)
train_size = int(train_percentage * total_samples)
train_dataset = torch.utils.data.Subset(data, range(train_size))
val_dataset = torch.utils.data.Subset(data, range(train_size, total_samples))

years_range = end_year - start_year + 1
print(f"Total samples in training (years:{years_range*train_percentage:.2f}) validation (years:{years_range*val_percentage:.2f}) and test (years:{test_end_year - test_start_year + 1:.2f})")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size= max_offset, shuffle=False)
# The plus one is for the 'current' day of the year
if include_gradient:
    input_size = lead_time*tot_fields*2+1
else:
    input_size = lead_time*tot_fields+1
model = nModel.DenseModel(input_size=input_size, hidden_size=input_size*2, num_layers=10, dropout_rate=0.2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        
    def forward(self, pred, target):
        # For now, equivalent to MSE
        # Add larger weight when target is small by dividing by target+epsilon
        # Using epsilon=0.1 to avoid division by very small numbers
        epsilon = 0.1
        return torch.mean((pred - target) ** 2 / (target/10 + epsilon))

# custom_loss = CustomLoss()
custom_loss = nn.MSELoss()
mse_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize lists to store training and validation losses
train_losses = []
val_losses = []
all_val_pairs = []

# %% Training loop
max_epochs = 1000
patience = 51
patience_counter = 0
min_val_loss = float('inf')
min_val_loss_epoch = 0
best_model = None
for epoch in range(max_epochs):
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
        loss = custom_loss(outputs, targets)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)  # Multiply by batch size
    
    # Calculate average training loss
    train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    print(f'Epoch [{epoch + 1}/{max_epochs}], Training Loss: {train_loss:.6f}')
    
    # Validation
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            val_outputs = model(val_inputs)
            val_targets = val_targets.unsqueeze(-1)
            
            # Store predictions and targets
            all_val_pairs.extend(list(zip(val_outputs.squeeze().tolist(), val_targets.squeeze().tolist())))
            
            # Compute validation loss
            batch_loss = mse_loss(val_outputs, val_targets)
            val_loss += batch_loss.item() * val_inputs.size(0)
        
        # Calculate average validation loss
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f'Validation Epoch [{epoch + 1}/{max_epochs}], Loss: {val_loss:.6f}')
    
    # Check if the validation loss is the lowest
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        min_val_loss_epoch = epoch
        patience_counter = 0
        best_model = model.state_dict()
    else:
        patience_counter += 1
    
    # If the patience is reached, stop the training
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

    if epoch % 20 == 0:
        # Plot the training and validation losses
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Losses Epoch {epoch} Min val loss: {min_val_loss:.2f} at epoch {min_val_loss_epoch}')
        plt.legend()
        plt.grid(True)
        plt.show()
        # plt.savefig(join(imgs_folder, f"loss_jv24b_{epoch}.png"))
        # Wait
        # plt.pause(0.1)

print(f"Stopped at epoch {epoch} with min val loss {min_val_loss:.6f} at epoch {min_val_loss_epoch}")
# %% Test
model.load_state_dict(best_model)
model.eval()
test_loss = 0
rmse_per_offset = {str(offset): [] for offset in range(max_offset)}
all_rmses = []
do_plot = False
with torch.no_grad():
    batch_idx = 0
    for test_inputs, test_targets in test_loader:
        print(f"Batch {batch_idx}")
        test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
        test_outputs = model(test_inputs)
        test_targets = test_targets.unsqueeze(-1)
        test_loss += mse_loss(test_outputs, test_targets).item()
        all_rmses.append(np.sqrt(custom_loss(test_outputs, test_targets).item()))
        for idx_offset, cur_offset in enumerate(range(max_offset)):
            rmse_per_offset[str(idx_offset)].append(np.sqrt(custom_loss(test_outputs[idx_offset], test_targets[idx_offset]).item()))
            print(f"Output value: {test_outputs[idx_offset].squeeze().tolist():.1f} Target value: {test_targets[idx_offset].squeeze().tolist():.1f}")
        # Plot the inputs and outputs and the targets
        examples_to_plot = num_examples_per_year
        # We will plot the inputs as scatter plots and the outputs and targets in the title
        if do_plot:
            fig, axs = plt.subplots(examples_to_plot, 3, figsize=(20, 4*examples_to_plot))
            for idx_to_plot in range(examples_to_plot):
                for cur_field in range(tot_fields):
                    axs[idx_to_plot, cur_field].scatter(range(lead_time), 
                                        test_inputs[idx_to_plot, 
                                                    cur_field*lead_time:(cur_field+1)*lead_time].squeeze().tolist(), 
                                        label = 'Inputs')
                    axs[idx_to_plot, cur_field].set_title(f"Year: {batch_idx+test_start_year} Target: {int(test_targets[idx_to_plot].squeeze().tolist())} Output: {test_outputs[idx_to_plot].squeeze().tolist():.1f}")
                    axs[idx_to_plot, cur_field].legend()
            # Show the RMSE for this batch in the title of the figure
            fig.suptitle(f"RMSE: {all_rmses[batch_idx]:.2f}", y=1.00)
            plt.tight_layout()
            plt.show()
            # Save the figure
            fig.savefig(join(imgs_folder, f"test_year_{batch_idx+test_start_year}.png"))
            plt.close(fig)
        batch_idx += 1

print(f"Test RMSE: {np.sqrt(test_loss/len(test_loader)):.4f}")
# %%
# Plot the RMSE per offset in a boxplot
fig, ax = plt.subplots(figsize=(20,5))
# Remove outliers
rmse_per_offset = {offset: [rmse for rmse in rmse_values if rmse < 50] for offset, rmse_values in rmse_per_offset.items()}
# Create scatter plot of RMSE values per offset
for offset, rmse_values in rmse_per_offset.items():
    ax.scatter([int(offset)] * len(rmse_values), rmse_values, alpha=0.5)

# ax.boxplot(list(rmse_per_offset.values()))
ax.set_xlabel("Offset")
ax.set_ylabel("RMSE")
plt.show()
# %% List all the RMSEs
for idx_to_plot, cur_rmse in enumerate(all_rmses):
    print(f"RMSE year {idx_to_plot+test_start_year}: {cur_rmse:.2f}")
# %%
