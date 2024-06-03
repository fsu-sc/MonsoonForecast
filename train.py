import argparse
import collections
import torch
import numpy as np
import data_loader.data_loader as dl
import model.loss as module_loss
import model.metric as module_metric
import model.model as mod
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import importlib
import torch.optim as optim
import torch.nn as nn
################# Loading value from config.json #######################

optimizers = {
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD,
    'RMSprop': torch.optim.RMSprop,
    # Add more optimizers as needed
}



config = ConfigParser('config.json')
arch = config['arch']['type']
dataDir = config['data_loader']['args']['data_dir']
batch_size = config['data_loader']['args']['batch_size'] 
shuffle = config['data_loader']['args']['shuffle']
validation_split = config['data_loader']['args']['validation_split']
num_workers = config['data_loader']['args']['num_workers']
optimizer_type = config['optimizer']['type']
learning_rate = config['optimizer']['args']['lr'] 
epochs = config['trainer']['epochs']
 # # Dynamically import the model class
 
model_class = getattr(importlib.import_module('model.NNmodel'), arch)


################# Training #######################

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main():
    file_info_dict = dl.createFileInfoDict("/Net/elnino/data/obs/ERA5/global/daily/")

    #Data loading instance
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



   
    
    # Initialize the model
    model = model_class()
    
    # prepare for GPU training
    #if torch.cuda.device_count() > 1:
        #print(f"Using {torch.cuda.device_count()} GPUs")
        #model = nn.DataParallel(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(device)
    

    ################# Training #######################
    # get function handles of loss and metrics
    criterion = nn.MSELoss()
   

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    optimizer = optimizers[optimizer_type](model.parameters(), lr=learning_rate)

    # Number of epochs
    # epochs = epochs
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
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
        
            prediction = outputs.squeeze()

            # Append predicted outputs and true values for training
            train_predicted_outputs.extend(prediction.tolist())
            train_true_values.extend(targets.tolist())  
    
            # Compute the loss
            loss = criterion(prediction, targets.float())
        
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
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)
                all_val_outputs = torch.zeros((mc_samples, len(val_inputs)))
            
        
                val_outputs = model(val_inputs)
            

                # Compute mean prediction for validation
                val_prediction = val_outputs.squeeze()
            
                # Append predicted outputs and true values for validation
                val_predicted_outputs.extend(val_prediction.tolist())
                val_true_values.extend(val_targets.tolist())

            # Compute validation loss
            val_loss = criterion(torch.tensor(val_predicted_outputs), torch.tensor(val_true_values))
            val_losses.append(val_loss.item())
        
            # Print validation loss
            print(f'Validation Epoch [{epoch + 1}/{epochs}], Loss: {val_loss.item()}')
            
main()
