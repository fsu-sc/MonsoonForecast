import torch.nn as nn
import torch
import torch.nn.functional as F
import random 

# Generate a dense model with two inputs 10 neurons and 4 hidden layers with BN in between. Output linear, Relu in hidden layers
class DenseModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, num_layers=4, dropout_rate=0.1):
        super(DenseModel, self).__init__()
        
        if hidden_size == 0:
            # Single linear layer for linear regression
            self.linear = nn.Linear(input_size, 1)
        else:
            # First layer components
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.dropout1 = nn.Dropout(dropout_rate)
            
            # Hidden layers components
            self.hidden_linears = nn.ModuleList()
            self.hidden_bns = nn.ModuleList() 
            self.hidden_dropouts = nn.ModuleList()
            
            for _ in range(num_layers - 1):
                self.hidden_linears.append(nn.Linear(hidden_size, hidden_size))
                self.hidden_bns.append(nn.BatchNorm1d(hidden_size))
                self.hidden_dropouts.append(nn.Dropout(dropout_rate))
            
            # Output layer
            self.output_layer = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        if hasattr(self, 'linear'):
            # Linear regression case
            return self.linear(x)
        else:
            # First layer
            x = self.linear1(x)
            x = F.relu(x)
            x = self.bn1(x)
            x = self.dropout1(x)
            
            # Hidden layers
            for linear, bn, dropout in zip(self.hidden_linears, self.hidden_bns, self.hidden_dropouts):
                x = linear(x)
                x = F.relu(x)
                x = bn(x)
                x = dropout(x)
                
            # Output layer
            x = self.output_layer(x)
            return x