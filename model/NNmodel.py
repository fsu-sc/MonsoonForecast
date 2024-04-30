import torch.nn as nn
import torch.nn.functional as F
import random 

rand = random.uniform(0.1, 0.9)

# Generate a random number between 0.01 and 0.1
rand2 = random.uniform(0.01, 0.1)


import torch.nn as nn
import torch.nn.functional as F

class EnhancedCNN(nn.Module):
    def __init__(self, p=0.3, num_classes=1, in_channels=7):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3, 3), padding=1)
        self.norm1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=1)
        self.norm2 = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Lnorm = nn.LayerNorm(64)
        self.fc1 = nn.Linear(64,32)
        # self.Lnorm= nn.LayerNorm# Adjusted input features for the linear layer
        self.dropout = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(rand)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.max_pool3d(x, kernel_size=x.size()[2:])
        # x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.Lnorm(x))# Flatten the output
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Generate a dense model with two inputs 10 neurons and 4 hidden layers with BN in between. Output linear, Relu in hidden layers
class DenseModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, num_layers=40):
        super(DenseModel, self).__init__()
        self.fc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.relu = nn.ReLU()
        
        # Input layer
        self.fc_layers.append(nn.Linear(input_size, hidden_size))
        self.bn_layers.append(nn.BatchNorm1d(hidden_size))
        
        # Hidden layers
        for _ in range(num_layers):
            self.fc_layers.append(nn.Linear(hidden_size, hidden_size))
            self.bn_layers.append(nn.BatchNorm1d(hidden_size))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, 1)  # Output only one value
        
    def forward(self, x):
        # Add a batch dimension if input is scalar
        if x.dim() == 0:
            x = x.unsqueeze(0)
        
        for fc, bn in zip(self.fc_layers, self.bn_layers):
            x = fc(x)
            x = bn(x)
            x = self.relu(x)
        x = self.output_layer(x)  # Use the output layer directly
        return x
    
import torch
  

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.fc1 = nn.Linear(1, 16)  # Input dimension is 1, output dimension is 1 (single output)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 12)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# class EnhancedCNN(nn.Module):
#     def __init__(self, p=0.3, num_classes=1, in_channels=7):
#         super(EnhancedCNN, self).__init__()
#         self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=(5, 5, 5), padding=1)
#         self.norm1 = nn.BatchNorm3d(32)
#         self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=1)
#         self.norm2 = nn.BatchNorm3d(64)
#         self.conv3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), padding=1)  # Corrected in_channels
#         self.norm3 = nn.BatchNorm3d(128)
#         self.pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)  # Corrected padding
#         self.Lnorm = nn.LayerNorm(64 * 357 * 358 * 717)  # Corrected input size for LayerNorm
#         self.fc1 = nn.Linear(64 * 357 * 358 * 717, 32)
#         self.dropout = nn.Dropout(p=p)
#         self.fc2 = nn.Linear(32, num_classes)

#     def forward(self, x):
#         x = F.relu(self.norm1(self.conv1(x)))  # Corrected conv layer indexing
#         x = F.relu(self.norm2(self.conv2(x)))
#         # x = F.relu(self.norm3(self.conv3(x)))  # Added missing convolutional layer
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.Lnorm(x))
#         x = self.dropout(x)
#         x = self.fc1(x)
#         x = self.dropout(x)  # Added dropout layer
#         x = self.fc2(x)
#         return x




# Initialize the modified model
# model = EnhancedCNN()

# Print the modified model architecture
# print(model)
