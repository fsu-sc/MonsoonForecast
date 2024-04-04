import torch.nn as nn
import torch.nn.functional as F
import random 

rand = random.uniform(0.1, 0.9)

# Generate a random number between 0.01 and 0.1
rand2 = random.uniform(0.01, 0.1)
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
        self.dropout = nn.Dropout(rand2)
        self.dropout2 = nn.Dropout(rand)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout2(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.max_pool3d(x, kernel_size=x.size()[2:])
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.Lnorm(x))# Flatten the output
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize the modified model
# model = EnhancedCNN()

# Print the modified model architecture
# print(model)
