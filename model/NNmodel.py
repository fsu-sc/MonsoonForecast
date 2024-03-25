import torch.nn as nn
import torch.nn.functional as F

class EnhancedCNN(nn.Module):
    def __init__(self, p=0.3, num_classes=1,in_channels=7):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=1)
        # self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        # Adjusted the linear layer's input features according to your model's architecture.
        self.fc1 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 1)
        
        # D, H, W need to be calculated based on the input size and convolutions

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        # x = self.pool(x)
        # x = F.relu(self.conv3(x))
        x = F.max_pool3d(x, kernel_size=x.size()[2:])
        x = x.view(x.size(0), -1)  # Flatten the output
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        return x

# Initialize the modified model
model = EnhancedCNN()

# Print the modified model architecture
print(model)
