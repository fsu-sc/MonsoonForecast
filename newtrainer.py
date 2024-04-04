import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16  # Using the B-16 variant as an example
import data_loader.dataset as ds 
# Modify ViT for regression
class ViTForRegression(nn.Module):
    def __init__(self):
        super(ViTForRegression, self).__init__()
        self.vit = vit_b_16(weights=None)
        self.regressor = nn.Linear(self.vit.heads[0].in_features, 1)  # Modify to output a single value

    def forward(self, x):
        x = self.vit(x)
        x = self.regressor(x)
        return x

# Initialize the model
model = ViTForRegression()
data_dir = "/Net/elnino/data/obs/ERA5/global/daily/"
year = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010]
offsets =[1,2,3,4,5,6,7,8,9,10,11,12]
# file_dic = ds.createFileInfoDict(data_dir)
variables = ['tp', 'mslp', 't2m', 'u200', 'u850', 'v200', 'v850']
# Assume the dataset is initialized as `dataset`
dataset = ds.NetCDFDataset(data_dir, year, offsets, variables)

# DataLoader
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for data, targets in dataloader:
        # Assuming data is already in the correct shape (B, C, H, W)
        outputs = model(data)
        loss = criterion(outputs.squeeze(), targets.float())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")