import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import model.NNmodel as ECNN
import data_loader.dataset as ds

# Create an instance of the model
model = ECNN.EnhancedCNN(in_channels=6)

# Load the saved model parameters
checkpoint = torch.load('trained_models/(drop_0.5)monsoon_offset_trainer.pth')

# If the model was trained with DataParallel, remove 'module.' from the keys
new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}

# Load the model state dictionary
model.load_state_dict(new_state_dict)

# Ensure that your model is set to evaluation mode if you intend to perform inference
model.eval()

# Define criterion
criterion = nn.MSELoss()

# Load test data
data_dir = "/Net/elnino/data/obs/ERA5/global/daily/"
val_data = ds.NetCDFDataset(data_dir, [1990], [12])
test_loader = DataLoader(val_data)

# Evaluate the model on test data
test_loss = 0.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs).squeeze()
        test_loss += criterion(outputs, targets).item()
        
        # Extracting and printing predicted values for each batch
        predicted_values = outputs.tolist()  # Convert tensor to list
        print("Predicted Values:", predicted_values)

# Calculate the average test loss over all batches
average_test_loss = test_loss / len(test_loader.dataset)
print("Average Test Loss:", average_test_loss)
