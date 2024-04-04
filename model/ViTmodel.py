import torch
from torch.utils.data import DataLoader

# Assuming train_data is a list of pairs [tensor, integer]


class ViTransform():
    def __init__(self, train_data ):
        self.train_data = train_data

    def __call__(self, tensor):
        # Split the tensor into patches
        patches = tensor.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(patches.size(0), -1, self.patch_size, self.patch_size)
        return patches
    
    
    def convert_to_tensor(self):
        train_data_tensor = torch.stack([pair[0] for pair in self.train_data])
        onset_labels = [pair[1] for pair in self.train_data]

        # Convert the onset labels into a tensor
        onset_labels_tensor = torch.tensor(onset_labels, dtype=torch.long)

        return train_data_tensor, onset_labels_tensor
       
