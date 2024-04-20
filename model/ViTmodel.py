import torch
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import einops
# Assuming train_data is a list of pairs [tensor, integer]
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

patch_size = 16         # Patch size (P) = 16
latent_size = 768       # Latent vector (D). ViT-Base uses 768
n_channels = 7          # Number of channels for input images
num_heads = 12          # ViT-Base uses 12 heads
num_encoders = 12       # ViT-Base uses 12 encoder layers
dropout = 0.1           # Dropout = 0.1 is used with ViT-Base & ImageNet-21k
num_classes = 1      # Number of classes in CIFAR10 dataset
size = 1440              # Size used for training = 224

epochs = 30             # Number of epochs
base_lr = 10e-3         # Base LR
weight_decay = 0.03     # Weight decay for ViT-Base (on ImageNet-21k)
batch_size = 4





class InputEmbedding3D(nn.Module):
    def __init__(self, patch_size, patch_depth, n_channels, device, latent_size):
        super(InputEmbedding3D, self).__init__()
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.patch_depth = patch_depth
        self.n_channels = n_channels
        self.device = device
        self.input_size = self.patch_size * self.patch_size * self.patch_depth * self.n_channels

        self.linearProjection = nn.Linear(self.input_size, self.latent_size)

        self.class_token = nn.Parameter(torch.randn(1, 1, self.latent_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + self.patch_depth * self.patch_size * self.patch_size, self.latent_size))

    def forward(self, input_data):
        assert input_data.size(2) % self.patch_depth == 0, "Input depth must be divisible by patch_depth"
        assert input_data.size(3) % self.patch_size == 0, "Input height must be divisible by patch_size"
        assert input_data.size(4) % self.patch_size == 0, "Input width must be divisible by patch_size"

        input_data = input_data.to(self.device)

        # Adjust the rearrange function to handle 3D patches
        patches = einops.rearrange(
            input_data, 'b c (d d1) (h h1) (w w1) -> b (d h w) (d1 h1 w1 c)', 
            d1=self.patch_depth, h1=self.patch_size, w1=self.patch_size)

        linear_projection = self.linearProjection(patches)

        # Dynamically expand the class token and positional embeddings to match the batch size
        class_token = self.class_token.expand(input_data.size(0), -1, -1)
        linear_projection = torch.cat((class_token, linear_projection), dim=1)

        pos_embed = self.pos_embedding.expand(input_data.size(0), -1, -1)
        linear_projection += pos_embed

        return linear_projection

class EncoderBlock(nn.Module):
    def __init__(self, latent_size, num_heads, device, dropout):
        super(EncoderBlock, self).__init__()

        self.latent_size = latent_size
        self.num_heads = num_heads
        self.device = device
        self.dropout = dropout

        # Normalization layer for both sublayers
        self.norm = nn.LayerNorm(self.latent_size).to(device)
        
        # Multi-Head Attention layer
        self.multihead = nn.MultiheadAttention(
            self.latent_size, self.num_heads, dropout=self.dropout).to(device)

        # MLP_head layer in the encoder. I use the same configuration as that 
        # used in the original VitTransformer implementation. The ViT-Base
        # variant uses MLP_head size 3072, which is latent_size*4.
        self.enc_MLP = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size * 4).to(device),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_size * 4, self.latent_size).to(device),
            nn.Dropout(self.dropout)
        )

    def forward(self, embedded_patches):
        # Ensure input is on the correct device
        embedded_patches = embedded_patches.to(self.device)

        # First sublayer: Norm + Multi-Head Attention + residual connection.
        first_norm_out = self.norm(embedded_patches)
        attention_output = self.multihead(first_norm_out, first_norm_out, first_norm_out)[0]

        # First residual connection
        first_added_output = attention_output + embedded_patches

        # Second sublayer: Norm + enc_MLP (Feed forward)
        second_norm_out = self.norm(first_added_output)
        ff_output = self.enc_MLP(second_norm_out)

        # Return the output of the second residual connection
        return ff_output + first_added_output

class VitTransformer(nn.Module):
    def __init__(self, num_encoders, latent_size, device, num_classes, num_heads, dropout, patch_size, n_channels):
        super(VitTransformer, self).__init__()
        self.num_encoders = num_encoders
        self.latent_size = latent_size
        self.device = device
        self.num_classes = num_classes
        self.dropout = dropout
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.latent_size = latent_size
        self.num_heads = num_heads
        # Removed patch_depth from here

        # Initialize the encoder stack
        self.encStack = nn.ModuleList([EncoderBlock(latent_size, num_heads, device, dropout) for i in range(self.num_encoders)])

        # MLP head for classification
        self.MLP_head = nn.Sequential(
            nn.LayerNorm(latent_size),
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_size, num_classes)
        ).to(device)

    def forward(self, test_input):
        # Dynamically infer patch_depth from the input
        patch_depth = test_input.size(2)  # Assuming test_input shape is [B, C, D, H, W]

        # Reinitialize the embedding layer on each forward pass (inefficient)
        self.embedding = InputEmbedding3D(self.patch_size, patch_depth, self.n_channels, self.device, self.latent_size).to(self.device)

        # Apply input embedding to the input data
        enc_output = self.embedding(test_input)

        # Pass the output through the encoder stack
        for enc_layer in self.encStack:
            enc_output = enc_layer(enc_output)

        # Extract the output embedding of the [class] token
        cls_token_embedding = enc_output[:, 0]

        # Return the classification output
        return self.MLP_head(cls_token_embedding)
