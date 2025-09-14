# model.py

import torch
import torch.nn as nn
import numpy as np

class FourierFeatureMapping(nn.Module):
    """
    This layer transforms the input coordinates using a Fourier feature mapping.
    This helps the network learn high-frequency functions much more effectively.
    """
    def __init__(self, input_dims, mapping_size, scale=10.0):
        super(FourierFeatureMapping, self).__init__()
        self.input_dims = input_dims
        self.mapping_size = mapping_size
        # B is a random but fixed matrix, drawn from a Gaussian distribution
        # This is a key part of the technique.
        self.B = torch.randn((input_dims, mapping_size)) * scale
        
    def forward(self, x):
        # Move B to the same device as the input x
        self.B = self.B.to(x.device)
        
        # The Fourier feature mapping formula
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        
        # --- NEW: Define the Fourier Feature Mapping layer ---
        self.fourier_mapper = FourierFeatureMapping(input_dims=2, mapping_size=256, scale=10.0)
        
        # The main network now takes the mapped features as input
        # The input size is 2 * mapping_size because we concatenate sin and cos
        self.net = nn.Sequential(
            nn.Linear(2 * 256, 128), # Input layer size is now 512
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 2)       # Output layer: 2 neurons (Ey, Bz)
        )

    def forward(self, t, x):
        inputs = torch.cat([t, x], dim=1)
        
        # --- NEW: Apply the mapping first ---
        mapped_inputs = self.fourier_mapper(inputs)
        
        return self.net(mapped_inputs)