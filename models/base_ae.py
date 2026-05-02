import torch
import torch.nn as nn

class ConvEncoder(nn.Module):
    """
    Standard Convolutional Encoder.
    Assumes input images are padded/resized to 32x32.
    """
    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Input: (in_channels, 32, 32)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1), # -> (32, 16, 16)
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),          # -> (64, 8, 8)
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),         # -> (128, 4, 4)
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),        # -> (256, 2, 2)
            nn.ReLU(True)
        )
        
        self.fc = nn.Linear(256 * 2 * 2, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, in_channels, 32, 32)
        Returns:
            latent: Tensor of shape (B, latent_dim)
        """
        h = self.net(x)
        h = h.view(h.size(0), -1) # Flatten
        latent = self.fc(h)
        return latent


class ConvDecoder(nn.Module):
    """
    Standard Convolutional Decoder.
    Outputs images of size 32x32.
    """
    def __init__(self, latent_dim: int, out_channels: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        
        self.fc = nn.Linear(latent_dim, 256 * 2 * 2)
        
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> (128, 4, 4)
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 8, 8)
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> (32, 16, 16)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1), # -> (out_channels, 32, 32)
            # Use Sigmoid or Tanh depending on data scaling (assumes [0, 1] normalization here)
            nn.Sigmoid() 
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Tensor of shape (B, latent_dim)
        Returns:
            recon: Tensor of shape (B, out_channels, 32, 32)
        """
        h = self.fc(z)
        h = h.view(h.size(0), 256, 2, 2) # Reshape
        recon = self.net(h)
        return recon
