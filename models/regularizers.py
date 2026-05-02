import abc
import torch
import torch.nn as nn
from typing import Dict

class BaseRegularizer(nn.Module, abc.ABC):
    """
    Abstract Base Class for Information Regularization mechanisms.
    All regularizers (VAE, VQ-VAE, Implicit Rank-Minimizing AE, Sparse Coding) 
    must inherit from this class and implement the required methods.
    """
    def __init__(self, latent_dim: int, config: dict):
        super().__init__()
        self.latent_dim = latent_dim
        self.config = config

    @abc.abstractmethod
    def forward(self, z_e: torch.Tensor) -> torch.Tensor:
        """
        Applies the regularization mechanism to the encoder's output.
        
        Args:
            z_e: Unconstrained latent representation from encoder. Shape (B, latent_dim)
            
        Returns:
            z_q: Constrained/Regularized latent representation to pass to decoder. 
                 Shape (B, latent_dim)
        """
        # TODO: Implement forward pass here (e.g., adding noise, quantizing, sparsifying)
        pass

    @abc.abstractmethod
    def compute_loss(self, x: torch.Tensor, x_recon: torch.Tensor, z_e: torch.Tensor, z_q: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes the total loss, including reconstruction and any regularization penalties.
        
        Args:
            x: Original input image
            x_recon: Reconstructed image from the decoder
            z_e: Original encoded latent
            z_q: Regularized latent
            
        Returns:
            loss_dict: A dictionary containing at least:
                - 'recon_loss': The reconstruction penalty (e.g., MSE)
                - 'reg_loss': The regularization penalty (e.g., KL divergence, sparsity penalty)
                - 'total_loss': recon_loss + beta * reg_loss
        """
        # TODO: Implement loss computation here
        pass


class IdentityRegularizer(BaseRegularizer):
    """
    A simple unregularized Autoencoder pass-through for testing the boilerplate.
    Look at this as an example of the expected interface.
    """
    def forward(self, z_e: torch.Tensor) -> torch.Tensor:
        return z_e

    def compute_loss(self, x: torch.Tensor, x_recon: torch.Tensor, z_e: torch.Tensor, z_q: torch.Tensor) -> Dict[str, torch.Tensor]:
        recon_loss = nn.functional.mse_loss(x_recon, x)
        reg_loss = torch.tensor(0.0, device=x.device)
        return {
            'recon_loss': recon_loss,
            'reg_loss': reg_loss,
            'total_loss': recon_loss
        }
