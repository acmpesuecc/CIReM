import torch
import torch.nn as nn
import torch.optim as optim

def compute_reconstruction_error(x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    """Computes Standard MSE Reconstruction Error."""
    return nn.functional.mse_loss(x_recon, x)

def compute_effective_rank(latents: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Computes the Effective Rank of a batch of latent vectors using SVD.
    Effective rank is calculated as the Shannon entropy of the normalized singular values.
    
    Args:
        latents: Tensor of shape (B, latent_dim)
        eps: Small value for numerical stability
    Returns:
        effective_rank: Scalar tensor
    """
    # Center the latents
    latents_centered = latents - latents.mean(dim=0, keepdim=True)
    
    # Compute SVD. We only need singular values (S)
    try:
        _, S, _ = torch.linalg.svd(latents_centered, full_matrices=False)
    except torch._C._LinAlgError:
        # Fallback if SVD fails to converge
        return torch.tensor(0.0, device=latents.device)
        
    # Normalize singular values to create a distribution
    p = (S + eps) / torch.sum(S + eps)
    
    # Shannon entropy of normalized singular values
    entropy = -torch.sum(p * torch.log(p))
    
    # Effective rank is exp(entropy)
    effective_rank = torch.exp(entropy)
    return effective_rank

def compute_active_code_percentage(latents: torch.Tensor, threshold: float = 1e-3) -> float:
    """
    Calculates what percentage of the latent dimensions are 'active'.
    An active dimension is one where the absolute value is > threshold 
    for at least one sample in the batch, or mean absolute value is > threshold.
    
    Args:
        latents: Tensor of shape (B, latent_dim)
        threshold: The threshold to consider a code 'active'
    Returns:
        active_pct: Float percentage (0 to 100)
    """
    # Using mean absolute activation over the batch per dimension
    mean_abs_activation = torch.mean(torch.abs(latents), dim=0)
    active_dims = torch.sum(mean_abs_activation > threshold).item()
    return (active_dims / latents.size(1)) * 100.0


class LinearProbe(nn.Module):
    """
    A simple linear probe to measure how linearly separable the classes 
    are in the latent space. It is trained concurrently but on detached latents.
    """
    def __init__(self, latent_dim: int, num_classes: int, lr: float = 1e-3):
        super().__init__()
        self.classifier = nn.Linear(latent_dim, num_classes)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
        
    def update(self, detached_latents: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
        """
        Takes one gradient step for the linear probe.
        
        Args:
            detached_latents: Latents detached from the main computational graph
            labels: Ground truth labels
            
        Returns:
            loss: The classification loss
            accuracy: The classification accuracy (0.0 to 1.0)
        """
        self.optimizer.zero_grad()
        logits = self(detached_latents)
        loss = self.criterion(logits, labels)
        loss.backward()
        self.optimizer.step()
        
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean().item()
            
        return loss.item(), acc
