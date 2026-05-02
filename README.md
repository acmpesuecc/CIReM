# CIReM

Comparing 4 classes of regularisers on a testbench to evaluate trade-off between reconstruction quality and latent disentanglement.

This repository serves as a boilerplate for a research project to compare four different information regularization mechanisms in Latent Variable Models: Standard VAE, VQ-VAE, Implicit Rank-Minimizing AE, and Sparse Coding.

## Setup Instructions

1. **Clone the repository and set up a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Weights & Biases (wandb) Setup**:
   The testbench uses `wandb` for experiment tracking. Make sure to log in to your account:
   ```bash
   wandb login
   ```
   *Note: Runs are automatically logged to the `cirem` team entity.*

## Workflow

Your task is to implement the specific regularization logic and loss functions. You do not need to worry about writing the training loop, metrics, or the autoencoder backbone.

1. **Create your regularizer**:
   Create a new file in the `models/` directory (e.g., `models/vq_vae.py`).

2. **Implement the interface**:
   Your class must inherit from `BaseRegularizer` (found in `models/regularizers.py`). You need to implement:
   - `forward(self, z_e)`: Applies the regularization (e.g. adds noise, quantizes) and returns the constrained latent vector `z_q`.
   - `compute_loss(self, x, x_recon, z_e, z_q)`: Computes and returns a dictionary of losses. It must include at least `'recon_loss'`, `'reg_loss'`, and `'total_loss'`.

3. **Train your model**:
   Update `train.py` to instantiate your custom regularizer instead of the `IdentityRegularizer`. Then run the script:
   ```bash
   python train.py --dataset mnist --epochs 20 --latent_dim 64
   ```
   
   To see all available hyperparameters, run:
   ```bash
   python train.py --help
   ```

## Repository Structure

- `models/base_ae.py`: Contains the standard Convolutional Encoder and Decoder.
- `models/regularizers.py`: Contains the `BaseRegularizer` Abstract Base Class and an `IdentityRegularizer` example.
- `utils/metrics.py`: Contains custom evaluation metrics (MSE, Linear Probe, Effective Rank, Active Code %).
- `train.py`: The main training and validation loop with wandb integration.
