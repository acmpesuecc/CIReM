import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb

from models import ConvEncoder, ConvDecoder, IdentityRegularizer
from utils import compute_effective_rank, compute_active_code_percentage, LinearProbe

def parse_args():
    parser = argparse.ArgumentParser(description="Latent Variable Models Testbench")
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
                        help='Dataset to use for training (mnist or cifar10)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for autoencoder')
    parser.add_argument('--probe_lr', type=float, default=1e-3, help='Learning rate for linear probe')
    parser.add_argument('--latent_dim', type=int, default=64, help='Dimensionality of the latent space')
    parser.add_argument('--beta', type=float, default=1.0, help='Weight for the regularization loss')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--entity', type=str, default='cirem', help='Wandb entity (username or team name)')
    return parser.parse_args()


def get_dataloaders(dataset_name, batch_size):
    """
    Returns train and test dataloaders for the specified dataset.
    Pads MNIST to 32x32 to match CIFAR-10 spatial dimensions for the ConvNet.
    """
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.Pad(2), # 28x28 -> 32x32
            transforms.ToTensor()
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        in_channels = 1
        num_classes = 10
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        in_channels = 3
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader, in_channels, num_classes


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize wandb
    mode = 'disabled' if args.disable_wandb else 'online'
    wandb.init(project="info-reg-testbench", entity=args.entity, config=vars(args), mode=mode)

    # 1. Dataloaders
    train_loader, test_loader, in_channels, num_classes = get_dataloaders(args.dataset, args.batch_size)

    # 2. Models
    encoder = ConvEncoder(in_channels=in_channels, latent_dim=args.latent_dim).to(device)
    decoder = ConvDecoder(latent_dim=args.latent_dim, out_channels=in_channels).to(device)
    
    # TODO: swap IdentityRegularizer with their own implementations
    regularizer_config = {'beta': args.beta}
    regularizer = IdentityRegularizer(latent_dim=args.latent_dim, config=regularizer_config).to(device)
    
    # Linear probe for evaluating representation quality
    linear_probe = LinearProbe(latent_dim=args.latent_dim, num_classes=num_classes, lr=args.probe_lr).to(device)

    # 3. Optimizers
    ae_params = list(encoder.parameters()) + list(decoder.parameters()) + list(regularizer.parameters())
    optimizer_ae = optim.Adam(ae_params, lr=args.lr)

    # 4. Training Loop
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        regularizer.train()
        linear_probe.train()
        
        train_recon_loss = 0.0
        train_reg_loss = 0.0
        train_total_loss = 0.0
        train_probe_acc = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # --- Autoencoder Forward Pass ---
            z_e = encoder(x)
            z_q = regularizer(z_e)
            x_recon = decoder(z_q)
            
            # Loss Computation
            loss_dict = regularizer.compute_loss(x, x_recon, z_e, z_q)
            
            recon_loss = loss_dict.get('recon_loss', torch.tensor(0.0, device=device))
            reg_loss = loss_dict.get('reg_loss', torch.tensor(0.0, device=device))
            total_loss = loss_dict.get('total_loss', recon_loss + args.beta * reg_loss)
            
            # Autoencoder Backward Pass
            optimizer_ae.zero_grad()
            total_loss.backward()
            optimizer_ae.step()
            
            # --- Linear Probe Forward/Backward ---
            # Train the probe concurrently on detached latents
            probe_loss, probe_acc = linear_probe.update(z_q.detach(), y)
            
            # Accumulate metrics
            train_recon_loss += recon_loss.item()
            train_reg_loss += reg_loss.item()
            train_total_loss += total_loss.item()
            train_probe_acc += probe_acc
            
        # Average metrics over epoch
        avg_train_recon = train_recon_loss / len(train_loader)
        avg_train_reg = train_reg_loss / len(train_loader)
        avg_train_total = train_total_loss / len(train_loader)
        avg_train_probe_acc = train_probe_acc / len(train_loader)
        
        print(f"Epoch [{epoch}/{args.epochs}] Train - Total: {avg_train_total:.4f}, Recon: {avg_train_recon:.4f}, Reg: {avg_train_reg:.4f}, Probe Acc: {avg_train_probe_acc:.4f}")

        # --- Validation Loop ---
        encoder.eval()
        decoder.eval()
        regularizer.eval()
        linear_probe.eval()
        
        val_recon_loss = 0.0
        val_reg_loss = 0.0
        val_total_loss = 0.0
        val_probe_acc = 0.0
        val_effective_rank = 0.0
        val_active_code_pct = 0.0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                
                z_e = encoder(x)
                z_q = regularizer(z_e)
                x_recon = decoder(z_q)
                
                loss_dict = regularizer.compute_loss(x, x_recon, z_e, z_q)
                recon_loss = loss_dict.get('recon_loss', torch.tensor(0.0, device=device))
                reg_loss = loss_dict.get('reg_loss', torch.tensor(0.0, device=device))
                total_loss = loss_dict.get('total_loss', recon_loss + args.beta * reg_loss)
                
                # Probe evaluation
                logits = linear_probe(z_q)
                preds = torch.argmax(logits, dim=1)
                acc = (preds == y).float().mean().item()
                
                # Custom Metrics
                eff_rank = compute_effective_rank(z_q)
                act_code_pct = compute_active_code_percentage(z_q)
                
                val_recon_loss += recon_loss.item()
                val_reg_loss += reg_loss.item()
                val_total_loss += total_loss.item()
                val_probe_acc += acc
                val_effective_rank += eff_rank.item()
                val_active_code_pct += act_code_pct
                
        # Average metrics over val set
        avg_val_recon = val_recon_loss / len(test_loader)
        avg_val_reg = val_reg_loss / len(test_loader)
        avg_val_total = val_total_loss / len(test_loader)
        avg_val_probe_acc = val_probe_acc / len(test_loader)
        avg_val_eff_rank = val_effective_rank / len(test_loader)
        avg_val_act_code_pct = val_active_code_pct / len(test_loader)
        
        print(f"Epoch [{epoch}/{args.epochs}] Val   - Total: {avg_val_total:.4f}, Recon: {avg_val_recon:.4f}, Reg: {avg_val_reg:.4f}, Probe Acc: {avg_val_probe_acc:.4f}, Eff Rank: {avg_val_eff_rank:.2f}, Active Codes: {avg_val_act_code_pct:.1f}%")
        
        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "train/recon_loss": avg_train_recon,
            "train/reg_loss": avg_train_reg,
            "train/total_loss": avg_train_total,
            "train/probe_acc": avg_train_probe_acc,
            "val/recon_loss": avg_val_recon,
            "val/reg_loss": avg_val_reg,
            "val/total_loss": avg_val_total,
            "val/probe_acc": avg_val_probe_acc,
            "val/effective_rank": avg_val_eff_rank,
            "val/active_code_pct": avg_val_act_code_pct
        })
        
    wandb.finish()

if __name__ == '__main__':
    main()
