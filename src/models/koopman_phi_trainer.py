#!/usr/bin/env python3
"""
Koopman Phi Trainer
===================

This module trains a feature encoder Phi (built with a lightweight ResNet)
and a set of class-dependent Koopman operators K(p) that satisfy:

    Phi(y_t) ≈ K(p) Phi(x_t)

where x_t and y_t are consecutive states from the trajectory and p is a
one-hot indicator of the fault class. For now we use six different Koopman
matrices (one per class) but the implementation supports arbitrary class
counts.
"""

import os
import sys
import json
import argparse
import random
from copy import deepcopy
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.dataloader as dataloader_module  # noqa: E402


DEFAULT_CONFIG: Dict = {
    "data": {
        "path": "./data",
        "sample_step": 1,
        "window_size": 2000,
        "batch_size": 16,
        "pca_dim": 2,
        "preprocessing_path": "./checkpoints/koopman_phi/pca_params.pkl"
    },
    "model": {
        "input_dim": 2,
        "latent_dim": 32,
        "hidden_dim": 128,
        "num_blocks": 3,
        "num_classes": 6
    },
    "train": {
        "epochs": 200,
        "lr": 0.001,
        "weight_decay": 1e-5,
        "grad_clip": 5.0,
        "seed": 42,
        "device": "auto",
        "save_every": 50,
        "save_path": "./checkpoints/koopman_phi/best_model.pt"
    },
    "logging": {
        "use_wandb": False,
        "project": "koopman_phi_training",
        "run_name": None
    }
}


def set_seed(seed: int) -> None:
    """Set RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_wandb_run(logging_cfg: Dict, config: Dict) -> bool:
    """
    Initialize a Weights & Biases run if requested and available.
    """
    if not logging_cfg.get("use_wandb", False):
        return False
    if not WANDB_AVAILABLE:
        print("Warning: wandb is not installed; skipping cloud logging.")
        return False

    run_name = logging_cfg.get("run_name") or f"koopman_phi_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=logging_cfg.get("project", "koopman_phi_training"),
        name=run_name,
        config=config
    )
    print(f"W&B run initialized: {run_name}")
    return True


class ResidualBlock(nn.Module):
    """Simple fully-connected residual block used inside Phi."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.act(out + residual)
        return out


class PhiResNet(nn.Module):
    """
    ResNet-style encoder for Koopman features that returns [1, x, NN(x)].
    """

    def __init__(self, input_dim: int, latent_dim: int,
                 hidden_dim: int, num_blocks: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        residual_dim = latent_dim - (1 + input_dim)
        if residual_dim <= 0:
            raise ValueError(
                f"latent_dim must be greater than 1 + input_dim ({1 + input_dim}). "
                f"Got latent_dim={latent_dim}."
            )
        self.residual_dim = residual_dim

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.output_layer = nn.Linear(hidden_dim, residual_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        for block in self.blocks:
            out = block(out)
        nn_features = self.output_layer(out)
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        phi = torch.cat([ones, x, nn_features], dim=1)
        return phi


class KoopmanPhiModel(nn.Module):
    """
    Joint model that holds Phi and a Koopman matrix per class.
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int,
                 num_blocks: int, num_classes: int = 6):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.phi = PhiResNet(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks
        )
        # K matrices initialized close to identity for stability
        eye = torch.eye(latent_dim)
        init_koopman = torch.stack([eye.clone() for _ in range(num_classes)], dim=0)
        self.koopman_matrices = nn.Parameter(init_koopman)

    def class_koopman(self, p: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of class probabilities (one-hot) into Koopman matrices.

        Args:
            p: Tensor of shape (batch, num_classes)

        Returns:
            Tensor of shape (batch, latent_dim, latent_dim)
        """
        return torch.einsum('bn,nij->bij', p, self.koopman_matrices)

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Phi(x) and the Koopman propagated Phi(x).

        Args:
            x: Raw state tensor (batch, input_dim)
            p: One-hot tensor (batch, num_classes)

        Returns:
            phi_x: Encoded representation
            koopman_phi: Koopman propagated phi_x
        """
        phi_x = self.phi(x)
        koopman = self.class_koopman(p)
        koopman_phi = torch.bmm(koopman, phi_x.unsqueeze(-1)).squeeze(-1)
        return phi_x, koopman_phi


def load_config(config_path: Optional[str]) -> Dict:
    """Load config from JSON; fall back to defaults when file is absent."""
    base_config = deepcopy(DEFAULT_CONFIG)
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        for section in ["data", "model", "train", "logging"]:
            if section in user_config:
                base_config[section].update(user_config[section])
        return base_config
    return base_config


def create_dataloaders(data_config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/validation dataloaders with consistent preprocessing.
    """
    train_dataset, val_dataset = dataloader_module.load_dataset_from_folder(
        data_config["path"],
        data_config
    )
    scaled_train = dataloader_module.ScaledDataset(
        train_dataset,
        pca_dim=data_config.get("pca_dim", 2),
        fit_scalers=True
    )
    scaled_val = dataloader_module.ScaledDataset(
        val_dataset,
        pca_dim=data_config.get("pca_dim", 2)
    )
    scaled_val.set_preprocessing_params(scaled_train.get_preprocessing_params())

    preprocess_path = data_config.get("preprocessing_path")
    if preprocess_path:
        os.makedirs(os.path.dirname(preprocess_path), exist_ok=True)
        scaled_train.save_preprocessing_params(preprocess_path)
        print(f"Saved PCA/preprocessing parameters to {preprocess_path}")

    batch_size = data_config.get("batch_size", 16)
    train_loader = DataLoader(
        scaled_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )
    val_loader = DataLoader(
        scaled_val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    return train_loader, val_loader


def prepare_state_pairs(x_batch: torch.Tensor, p_batch: torch.Tensor
                        ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Take a batch of sequences and return flattened (x_t, y_t, p) pairs.
    """
    seq_len = x_batch.size(1)
    if seq_len < 2:
        return None
    x_t = x_batch[:, :-1, :]
    y_t = x_batch[:, 1:, :]
    num_pairs = (seq_len - 1) * x_batch.size(0)
    x_flat = x_t.reshape(num_pairs, -1)
    y_flat = y_t.reshape(num_pairs, -1)
    p_expanded = p_batch.unsqueeze(1).expand(-1, seq_len - 1, -1)
    p_flat = p_expanded.reshape(num_pairs, -1)
    return x_flat, y_flat, p_flat


def train_epoch(model: KoopmanPhiModel, loader: DataLoader,
                optimizer: torch.optim.Optimizer, device: torch.device,
                grad_clip: float) -> float:
    """Train model for one epoch."""
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_pairs = 0

    for x_batch, _, p_batch in tqdm(loader, desc="Train", leave=False):
        x_batch = x_batch.to(device)
        p_batch = p_batch.to(device)
        prepared = prepare_state_pairs(x_batch, p_batch)
        if prepared is None:
            continue
        x_flat, y_flat, p_flat = prepared
        x_flat = x_flat.to(device)
        y_flat = y_flat.to(device)
        p_flat = p_flat.to(device)

        optimizer.zero_grad()
        phi_x, koopman_phi = model(x_flat, p_flat)
        phi_y = model.phi(y_flat)
        loss = criterion(koopman_phi, phi_y)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_pairs = x_flat.size(0)
        total_loss += loss.item() * batch_pairs
        total_pairs += batch_pairs

    return total_loss / max(total_pairs, 1)


@torch.no_grad()
def evaluate(model: KoopmanPhiModel, loader: DataLoader,
             device: torch.device) -> float:
    """Compute validation loss."""
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_pairs = 0

    for x_batch, _, p_batch in tqdm(loader, desc="Val", leave=False):
        x_batch = x_batch.to(device)
        p_batch = p_batch.to(device)
        prepared = prepare_state_pairs(x_batch, p_batch)
        if prepared is None:
            continue
        x_flat, y_flat, p_flat = prepared
        x_flat = x_flat.to(device)
        y_flat = y_flat.to(device)
        p_flat = p_flat.to(device)

        _, koopman_phi = model(x_flat, p_flat)
        phi_y = model.phi(y_flat)
        loss = criterion(koopman_phi, phi_y)

        batch_pairs = x_flat.size(0)
        total_loss += loss.item() * batch_pairs
        total_pairs += batch_pairs

    return total_loss / max(total_pairs, 1)


def save_checkpoint(model: KoopmanPhiModel, optimizer: torch.optim.Optimizer,
                    epoch: int, config: Dict, val_loss: float) -> None:
    """Persist model/optimizer state."""
    save_path = config["train"]["save_path"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "best_val_loss": val_loss
    }
    torch.save(payload, save_path)
    print(f"Checkpoint saved to {save_path} (val loss={val_loss:.6f})")


def train_koopman_phi(config: Dict) -> None:
    """Main training routine."""
    train_cfg = config["train"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    logging_cfg = config.get("logging", {})

    set_seed(train_cfg.get("seed", 42))

    device_str = train_cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    train_loader, val_loader = create_dataloaders(data_cfg)

    model = KoopmanPhiModel(
        input_dim=model_cfg["input_dim"],
        latent_dim=model_cfg["latent_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_blocks=model_cfg["num_blocks"],
        num_classes=model_cfg.get("num_classes", 6)
    ).to(device)

    use_wandb = init_wandb_run(logging_cfg, config)
    if use_wandb:
        wandb.watch(model, log="gradients", log_freq=100)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0)
    )

    epochs = train_cfg["epochs"]
    grad_clip = train_cfg.get("grad_clip", 0.0)
    save_every = train_cfg.get("save_every", 0)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device, grad_clip)
        val_loss = evaluate(model, val_loader, device)
        print(f"Train loss: {train_loss:.6e} | Val loss: {val_loss:.6e}")

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"]
                },
                step=epoch
            )

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, config, val_loss)
            if use_wandb:
                wandb.log({"best_val_loss": best_val}, step=epoch)
        elif save_every and epoch % save_every == 0:
            save_checkpoint(model, optimizer, epoch, config, val_loss)

    print(f"Training complete. Best validation loss: {best_val:.6f}")
    if use_wandb:
        wandb.summary["best_val_loss"] = best_val
        wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Koopman Phi model with class-dependent operators."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config. If omitted, DEFAULT_CONFIG is used."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    train_koopman_phi(cfg)
