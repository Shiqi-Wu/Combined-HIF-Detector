#!/usr/bin/env python3
"""
Koopman Phi residual training.

Stage 1: learn shared Phi and a global Koopman operator K_base.
Stage 2: freeze Phi/K_base, compute normalized residuals, and train
class-dependent residual operators K_res(p) such that:

    K_res(p) Phi(x) ≈ (Phi(y) - K_base Phi(x) - μ) / σ

This follows the requested two-step procedure where Phi and K are
shared initially, and per-class dynamics learn to explain residuals
(with a second rescaling of the residual term).
"""

import argparse
import json
import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow imports from project root
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.dataloader as dataloader_module  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ResidualBlock(nn.Module):
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


class PhiEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int,
                 hidden_dim: int, num_blocks: int):
        super().__init__()
        if latent_dim <= input_dim + 1:
            raise ValueError("latent_dim must be greater than input_dim + 1.")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.residual_dim = latent_dim - (input_dim + 1)

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.output_layer = nn.Linear(hidden_dim, self.residual_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        for block in self.blocks:
            out = block(out)
        nn_features = self.output_layer(out)
        ones = torch.ones(x.size(0), 1, dtype=x.dtype, device=x.device)
        phi = torch.cat([ones, x, nn_features], dim=1)
        return phi


class ResidualKoopmanModel(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int,
                 num_blocks: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.phi = PhiEncoder(input_dim, latent_dim, hidden_dim, num_blocks)
        self.base_K = nn.Parameter(torch.eye(latent_dim))
        self.residual_K = nn.Parameter(torch.zeros(num_classes, latent_dim, latent_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.phi(x)

    def apply_base(self, phi_x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(phi_x, self.base_K.t())

    def apply_residual(self, phi_x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phi_x: (batch_pairs, latent_dim)
            p: (batch_pairs, num_classes) one-hot
        """
        koopman = torch.einsum('bn,nij->bij', p, self.residual_K)
        pred = torch.bmm(koopman, phi_x.unsqueeze(-1)).squeeze(-1)
        return pred


DEFAULT_CONFIG: Dict = {
    "data": {
        "path": "./data",
        "sample_step": 1,
        "window_size": 2000,
        "batch_size": 16,
        "pca_dim": 2,
        "preprocessing_path": "./checkpoints/koopman_residual/pca_params.pkl"
    },
    "model": {
        "input_dim": 2,
        "latent_dim": 32,
        "hidden_dim": 128,
        "num_blocks": 3,
        "num_classes": 6
    },
    "train": {
        "base_epochs": 150,
        "residual_epochs": 100,
        "base_lr": 0.001,
        "residual_lr": 0.001,
        "weight_decay": 1e-5,
        "grad_clip": 5.0,
        "seed": 42,
        "device": "auto",
        "save_path": "./checkpoints/koopman_residual/best_model.pt"
    }
}


def load_config(config_path: Optional[str]) -> Dict:
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
        for section in cfg:
            if section in user_cfg:
                cfg[section].update(user_cfg[section])
        return cfg
    return json.loads(json.dumps(DEFAULT_CONFIG))


def create_dataloaders(data_cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = dataloader_module.load_dataset_from_folder(
        data_cfg["path"],
        data_cfg
    )
    scaled_train = dataloader_module.ScaledDataset(
        train_dataset,
        pca_dim=data_cfg.get("pca_dim", 2),
        fit_scalers=True
    )
    scaled_val = dataloader_module.ScaledDataset(
        val_dataset,
        pca_dim=data_cfg.get("pca_dim", 2)
    )
    scaled_val.set_preprocessing_params(scaled_train.get_preprocessing_params())

    preprocess_path = data_cfg.get("preprocessing_path")
    if preprocess_path:
        os.makedirs(os.path.dirname(preprocess_path), exist_ok=True)
        scaled_train.save_preprocessing_params(preprocess_path)
        print(f"Saved preprocessing parameters to {preprocess_path}")

    train_loader = DataLoader(
        scaled_train,
        batch_size=data_cfg.get("batch_size", 16),
        shuffle=True,
        drop_last=False
    )
    val_loader = DataLoader(
        scaled_val,
        batch_size=data_cfg.get("batch_size", 16),
        shuffle=False,
        drop_last=False
    )
    return train_loader, val_loader


def prepare_pairs(x_batch: torch.Tensor, p_batch: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_len = x_batch.size(1)
    if seq_len < 2:
        raise ValueError("Sequence length must be >= 2.")
    x_t = x_batch[:, :-1, :]
    y_t = x_batch[:, 1:, :]
    batch_size = x_batch.size(0)
    pairs = x_t.size(1)

    x_flat = x_t.reshape(batch_size * pairs, -1)
    y_flat = y_t.reshape(batch_size * pairs, -1)
    p_expand = p_batch.unsqueeze(1).expand(-1, pairs, -1).reshape(batch_size * pairs, -1)
    return x_flat, y_flat, p_expand


def train_shared_phase(model: ResidualKoopmanModel, loader: DataLoader,
                       optimizer: torch.optim.Optimizer, device: torch.device,
                       grad_clip: float, epochs: int) -> None:
    criterion = nn.MSELoss()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        total = 0
        for x_batch, _, p_batch in tqdm(loader, desc=f"Shared Epoch {epoch}", leave=False):
            x_batch = x_batch.to(device)
            p_batch = p_batch.to(device)
            optimizer.zero_grad()
            x_flat, y_flat, _ = prepare_pairs(x_batch, p_batch)
            phi_x = model.encode(x_flat.to(device))
            phi_y = model.encode(y_flat.to(device))
            pred = model.apply_base(phi_x)
            loss = criterion(pred, phi_y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item() * phi_x.size(0)
            total += phi_x.size(0)
        print(f"[Shared] Epoch {epoch}/{epochs} - Loss: {epoch_loss / total:.6e}")


def compute_residual_stats(model: ResidualKoopmanModel, loader: DataLoader,
                           device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    residuals = []
    with torch.no_grad():
        for x_batch, _, p_batch in loader:
            x_batch = x_batch.to(device)
            p_batch = p_batch.to(device)
            x_flat, y_flat, _ = prepare_pairs(x_batch, p_batch)
            phi_x = model.encode(x_flat.to(device))
            phi_y = model.encode(y_flat.to(device))
            base_pred = model.apply_base(phi_x)
            residual = (phi_y - base_pred).cpu()
            residuals.append(residual)
    residual_cat = torch.cat(residuals, dim=0)
    mean = residual_cat.mean(dim=0)
    std = residual_cat.std(dim=0)
    std[std == 0] = 1.0
    return mean, std


def train_residual_phase_dynamic_norm(model: ResidualKoopmanModel, loader: DataLoader,
                                      optimizer: torch.optim.Optimizer, device: torch.device,
                                      grad_clip: float, epochs: int,
                                      update_freq: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    criterion = nn.MSELoss()
    model.base_K.requires_grad_(False)

    residual_mean = None
    residual_std = None

    for epoch in range(1, epochs + 1):
        if residual_mean is None or (epoch - 1) % update_freq == 0:
            mean, std = compute_residual_stats(model, loader, device)
            std = torch.clamp(std, min=1e-3)
            residual_mean = mean.to(device)
            residual_std = std.to(device)
            print(f"[Residual Phase] Updated μ, σ at epoch {epoch}")

        model.train()
        epoch_loss = 0.0
        total = 0
        for x_batch, _, p_batch in tqdm(loader, desc=f"Residual Epoch {epoch}", leave=False):
            x_batch = x_batch.to(device)
            p_batch = p_batch.to(device)
            optimizer.zero_grad()
            x_flat, y_flat, p_flat = prepare_pairs(x_batch, p_batch)
            phi_x = model.encode(x_flat.to(device))
            with torch.no_grad():
                phi_y = model.encode(y_flat.to(device))
                base_pred = model.apply_base(phi_x)
                target = (phi_y - base_pred - residual_mean) / residual_std
            pred = model.apply_residual(phi_x, p_flat.to(device))
            loss = criterion(pred, target)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(model.phi.parameters()) + [model.residual_K], grad_clip)
            optimizer.step()
            epoch_loss += loss.item() * phi_x.size(0)
            total += phi_x.size(0)
        print(f"[Residual] Epoch {epoch}/{epochs} - Loss: {epoch_loss / total:.6e}")

    return residual_mean, residual_std


def train_model(config: Dict) -> None:
    train_cfg = config["train"]
    data_cfg = config["data"]
    model_cfg = config["model"]

    set_seed(train_cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() and train_cfg.get("device", "auto") == "auto"
                          else train_cfg.get("device", "cpu"))
    print(f"Using device: {device}")

    train_loader, val_loader = create_dataloaders(data_cfg)

    model = ResidualKoopmanModel(
        input_dim=model_cfg["input_dim"],
        latent_dim=model_cfg["latent_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_blocks=model_cfg["num_blocks"],
        num_classes=model_cfg.get("num_classes", 6)
    ).to(device)

    base_optimizer = torch.optim.Adam(
        list(model.phi.parameters()) + [model.base_K],
        lr=train_cfg["base_lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0)
    )
    residual_optimizer = torch.optim.Adam(
        list(model.phi.parameters()) + [model.residual_K],
        lr=train_cfg["residual_lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0)
    )

    train_shared_phase(
        model, train_loader, base_optimizer, device,
        train_cfg.get("grad_clip", 0.0), train_cfg["base_epochs"]
    )
    final_mean, final_std = train_residual_phase_dynamic_norm(
        model, train_loader, residual_optimizer, device,
        train_cfg.get("grad_clip", 0.0), train_cfg["residual_epochs"]
    )

    save_path = train_cfg["save_path"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "residual_mean": final_mean.detach().cpu(),
            "residual_std": final_std.detach().cpu()
        },
        save_path
    )
    print(f"Saved residual Koopman model to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train residual Koopman Phi model.")
    parser.add_argument("--config", type=str, default=None, help="Config JSON path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    train_model(cfg)
