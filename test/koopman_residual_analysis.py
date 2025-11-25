#!/usr/bin/env python3
"""
Diagnostic tool for the residual Koopman Phi model.

It reports:
  1. Variance of each Phi dimension on a sampled batch.
  2. Variance of residual targets per dimension after normalization.
  3. Heatmaps and statistics for each residual Koopman matrix.

Outputs are saved under a user-specified directory (default: ./test/results).
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.append('src')
import utils.dataloader as dataloader_module  # noqa: E402
from models.koopman_phi_residual_trainer import (
    ResidualKoopmanModel,
    load_config as load_residual_config,
    create_dataloaders
)


def plot_bar(values: np.ndarray, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(values)), values)
    plt.title(title)
    plt.xlabel("Dimension")
    plt.ylabel("Variance")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved bar plot to {path}")


def plot_heatmap(matrix: np.ndarray, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    im = plt.imshow(matrix, cmap="coolwarm", aspect="auto")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("Latent dim")
    plt.ylabel("Latent dim")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze residual Koopman model internals.")
    parser.add_argument("--config", type=str, required=True, help="Residual Koopman config JSON.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path.")
    parser.add_argument("--output_dir", type=str, default="./test/results",
                        help="Directory to store diagnostic plots.")
    parser.add_argument("--batch_samples", type=int, default=4,
                        help="Number of batches to sample for statistics.")
    args = parser.parse_args()

    cfg = load_residual_config(args.config)
    data_cfg = cfg["data"]
    train_loader, _ = create_dataloaders(data_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg["model"]
    model = ResidualKoopmanModel(
        input_dim=model_cfg["input_dim"],
        latent_dim=model_cfg["latent_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_blocks=model_cfg["num_blocks"],
        num_classes=model_cfg.get("num_classes", 6)
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    residual_mean = checkpoint["residual_mean"].to(device)
    residual_std = checkpoint["residual_std"].to(device)

    phi_values = []
    target_values = []
    sampled = 0
    model.eval()
    with torch.no_grad():
        for x_batch, _, p_batch in train_loader:
            x_batch = x_batch.to(device)
            p_batch = p_batch.to(device)
            x_flat = x_batch[:, :-1, :].reshape(-1, x_batch.size(2))
            y_flat = x_batch[:, 1:, :].reshape(-1, x_batch.size(2))
            phi_x = model.encode(x_flat).cpu()
            phi_y = model.encode(y_flat)
            base_pred = model.apply_base(phi_x.to(device))
            target = (phi_y - base_pred - residual_mean) / residual_std
            phi_values.append(phi_x.numpy())
            target_values.append(target.cpu().numpy())
            sampled += 1
            if sampled >= args.batch_samples:
                break

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phi_arr = np.concatenate(phi_values, axis=0)
    target_arr = np.concatenate(target_values, axis=0)
    phi_var = phi_arr.var(axis=0)
    target_var = target_arr.var(axis=0)

    np.save(output_dir / "phi_variance.npy", phi_var)
    np.save(output_dir / "target_variance.npy", target_var)
    print(f"Saved variance arrays under {output_dir}")

    plot_bar(phi_var, "Phi dimension variance", output_dir / "phi_variance.png")
    plot_bar(target_var, "Residual target variance", output_dir / "target_variance.png")

    koopman_np = model.residual_K.detach().cpu().numpy()
    for idx in range(koopman_np.shape[0]):
        plot_heatmap(koopman_np[idx], f"Residual K_{idx}", output_dir / f"residual_K_{idx}.png")

    norms = np.linalg.norm(koopman_np, axis=(1, 2))
    print("Residual K Frobenius norms per class:")
    for idx, val in enumerate(norms):
        print(f"  Class {idx}: {val:.6f}")


if __name__ == "__main__":
    main()
