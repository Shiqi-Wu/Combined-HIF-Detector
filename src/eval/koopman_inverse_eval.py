#!/usr/bin/env python3
"""
Koopman inverse evaluation.

Given a trajectory x[t], we measure the residuals ||Phi(x[t+1]) - K(p)Phi(x[t])||
for every Koopman operator and pick the class with minimal residual. Results
are summarized as top-k accuracies for both the training and validation splits.
"""

import os
import sys
import json
import argparse
from typing import Dict, Sequence, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Project imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.dataloader as dataloader_module  # noqa: E402
from models.koopman_phi_trainer import (  # noqa: E402
    KoopmanPhiModel,
    load_config
)


def load_checkpoint(model: KoopmanPhiModel, checkpoint_path: str) -> Dict:
    """Load a Koopman Phi checkpoint and return metadata."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    return payload


def build_scaled_datasets(data_cfg: Dict):
    """
    Create scaled train/val datasets using saved preprocessing parameters.
    """
    preprocess_path = data_cfg.get("preprocessing_path")
    if not preprocess_path or not os.path.exists(preprocess_path):
        raise FileNotFoundError(
            "Preprocessing parameters not found. Ensure Koopman training "
            "saved them via data.preprocessing_path."
        )

    train_dataset, val_dataset = dataloader_module.load_dataset_from_folder(
        data_cfg["path"],
        data_cfg
    )

    scaled_train = dataloader_module.ScaledDataset(
        train_dataset,
        pca_dim=data_cfg.get("pca_dim", 2),
        fit_scalers=False
    )
    scaled_train.load_preprocessing_params(preprocess_path)

    scaled_val = dataloader_module.ScaledDataset(
        val_dataset,
        pca_dim=data_cfg.get("pca_dim", 2),
        fit_scalers=False
    )
    scaled_val.set_preprocessing_params(scaled_train.get_preprocessing_params())

    train_loader = DataLoader(
        scaled_train,
        batch_size=data_cfg.get("batch_size", 16),
        shuffle=False
    )
    val_loader = DataLoader(
        scaled_val,
        batch_size=data_cfg.get("batch_size", 16),
        shuffle=False
    )
    return train_loader, val_loader


def compute_class_errors(model: KoopmanPhiModel, x_batch: torch.Tensor,
                         device: torch.device, return_prefix: bool = False
                         ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute mean residual error per class for every trajectory in the batch.

    Args:
        model: Trained KoopmanPhiModel
        x_batch: State sequences (batch, seq_len, input_dim)
        device: Torch device
        return_prefix: Whether to return prefix-wise residuals for each length

    Returns:
        final_errors: Tensor (batch, num_classes) of mean residuals per class
        prefix_errors: Tensor (batch, num_classes, seq_len-1) of cumulative means
                       if return_prefix is True, else None
    """
    seq_len = x_batch.size(1)
    if seq_len < 2:
        raise ValueError("Sequence length must be >= 2 for Koopman evaluation.")

    x_t = x_batch[:, :-1, :].reshape(-1, x_batch.size(2))
    y_t = x_batch[:, 1:, :].reshape(-1, x_batch.size(2))

    phi_x = model.phi(x_t.to(device))
    phi_y = model.phi(y_t.to(device))

    koopman_mats = model.koopman_matrices  # (num_classes, latent_dim, latent_dim)
    predicted = torch.einsum('cij,nj->cni', koopman_mats, phi_x)
    residual = (predicted - phi_y.unsqueeze(0)) ** 2
    residual = residual.mean(dim=2)  # (num_classes, batch*(seq_len-1))
    residual = residual.view(koopman_mats.size(0), x_batch.size(0), seq_len - 1)

    cumulative = residual.cumsum(dim=2)
    denom = torch.arange(1, seq_len, device=device, dtype=residual.dtype).view(1, 1, -1)
    prefix_errors = (cumulative / denom).permute(1, 0, 2).contiguous()
    final_errors = prefix_errors[:, :, -1]

    if return_prefix:
        return final_errors, prefix_errors
    return final_errors, None


def topk_accuracy(errors: torch.Tensor, labels: torch.Tensor,
                  topk: Sequence[int]) -> Dict[int, float]:
    """
    Compute top-k accuracy from class errors (lower is better).
    """
    max_k = max(topk)
    _, rankings = torch.topk(-errors, k=max_k, dim=1)  # negate to mimic ascending sort
    metrics = {}
    for k in topk:
        hits = (rankings[:, :k] == labels.unsqueeze(1)).any(dim=1)
        metrics[k] = hits.float().mean().item()
    return metrics


def evaluate_split(model: KoopmanPhiModel, loader: DataLoader,
                   device: torch.device, topk: Sequence[int],
                   collect_errors: bool = False) -> Tuple[Dict[int, float],
                                                          Optional[np.ndarray],
                                                          Optional[np.ndarray]]:
    """Evaluate a dataloader and return top-k accuracies plus optional error arrays."""
    model.eval()
    total = 0
    agg = {k: 0.0 for k in topk}
    pos_errors = []
    neg_errors = []

    with torch.no_grad():
        for x_batch, _, p_batch in loader:
            x_batch = x_batch.to(device)
            p_batch = p_batch.to(device)
            labels = torch.argmax(p_batch, dim=1)

            errors, _ = compute_class_errors(model, x_batch, device)
            metrics = topk_accuracy(errors, labels, topk)

            if collect_errors:
                idx = torch.arange(errors.size(0), device=device)
                pos = errors[idx, labels].detach().cpu()
                mask = torch.ones_like(errors, dtype=torch.bool)
                mask[idx, labels] = False
                neg = errors[mask].view(errors.size(0), -1).detach().cpu()
                pos_errors.append(pos)
                neg_errors.append(neg)

            batch_size = x_batch.size(0)
            for k, val in metrics.items():
                agg[k] += val * batch_size
            total += batch_size

    metrics = {k: agg[k] / total for k in topk}
    if not collect_errors:
        return metrics, None, None

    pos_arr = torch.cat(pos_errors).numpy()
    neg_arr = torch.cat(neg_errors).reshape(-1).numpy()
    return metrics, pos_arr, neg_arr


def plot_topk(train_scores: Dict[int, float], val_scores: Dict[int, float],
              output_path: str) -> None:
    """Plot train/val top-k accuracy curves."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    topk = sorted(train_scores.keys())
    train_values = [train_scores[k] for k in topk]
    val_values = [val_scores[k] for k in topk]

    plt.figure(figsize=(6, 4))
    plt.plot(topk, train_values, marker='o', label='Train')
    plt.plot(topk, val_values, marker='s', label='Validation')
    plt.xlabel("Top-k")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.title("Koopman inverse evaluation")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved top-k plot to {output_path}")


def plot_koopman_heatmaps(koopman_mats: np.ndarray, output_dir: str,
                          title_prefix: str = "Koopman") -> None:
    """Plot individual Koopman matrix heatmaps."""
    os.makedirs(output_dir, exist_ok=True)
    num_classes = koopman_mats.shape[0]
    for idx in range(num_classes):
        mat = koopman_mats[idx]
        plt.figure(figsize=(5, 4))
        im = plt.imshow(mat, cmap="coolwarm", aspect="auto")
        plt.title(f"{title_prefix} K_{idx}")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Latent dim")
        plt.ylabel("Latent dim")
        out_path = os.path.join(output_dir, f"koopman_K_{idx}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved Koopman heatmap to {out_path}")


def plot_koopman_distance_heatmap(koopman_mats: np.ndarray, output_path: str,
                                  title: str = "Koopman matrix distance") -> None:
    """Plot pairwise Frobenius norm differences between Koopman matrices."""
    num_classes = koopman_mats.shape[0]
    dist = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            dist[i, j] = np.linalg.norm(koopman_mats[i] - koopman_mats[j], ord="fro")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(5, 4))
    im = plt.imshow(dist, cmap="viridis")
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Class")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Koopman distance heatmap to {output_path}")


def plot_error_distributions(pos_errors: np.ndarray, neg_errors: np.ndarray,
                             output_path: str, title: str) -> None:
    """Plot histogram/KDE-style comparison for positive vs negative errors."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(6, 4))
    bins = 50
    plt.hist(pos_errors, bins=bins, alpha=0.6, label="True class", density=True)
    plt.hist(neg_errors, bins=bins, alpha=0.4, label="Other classes", density=True)
    plt.xlabel("Residual error")
    plt.ylabel("Density")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved error distribution plot to {output_path}")


def summarize_errors(split: str, pos_errors: np.ndarray, neg_errors: np.ndarray) -> None:
    """Print summary statistics for positive vs negative class residuals."""
    summary = {
        "pos_mean": float(pos_errors.mean()),
        "pos_std": float(pos_errors.std()),
        "pos_median": float(np.median(pos_errors)),
        "neg_mean": float(neg_errors.mean()),
        "neg_std": float(neg_errors.std()),
        "neg_median": float(np.median(neg_errors))
    }
    print(f"{split} error summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.6e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Koopman inverse evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to Koopman config JSON.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path.")
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 3, 5], help="Top-k values.")
    parser.add_argument("--figure_path", type=str,
                        default="./results/koopman_phi_topk.png",
                        help="Where to save the accuracy plot.")
    parser.add_argument("--error_figure_prefix", type=str,
                        default="./results/koopman_phi_errors",
                        help="Prefix path for error distribution figures (train/val).")
    parser.add_argument("--koopman_heatmap_dir", type=str,
                        default="./results/koopman_matrices",
                        help="Directory to store Koopman matrix heatmaps.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    checkpoint_path = args.checkpoint or cfg["train"]["save_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg["model"]
    model = KoopmanPhiModel(
        input_dim=model_cfg["input_dim"],
        latent_dim=model_cfg["latent_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_blocks=model_cfg["num_blocks"],
        num_classes=model_cfg.get("num_classes", 6)
    ).to(device)

    load_checkpoint(model, checkpoint_path)
    print(f"Loaded Koopman checkpoint from {checkpoint_path}")

    train_loader, val_loader = build_scaled_datasets(cfg["data"])
    topk = sorted(set(args.topk))
    print(f"Evaluating top-k metrics for k={topk}")

    train_scores, train_pos, train_neg = evaluate_split(
        model, train_loader, device, topk, collect_errors=True
    )
    val_scores, val_pos, val_neg = evaluate_split(
        model, val_loader, device, topk, collect_errors=True
    )

    print("Train split accuracies:")
    for k, val in train_scores.items():
        print(f"  Top-{k}: {val:.4f}")
    print("Validation split accuracies:")
    for k, val in val_scores.items():
        print(f"  Top-{k}: {val:.4f}")

    plot_topk(train_scores, val_scores, args.figure_path)

    train_error_fig = f"{args.error_figure_prefix}_train.png"
    val_error_fig = f"{args.error_figure_prefix}_val.png"
    plot_error_distributions(train_pos, train_neg, train_error_fig, "Train error distribution")
    plot_error_distributions(val_pos, val_neg, val_error_fig, "Validation error distribution")
    summarize_errors("Train", train_pos, train_neg)
    summarize_errors("Validation", val_pos, val_neg)

    koopman_np = model.koopman_matrices.detach().cpu().numpy()
    plot_koopman_heatmaps(koopman_np, args.koopman_heatmap_dir)
    dist_path = os.path.join(args.koopman_heatmap_dir, "koopman_pairwise_distance.png")
    plot_koopman_distance_heatmap(koopman_np, dist_path)


if __name__ == "__main__":
    main()
