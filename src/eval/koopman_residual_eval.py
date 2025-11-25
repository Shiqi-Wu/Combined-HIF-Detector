#!/usr/bin/env python3
"""
Evaluation for the residual Koopman Phi model.

Produces both summary metrics and window-length trends compatible with
tools/plot_method_comparison.py.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.dataloader as dataloader_module  # noqa: E402
from models.koopman_phi_residual_trainer import (
    ResidualKoopmanModel,
    load_config as load_residual_config,
    create_dataloaders,
    compute_residual_stats
)


def determine_window_lengths(max_len: int, requested: Optional[Sequence[int]]) -> List[int]:
    if max_len < 2:
        return []
    if requested:
        lengths = {min(max_len, max(2, int(val))) for val in requested}
    else:
        lengths = {max_len}
    return sorted(lengths)


def prepare_pairs(x_batch: torch.Tensor, p_batch: torch.Tensor):
    seq_len = x_batch.size(1)
    x_t = x_batch[:, :-1, :]
    y_t = x_batch[:, 1:, :]
    batch = x_batch.size(0)
    pairs = seq_len - 1
    x_flat = x_t.reshape(batch * pairs, -1)
    y_flat = y_t.reshape(batch * pairs, -1)
    p_flat = p_batch.unsqueeze(1).expand(-1, pairs, -1).reshape(batch * pairs, -1)
    return x_flat, y_flat, p_flat, seq_len


def compute_errors(model: ResidualKoopmanModel, x_batch: torch.Tensor,
                   p_batch: torch.Tensor, device: torch.device,
                   residual_mean: torch.Tensor, residual_std: torch.Tensor):
    x_flat, y_flat, p_flat, seq_len = prepare_pairs(x_batch, p_batch)
    phi_x = model.encode(x_flat.to(device))
    with torch.no_grad():
        phi_y = model.encode(y_flat.to(device))
        base_pred = model.apply_base(phi_x)
        target = (phi_y - base_pred - residual_mean) / residual_std
    pred = torch.einsum('cij,nj->cni', model.residual_K, phi_x)
    target = target.unsqueeze(0)

    num_classes = model.num_classes
    batch = x_batch.size(0)
    pairs = seq_len - 1
    pred = pred.view(num_classes, batch, pairs, -1)
    target = target.view(1, batch, pairs, -1)
    diff = (pred - target) ** 2
    err = diff.mean(dim=3)  # (num_classes, batch, pairs)
    err = err.permute(1, 0, 2)  # (batch, num_classes, pairs)

    denom = torch.arange(1, pairs + 1, device=device, dtype=err.dtype).view(1, 1, -1)
    prefix = err.cumsum(dim=2) / denom
    final = prefix[:, :, -1]
    return final, prefix, seq_len


def project_simplex(v: torch.Tensor) -> torch.Tensor:
    """Project vector v onto the probability simplex."""
    if v.numel() == 0:
        return v
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1
    arange = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
    cond = u - cssv / arange > 0
    if not torch.any(cond):
        w = torch.ones_like(v) / v.numel()
        return w
    rho = torch.nonzero(cond, as_tuple=False)[-1, 0]
    theta = cssv[rho] / arange[rho]
    w = torch.clamp(v - theta, min=0)
    sum_w = w.sum()
    if sum_w > 0:
        w = w / sum_w
    else:
        w = torch.ones_like(v) / v.numel()
    return w


def solve_p_for_sequence(residual_K: torch.Tensor,
                         phi_seq: torch.Tensor,
                         target_seq: torch.Tensor,
                         reg_scale: float = 1e-2,
                         debug: bool = True) -> torch.Tensor:
    """
    Robust solver for p on the simplex minimizing ||K(p)Phi(x) - target||².
    Adds normalization, adaptive regularization, and clamping for stability.
    """

    num_classes = residual_K.size(0)
    device = residual_K.device
    dtype = residual_K.dtype

    # Compute projections
    basis = torch.einsum('cij,pj->cpi', residual_K, phi_seq)  # (C, pairs, latent)
    S = torch.einsum('cpi,dpi->cd', basis, basis)             # (C, C)
    t = torch.einsum('cpi,pi->c', basis, target_seq)          # (C,)

    # Normalize to control magnitude
    S_norm = S.norm() + 1e-8
    t_norm = t.norm() + 1e-8
    S = S / S_norm
    t = t / t_norm

    # Adaptive regularization
    reg_lambda = reg_scale * S.diag().mean().item()
    S = S + reg_lambda * torch.eye(num_classes, device=device, dtype=dtype)

    # Solve linear system
    try:
        L = torch.linalg.cholesky(S)
        p = torch.cholesky_solve(t.unsqueeze(1), L).squeeze(1)
        solver_used = "cholesky"
    except RuntimeError:
        p = torch.linalg.pinv(S) @ t
        solver_used = "pinv"

    # Clean up numerical issues
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = torch.clamp(p, min=-10.0, max=10.0)  # prevent blow-up

    # Project to simplex (sum=1, non-negative)
    p_raw = p.clone()
    p = project_simplex(p)

    if debug:
        with torch.no_grad():
            cond = torch.linalg.cond(S).item() if torch.all(torch.isfinite(S)) else float('inf')
            norms = [torch.norm(K).item() for K in residual_K]
            print(f"\n[solve_p_for_sequence debug]")
            print(f"  Solver: {solver_used}")
            print(f"  Condition number(S): {cond:.3e}")
            print(f"  Residual_K Frobenius norms: {[round(n, 4) for n in norms]}")
            print(f"  t vector: min={t.min().item():.3e}, max={t.max().item():.3e}")
            print(f"  Raw p before projection: {p_raw.cpu().numpy()}")
            print(f"  Projected p: {p.cpu().numpy()}\n")

    return p





def compute_topk_from_prob(probs: torch.Tensor, labels: torch.Tensor,
                           topk: Sequence[int]) -> Dict[int, float]:
    max_k = max(topk)
    _, rankings = torch.topk(probs, k=max_k, dim=1)
    metrics = {}
    for k in topk:
        hits = (rankings[:, :k] == labels.unsqueeze(1)).any(dim=1).float().mean().item()
        metrics[k] = hits
    return metrics


def compute_topk_from_errors(errors: torch.Tensor, labels: torch.Tensor,
                             topk: Sequence[int]) -> Dict[int, float]:
    max_k = max(topk)
    _, rankings = torch.topk(-errors, k=max_k, dim=1)
    metrics = {}
    for k in topk:
        hits = (rankings[:, :k] == labels.unsqueeze(1)).any(dim=1).float().mean().item()
        metrics[k] = hits
    return metrics


def evaluate_split(model: ResidualKoopmanModel, loader: DataLoader, device: torch.device,
                   residual_mean: torch.Tensor, residual_std: torch.Tensor,
                   topk: Sequence[int]) -> Tuple[Dict[int, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    total = 0
    agg = {k: 0.0 for k in topk}
    prefix_list, label_list, final_list, prob_list = [], [], [], []
    model.eval()

    with torch.no_grad():
        for x_batch, _, p_batch in loader:
            x_batch = x_batch.to(device)
            p_batch = p_batch.to(device)
            labels = torch.argmax(p_batch, dim=1)

            x_flat, y_flat, _, seq_len = prepare_pairs(x_batch, p_batch)
            phi_x_all = model.encode(x_flat.to(device))
            phi_y_all = model.encode(y_flat.to(device))
            base_pred = model.apply_base(phi_x_all)
            target_all = (phi_y_all - base_pred - residual_mean) / residual_std

            batch_size = labels.size(0)
            pairs = seq_len - 1
            latent_dim = phi_x_all.size(1)

            phi_x_seq = phi_x_all.view(batch_size, pairs, latent_dim)
            target_seq = target_all.view(batch_size, pairs, latent_dim)

            pred_all = torch.einsum('cij,nj->cni', model.residual_K, phi_x_all)
            pred_all = pred_all.view(model.num_classes, batch_size, pairs, latent_dim)
            diff = (pred_all - target_seq.unsqueeze(0)) ** 2
            err = diff.mean(dim=3).permute(1, 0, 2)
            denom = torch.arange(1, pairs + 1, device=device, dtype=err.dtype).view(1, 1, -1)
            prefix = err.cumsum(dim=2) / denom
            final = prefix[:, :, -1]

            prefix_list.append(prefix.cpu())
            final_list.append(final.cpu())
            label_list.append(labels.cpu())

            probs = []
            for i in range(batch_size):
                p_vec = solve_p_for_sequence(model.residual_K, phi_x_seq[i], target_seq[i])
                probs.append(p_vec)
                if len(prob_list) == 0 and i < 5:  # 打印前5个样本
                    print(f"\nSample {i}:")
                    print(f"  label = {labels[i].item()}")
                    print(f"  p = {p_vec.cpu().numpy()}")
            probs = torch.stack(probs, dim=0)
            prob_list.append(probs.cpu())

            metrics = compute_topk_from_prob(probs, labels, topk)
            for k, val in metrics.items():
                agg[k] += val * batch_size
            total += batch_size

    topk_scores = {k: agg[k] / total for k in topk}
    prefix_tensor = torch.cat(prefix_list, dim=0) if prefix_list else torch.empty(0)
    label_tensor = torch.cat(label_list, dim=0) if label_list else torch.empty(0)
    final_tensor = torch.cat(final_list, dim=0) if final_list else torch.empty(0)

    # 保存所有p与标签
    if prob_list:
        all_probs = torch.cat(prob_list, dim=0)
        df = pd.DataFrame(all_probs.numpy(), columns=[f"class_{i}" for i in range(all_probs.size(1))])
        df["label"] = label_tensor.numpy()
        os.makedirs("results", exist_ok=True)
        df.to_csv("results/p_analysis.csv", index=False)
        print("Saved p-vectors with labels to results/p_analysis.csv")

    return topk_scores, prefix_tensor, label_tensor, final_tensor


def window_metrics(prefix_errors: torch.Tensor, labels: torch.Tensor, lengths: List[int],
                   topk: Sequence[int], seq_len: int) -> Dict[int, Dict[int, float]]:
    metrics = {}
    if prefix_errors.numel() == 0:
        return metrics
    for length in lengths:
        if length > seq_len or length < 2:
            continue
        idx = length - 2
        window_err = prefix_errors[:, :, idx]
        metrics[length] = compute_topk_from_errors(window_err, labels, topk)
    return metrics


def save_summary(train_scores: Dict[int, float], val_scores: Dict[int, float],
                 topk: Sequence[int], output_path: Path) -> None:
    rows = {
        "dataset": ["train", "val"],
        "loss": [np.nan, np.nan]
    }
    for k in topk:
        rows[f"top{k}_acc"] = [train_scores.get(k, 0.0), val_scores.get(k, 0.0)]
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved summary metrics to {output_path}")


def save_window_csv(metrics: Dict[str, Dict[int, Dict[int, float]]],
                    topk: Sequence[int], output_path: Path) -> None:
    rows = []
    for dataset, length_dict in metrics.items():
        for length, scores in length_dict.items():
            row = {"dataset": dataset, "window_length": length}
            for k in topk:
                row[f"top{k}_acc"] = scores.get(k, 0.0)
            rows.append(row)
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved window metrics to {output_path}")


def extract_pos_neg(final_errors: torch.Tensor, labels: torch.Tensor):
    if final_errors.numel() == 0:
        return np.array([]), np.array([])
    idx = torch.arange(final_errors.size(0))
    pos = final_errors[idx, labels].detach().cpu().numpy()
    mask = torch.ones_like(final_errors, dtype=torch.bool)
    mask[idx, labels] = False
    neg = final_errors[mask].view(final_errors.size(0), -1).detach().cpu().numpy().reshape(-1)
    return pos, neg


def plot_error_distributions(final_errors: torch.Tensor, labels: torch.Tensor,
                             output_path: str, title: str) -> None:
    pos, neg = extract_pos_neg(final_errors, labels)
    if pos.size == 0 or neg.size == 0:
        print("Skipping error distribution plot due to insufficient data.")
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(6, 4))
    bins = 50
    plt.hist(pos, bins=bins, alpha=0.6, label="True class", density=True)
    plt.hist(neg, bins=bins, alpha=0.4, label="Other classes", density=True)
    plt.xlabel("Residual error")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved error distribution to {output_path}")


def summarize_errors(split: str, final_errors: torch.Tensor, labels: torch.Tensor) -> None:
    pos, neg = extract_pos_neg(final_errors, labels)
    if pos.size == 0 or neg.size == 0:
        print(f"{split}: not enough data for error summary.")
        return
    stats = {
        "pos_mean": float(pos.mean()),
        "pos_std": float(pos.std()),
        "pos_median": float(np.median(pos)),
        "neg_mean": float(neg.mean()),
        "neg_std": float(neg.std()),
        "neg_median": float(np.median(neg))
    }
    print(f"{split} error summary:")
    for key, val in stats.items():
        print(f"  {key}: {val:.6f}")


def plot_residual_heatmaps(koopman_mats: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    num_classes = koopman_mats.shape[0]
    for idx in range(num_classes):
        plt.figure(figsize=(5, 4))
        im = plt.imshow(koopman_mats[idx], cmap="coolwarm", aspect="auto")
        plt.title(f"Residual K_{idx}")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Latent dim")
        plt.ylabel("Latent dim")
        path = output_dir / f"residual_K_{idx}.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved residual heatmap to {path}")


def plot_residual_distance(koopman_mats: np.ndarray, output_path: Path) -> None:
    num_classes = koopman_mats.shape[0]
    dist = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            dist[i, j] = np.linalg.norm(koopman_mats[i] - koopman_mats[j], ord="fro")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    im = plt.imshow(dist, cmap="viridis")
    plt.title("Residual K pairwise distance")
    plt.xlabel("Class")
    plt.ylabel("Class")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved residual distance heatmap to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate residual Koopman model.")
    parser.add_argument("--config", type=str, required=True, help="Residual Koopman config JSON.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path.")
    parser.add_argument("--window_lengths", type=int, nargs='*', default=None,
                        help="Window lengths for trend analysis.")
    parser.add_argument("--topk", type=int, nargs='+', default=[1, 2, 3],
                        help="Top-k metrics to compute.")
    parser.add_argument("--output_prefix", type=str, default="./results/koopman_residual",
                        help="Prefix for output CSV files.")
    parser.add_argument("--error_figure_prefix", type=str,
                        default="./results/koopman_residual_errors",
                        help="Prefix for error distribution plots.")
    parser.add_argument("--heatmap_dir", type=str,
                        default="./results/koopman_residual_matrices",
                        help="Directory for Koopman residual heatmaps.")
    args = parser.parse_args()

    cfg = load_residual_config(args.config)
    data_cfg = cfg["data"]
    train_loader, val_loader = create_dataloaders(data_cfg)

    seq_len = train_loader.dataset[0][0].shape[0]
    lengths = determine_window_lengths(seq_len, args.window_lengths)

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

    # Recompute normalization stats per split and clamp std
    train_mean, train_std = compute_residual_stats(model, train_loader, device)
    val_mean, val_std = compute_residual_stats(model, val_loader, device)
    train_std = torch.clamp(train_std, min=1e-3).to(device)
    val_std = torch.clamp(val_std, min=1e-3).to(device)
    train_mean = train_mean.to(device)
    val_mean = val_mean.to(device)

    topk = tuple(args.topk)
    train_scores, train_prefix, train_labels, train_final = evaluate_split(
        model, train_loader, device, train_mean, train_std, topk
    )
    val_scores, val_prefix, val_labels, val_final = evaluate_split(
        model, val_loader, device, val_mean, val_std, topk
    )

    summary_path = Path(args.output_prefix).with_suffix(".csv")
    window_path = summary_path.with_name(summary_path.stem + "_window.csv")
    save_summary(train_scores, val_scores, topk, summary_path)

    metrics = {
        "train": window_metrics(train_prefix, train_labels, lengths, topk, seq_len),
        "val": window_metrics(val_prefix, val_labels, lengths, topk, seq_len)
    }
    save_window_csv(metrics, topk, window_path)

    plot_error_distributions(train_final, train_labels,
                             args.error_figure_prefix + "_train.png",
                             "Residual model - Train error distribution")
    plot_error_distributions(val_final, val_labels,
                             args.error_figure_prefix + "_val.png",
                             "Residual model - Validation error distribution")
    summarize_errors("Train", train_final, train_labels)
    summarize_errors("Validation", val_final, val_labels)

    koopman_dir = Path(args.heatmap_dir)
    plot_residual_heatmaps(model.residual_K.detach().cpu().numpy(), koopman_dir)
    dist_path = koopman_dir / "pairwise_distance.png"
    plot_residual_distance(model.residual_K.detach().cpu().numpy(), dist_path)


if __name__ == "__main__":
    main()
