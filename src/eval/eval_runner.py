#!/usr/bin/env python3
"""
Unified Evaluation Runner.

Supports LSTM, dynamic-system, and Koopman-based classifiers.
Besides overall metrics, this runner also reports top-k accuracy trends
as a function of trajectory length (window size).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append("./src")
import utils.dataloader as dataloader_module
from models.fault_lstm_classifier import LSTMClassifier
from dynamic_eval import KnownControlClassifier, calculate_top_k_accuracy, compute_loss_from_logits
from dynamic_eval import DynamicSystemDataset
from koopman_inverse_eval import (
    KoopmanPhiModel,
    load_config as load_koopman_config,
    build_scaled_datasets as build_koopman_dataloaders,
    load_checkpoint as load_koopman_checkpoint,
    evaluate_split as koopman_evaluate_split,
    compute_class_errors as koopman_compute_class_errors
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def determine_window_lengths(max_len: int, requested: Optional[Sequence[int]]) -> List[int]:
    """Determine window lengths (>=2) to evaluate."""
    if max_len < 2:
        return []

    if requested:
        lengths = {min(max_len, max(2, int(val))) for val in requested}
    else:
        num_points = min(5, max_len - 1) or 1
        fractions = np.linspace(1 / (num_points + 1), 1.0, num=num_points, endpoint=True)
        lengths = {min(max_len, max(2, int(round(max_len * frac)))) for frac in fractions}
        lengths.add(max_len)

    return sorted(lengths)


def init_topk_tracker(lengths: Sequence[int], topk_vals: Sequence[int]) -> Tuple[Dict[int, Dict[int, float]], Dict[int, int]]:
    tracker = {length: {k: 0.0 for k in topk_vals} for length in lengths}
    totals = {length: 0 for length in lengths}
    return tracker, totals


def update_topk_tracker(scores: torch.Tensor, labels: torch.Tensor, length: int,
                        tracker: Dict[int, Dict[int, float]], totals: Dict[int, int],
                        topk_vals: Sequence[int]) -> None:
    """Accumulate top-k hits for a specific window length."""
    if length not in tracker or scores.numel() == 0:
        return
    max_k = max(topk_vals)
    _, rankings = torch.topk(scores, k=max_k, dim=1, largest=True)
    for k in topk_vals:
        hits = (rankings[:, :k] == labels.unsqueeze(1)).any(dim=1).float().sum().item()
        tracker[length][k] += hits
    totals[length] += labels.size(0)


def finalize_tracker(tracker: Dict[int, Dict[int, float]], totals: Dict[int, int]) -> Dict[int, Dict[int, float]]:
    metrics = {}
    for length, counts in tracker.items():
        total = totals.get(length, 0)
        if total == 0:
            continue
        metrics[length] = {k: counts[k] / total for k in counts}
    return metrics


def metrics_to_rows(metrics: Dict[int, Dict[int, float]], dataset_name: str,
                    topk_vals: Sequence[int]) -> List[Dict[str, float]]:
    rows = []
    for length in sorted(metrics.keys()):
        row = {"dataset": dataset_name, "window_length": length}
        for k in topk_vals:
            row[f"top{k}_acc"] = metrics[length].get(k, 0.0)
        rows.append(row)
    return rows


def save_overall_results(train_loss: float, train_scores: Dict[int, float],
                         val_loss: float, val_scores: Dict[int, float],
                         topk_vals: Sequence[int], save_path: Path) -> None:
    results = {
        "dataset": ["train", "val"],
        "loss": [train_loss, val_loss]
    }
    for k in topk_vals:
        results[f"top{k}_acc"] = [train_scores.get(k, 0.0), val_scores.get(k, 0.0)]
    df = pd.DataFrame(results)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved summary metrics to {save_path}")
    print(df)


def save_window_rows(rows: List[Dict[str, float]], save_path: Path) -> None:
    if not rows:
        print(f"No windowed metrics to save for {save_path}, skipping.")
        return
    df = pd.DataFrame(rows)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved windowed metrics to {save_path}")
    print(df.head())


def extract_numpy_sequences(dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a dataset into numpy arrays for x, u, and labels."""
    x_list, u_list, labels = [], [], []
    for i in range(len(dataset)):
        x, u, p = dataset[i]
        x_list.append(x.numpy())
        u_list.append(u.numpy())
        labels.append(int(torch.argmax(p).item()))
    return np.array(x_list), np.array(u_list), np.array(labels)


# ---------------------------------------------------------------------------
# LSTM evaluation
# ---------------------------------------------------------------------------
def lstm_window_trend(model: nn.Module, loader: DataLoader, device: torch.device,
                      lengths: Sequence[int], topk_vals: Sequence[int]) -> Dict[int, Dict[int, float]]:
    tracker, totals = init_topk_tracker(lengths, topk_vals)
    max_k = max(topk_vals)
    with torch.no_grad():
        for x, _, p in loader:
            seq_len = x.size(1)
            x = x.to(device).to(torch.float64)
            targets = torch.argmax(p, dim=1).to(device)
            for length in lengths:
                if length > seq_len or length < 2:
                    continue
                outputs = model(x[:, :length, :])
                update_topk_tracker(outputs, targets, length, tracker, totals, topk_vals)
    return finalize_tracker(tracker, totals)


def evaluate_lstm(config: Dict, model_path: str, save_csv: Path, window_csv: Path,
                  requested_lengths: Optional[Sequence[int]], topk_vals: Sequence[int]) -> None:
    data_cfg = config["data"]
    train_dataset, val_dataset = dataloader_module.load_dataset_from_folder(
        data_dir=data_cfg["path"], config=data_cfg
    )
    scaled_train = dataloader_module.ScaledDataset(train_dataset, pca_dim=data_cfg.get("pca_dim", 2), fit_scalers=True)
    scaled_val = dataloader_module.ScaledDataset(val_dataset, pca_dim=data_cfg.get("pca_dim", 2))
    scaled_val.set_preprocessing_params(scaled_train.get_preprocessing_params())

    train_loader = DataLoader(scaled_train, batch_size=config["train"]["batch"], shuffle=False)
    val_loader = DataLoader(scaled_val, batch_size=config["train"]["batch"], shuffle=False)

    seq_len = scaled_train[0][0].shape[0]
    window_lengths = determine_window_lengths(seq_len, requested_lengths)
    print(f"LSTM window lengths: {window_lengths}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(config=config["model"]).to(device).to(torch.float64)
    model.load_state_dict(torch.load(model_path, map_location=device))
    criterion = nn.CrossEntropyLoss()

    max_k = max(topk_vals)

    def eval_epoch(loader: DataLoader) -> Tuple[float, Dict[int, float]]:
        model.eval()
        total_loss, total = 0.0, 0
        topk_correct = {k: 0.0 for k in topk_vals}
        with torch.no_grad():
            for x, _, p in loader:
                x = x.to(device).to(torch.float64)
                targets = torch.argmax(p, dim=1).to(device)
                outputs = model(x)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)
                total += targets.size(0)
                _, pred = outputs.topk(max_k, dim=1, largest=True, sorted=True)
                for k in topk_vals:
                    hits = (pred[:, :k] == targets.view(-1, 1)).any(dim=1).float().sum().item()
                    topk_correct[k] += hits
        avg_loss = total_loss / total
        avg_accs = {k: topk_correct[k] / total for k in topk_vals}
        return avg_loss, avg_accs

    train_loss, train_scores = eval_epoch(train_loader)
    val_loss, val_scores = eval_epoch(val_loader)

    save_overall_results(train_loss, train_scores, val_loss, val_scores, topk_vals, save_csv)

    window_rows: List[Dict[str, float]] = []
    if window_lengths:
        window_rows.extend(metrics_to_rows(
            lstm_window_trend(model, train_loader, device, window_lengths, topk_vals),
            "train", topk_vals
        ))
        window_rows.extend(metrics_to_rows(
            lstm_window_trend(model, val_loader, device, window_lengths, topk_vals),
            "val", topk_vals
        ))
    save_window_rows(window_rows, window_csv)


# ---------------------------------------------------------------------------
# Dynamic evaluation
# ---------------------------------------------------------------------------
def dynamic_window_trend(classifier: KnownControlClassifier, x_data: np.ndarray, u_data: np.ndarray,
                         labels: np.ndarray, lengths: Sequence[int],
                         topk_vals: Sequence[int]) -> Dict[int, Dict[int, float]]:
    metrics = {}
    torch_labels = torch.tensor(labels, dtype=torch.long)
    for length in lengths:
        if length < 2:
            continue
        trunc_len = min(length, x_data.shape[1])
        errors = []
        for i in range(x_data.shape[0]):
            _, err = classifier.predict_single_sequence(
                x_data[i, :trunc_len, :],
                u_data[i, :trunc_len, :]
            )
            errors.append(err)
        errors_tensor = torch.tensor(np.array(errors), dtype=torch.float64)
        max_k = max(topk_vals)
        _, rankings = torch.topk(-errors_tensor, k=max_k, dim=1, largest=True)
        metrics[length] = {}
        for k in topk_vals:
            hits = (rankings[:, :k] == torch_labels.unsqueeze(1)).any(dim=1).float().mean().item()
            metrics[length][k] = hits
    return metrics


def evaluate_dynamic(config: Dict, save_csv: Path, window_csv: Path,
                     requested_lengths: Optional[Sequence[int]], topk_vals: Sequence[int]) -> None:
    data_cfg = config["data"]
    train_dataset, val_dataset = dataloader_module.load_dataset_from_folder(
        data_dir=data_cfg["path"], config=data_cfg
    )
    scaled_train = dataloader_module.ScaledDataset(train_dataset, pca_dim=data_cfg.get("pca_dim", 2), fit_scalers=True)
    scaled_val = dataloader_module.ScaledDataset(val_dataset, pca_dim=data_cfg.get("pca_dim", 2))
    scaled_val.set_preprocessing_params(scaled_train.get_preprocessing_params())

    window_lengths = determine_window_lengths(scaled_train[0][0].shape[0], requested_lengths)
    print(f"Dynamic window lengths: {window_lengths}")

    train_dynamic = DynamicSystemDataset(scaled_train)
    val_dynamic = DynamicSystemDataset(scaled_val)

    classifier = KnownControlClassifier()
    classifier.fit_dynamic_system(train_dynamic, shared_K=True)

    train_results = classifier.evaluate_dataset(train_dynamic)
    val_results = classifier.evaluate_dataset(val_dynamic)

    train_topk = calculate_top_k_accuracy(train_results['prediction_errors'], train_results['true_labels'], topk_vals)
    val_topk = calculate_top_k_accuracy(val_results['prediction_errors'], val_results['true_labels'], topk_vals)
    train_loss = compute_loss_from_logits(train_results['prediction_errors'], train_results['true_labels'])
    val_loss = compute_loss_from_logits(val_results['prediction_errors'], val_results['true_labels'])

    save_overall_results(train_loss, train_topk, val_loss, val_topk, topk_vals, save_csv)

    window_rows: List[Dict[str, float]] = []
    if window_lengths:
        train_x, train_u, train_labels = extract_numpy_sequences(scaled_train)
        val_x, val_u, val_labels = extract_numpy_sequences(scaled_val)
        window_rows.extend(metrics_to_rows(
            dynamic_window_trend(classifier, train_x, train_u, train_labels, window_lengths, topk_vals),
            "train", topk_vals
        ))
        window_rows.extend(metrics_to_rows(
            dynamic_window_trend(classifier, val_x, val_u, val_labels, window_lengths, topk_vals),
            "val", topk_vals
        ))
    save_window_rows(window_rows, window_csv)


# ---------------------------------------------------------------------------
# Koopman evaluation
# ---------------------------------------------------------------------------
def koopman_window_trend(model: KoopmanPhiModel, loader: DataLoader, device: torch.device,
                         lengths: Sequence[int], topk_vals: Sequence[int]) -> Dict[int, Dict[int, float]]:
    tracker, totals = init_topk_tracker(lengths, topk_vals)
    with torch.no_grad():
        for x_batch, _, p_batch in loader:
            x_batch = x_batch.to(device)
            labels = torch.argmax(p_batch, dim=1).to(device)
            _, prefix_errors = koopman_compute_class_errors(model, x_batch, device, return_prefix=True)
            if prefix_errors is None:
                continue
            available_states = prefix_errors.size(2) + 1
            for length in lengths:
                if length > available_states or length < 2:
                    continue
                idx = length - 2  # transitions = length - 1
                errors_slice = prefix_errors[:, :, idx]
                update_topk_tracker(-errors_slice, labels, length, tracker, totals, topk_vals)
    return finalize_tracker(tracker, totals)


def evaluate_koopman(config_path: str, checkpoint_path: Optional[str], save_csv: Path,
                     window_csv: Path, requested_lengths: Optional[Sequence[int]],
                     topk_vals: Sequence[int]) -> None:
    koopman_cfg = load_koopman_config(config_path)
    data_cfg = koopman_cfg["data"]
    train_loader, val_loader = build_koopman_dataloaders(data_cfg)
    seq_len = train_loader.dataset[0][0].shape[0]
    window_lengths = determine_window_lengths(seq_len, requested_lengths)
    print(f"Koopman window lengths: {window_lengths}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = koopman_cfg["model"]
    model = KoopmanPhiModel(
        input_dim=model_cfg["input_dim"],
        latent_dim=model_cfg["latent_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_blocks=model_cfg["num_blocks"],
        num_classes=model_cfg.get("num_classes", 6)
    ).to(device)

    ckpt_path = checkpoint_path or koopman_cfg["train"]["save_path"]
    load_koopman_checkpoint(model, ckpt_path)
    print(f"Loaded Koopman checkpoint from {ckpt_path}")

    train_scores, train_pos, _ = koopman_evaluate_split(
        model, train_loader, device, topk_vals, collect_errors=True
    )
    val_scores, val_pos, _ = koopman_evaluate_split(
        model, val_loader, device, topk_vals, collect_errors=True
    )
    train_loss = float(train_pos.mean()) if train_pos is not None else float("nan")
    val_loss = float(val_pos.mean()) if val_pos is not None else float("nan")

    save_overall_results(train_loss, train_scores, val_loss, val_scores, topk_vals, save_csv)

    window_rows: List[Dict[str, float]] = []
    if window_lengths:
        window_rows.extend(metrics_to_rows(
            koopman_window_trend(model, train_loader, device, window_lengths, topk_vals),
            "train", topk_vals
        ))
        window_rows.extend(metrics_to_rows(
            koopman_window_trend(model, val_loader, device, window_lengths, topk_vals),
            "val", topk_vals
        ))
    save_window_rows(window_rows, window_csv)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------
def resolve_paths(base: Path, method: str, multi_method: bool) -> Tuple[Path, Path]:
    if multi_method:
        summary_path = base.with_name(f"{base.stem}_{method}{base.suffix}")
    else:
        summary_path = base
    window_path = summary_path.with_name(f"{summary_path.stem}_window{summary_path.suffix}")
    return summary_path, window_path


def main():
    parser = argparse.ArgumentParser(description="Unified Evaluation Runner with window-size trends")
    parser.add_argument('--config', type=str, default='config_lstm_classifier.json', help='Path to LSTM config file')
    parser.add_argument('--method', type=str, choices=['lstm', 'dynamic', 'koopman', 'all'], required=True,
                        help='Which evaluation method to run')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                        help='Path to trained LSTM model (for method=lstm)')
    parser.add_argument('--save_csv', type=str, default='evaluation_results.csv',
                        help='Base path for saving evaluation CSVs')
    parser.add_argument('--window_lengths', type=int, nargs='+', default=None,
                        help='Explicit window lengths to evaluate (default evenly spaced)')
    parser.add_argument('--topk', type=int, nargs='+', default=[1, 2, 3], help='Top-k values to compute')
    parser.add_argument('--koopman_config', type=str, default='configs/koopman_phi_config.json',
                        help='Config for Koopman Phi evaluation')
    parser.add_argument('--koopman_checkpoint', type=str, default=None,
                        help='Checkpoint path for Koopman Phi model (defaults to config train.save_path)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        lstm_config = json.load(f)

    methods = ['lstm', 'dynamic', 'koopman'] if args.method == 'all' else [args.method]
    base_path = Path(args.save_csv)

    for method in methods:
        summary_path, window_path = resolve_paths(base_path, method, len(methods) > 1)
        if method == 'lstm':
            if not os.path.exists(args.model_path):
                raise FileNotFoundError(f"LSTM model checkpoint not found: {args.model_path}")
            evaluate_lstm(lstm_config, args.model_path, summary_path, window_path, args.window_lengths, args.topk)
        elif method == 'dynamic':
            evaluate_dynamic(lstm_config, summary_path, window_path, args.window_lengths, args.topk)
        elif method == 'koopman':
            evaluate_koopman(args.koopman_config, args.koopman_checkpoint,
                             summary_path, window_path, args.window_lengths, args.topk)

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
