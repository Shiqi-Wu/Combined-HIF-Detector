#!/usr/bin/env python3
"""
Unified Evaluation Runner for LSTM and Dynamic System Classifiers

Usage:
    python eval_runner.py --config config_lstm_classifier.json --method lstm --save_csv lstm_eval.csv
    python eval_runner.py --config config_lstm_classifier.json --method dynamic --save_csv dynamic_eval.csv
"""

import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
import numpy as np

# ===== Import your modules =====
import sys
import os
sys.path.append("./src")
import utils.dataloader as dataloader_module
from models.fault_lstm_classifier import LSTMClassifier
from dynamic_eval import KnownControlClassifier, calculate_top_k_accuracy, compute_loss_from_logits
from dynamic_eval import DynamicSystemDataset


# ---------- LSTM Evaluation ----------
def evaluate_lstm(config, model_path, save_csv):
    # Load dataset
    data_cfg = config["data"]
    train_dataset, val_dataset = dataloader_module.load_dataset_from_folder(
        data_dir=data_cfg["path"], config=data_cfg
    )
    scaled_train = dataloader_module.ScaledDataset(train_dataset, pca_dim=data_cfg.get("pca_dim", 2), fit_scalers=True)
    scaled_val = dataloader_module.ScaledDataset(val_dataset, pca_dim=data_cfg.get("pca_dim", 2))
    scaled_val.set_preprocessing_params(scaled_train.get_preprocessing_params())

    train_loader = DataLoader(scaled_train, batch_size=config["train"]["batch"], shuffle=False)
    val_loader = DataLoader(scaled_val, batch_size=config["train"]["batch"], shuffle=False)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(config=config["model"]).to(device).to(torch.float64)
    model.load_state_dict(torch.load(model_path, map_location=device))
    criterion = nn.CrossEntropyLoss()

    def eval_epoch(loader):
        model.eval()
        total_loss, total_correct, total = 0, 0, 0
        topk_sums = {k: 0 for k in [1,2,3,4,5]}
        with torch.no_grad():
            for x, _, p in loader:
                x = x.to(device).to(torch.float64)
                target = torch.argmax(p, dim=1).to(device)
                outputs = model(x)
                loss = criterion(outputs, target)
                total_loss += loss.item() * target.size(0)
                total_correct += (outputs.argmax(1) == target).sum().item()
                total += target.size(0)
                for k in topk_sums:
                    _, pred = outputs.topk(k, 1, True, True)
                    correct = pred.eq(target.view(-1,1).expand_as(pred))
                    topk_sums[k] += correct.sum().item()
        avg_loss = total_loss / total
        avg_accs = [topk_sums[k] / total for k in [1,2,3,4,5]]
        return avg_loss, avg_accs

    train_loss, train_accs = eval_epoch(train_loader)
    val_loss, val_accs = eval_epoch(val_loader)

    results = {
        "dataset": ["train", "val"],
        "loss": [train_loss, val_loss]
    }
    for i, k in enumerate([1,2,3,4,5]):
        results[f"top{k}_acc"] = [train_accs[i], val_accs[i]]

    df = pd.DataFrame(results)
    save_csv = Path(save_csv)
    save_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_csv, index=False)
    print(f"LSTM evaluation saved to {save_csv}")
    print(df)


# ---------- Dynamic Evaluation ----------
def evaluate_dynamic(config, save_csv):
    # 只用 lstm config 的 data 来生成数据
    data_cfg = config["data"]
    train_dataset, val_dataset = dataloader_module.load_dataset_from_folder(
        data_dir=data_cfg["path"], config=data_cfg
    )
    scaled_train = dataloader_module.ScaledDataset(train_dataset, pca_dim=data_cfg.get("pca_dim", 2), fit_scalers=True)
    scaled_val = dataloader_module.ScaledDataset(val_dataset, pca_dim=data_cfg.get("pca_dim", 2))
    scaled_val.set_preprocessing_params(scaled_train.get_preprocessing_params())

    # 转成 dynamic dataset
    train_dynamic = DynamicSystemDataset(scaled_train)
    val_dynamic = DynamicSystemDataset(scaled_val)

    # Fit classifier
    classifier = KnownControlClassifier()
    classifier.fit_dynamic_system(train_dynamic, shared_K=True)

    # Eval
    train_results = classifier.evaluate_dataset(train_dynamic)
    val_results = classifier.evaluate_dataset(val_dynamic)

    k_values = [1,2,3,4,5]
    train_topk = calculate_top_k_accuracy(train_results['prediction_errors'], train_results['true_labels'], k_values)
    val_topk = calculate_top_k_accuracy(val_results['prediction_errors'], val_results['true_labels'], k_values)

    train_loss = compute_loss_from_logits(train_results['prediction_errors'], train_results['true_labels'])
    val_loss = compute_loss_from_logits(val_results['prediction_errors'], val_results['true_labels'])

    results = {
        "dataset": ["train", "val"],
        "loss": [train_loss, val_loss]
    }
    for k in k_values:
        results[f"top{k}_acc"] = [float(train_topk.get(k, 0.0)), float(val_topk.get(k, 0.0))]

    df = pd.DataFrame(results)
    save_csv = Path(save_csv)
    save_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_csv, index=False)
    print(f"Dynamic evaluation saved to {save_csv}")
    print(df)


# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Evaluation Runner")
    parser.add_argument('--config', type=str, default='config_lstm_classifier.json', help='Path to LSTM config file')
    parser.add_argument('--method', type=str, choices=['lstm', 'dynamic'], required=True,
                        help='Which evaluation method to run')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                        help='Path to trained LSTM model (only for method=lstm)')
    parser.add_argument('--save_csv', type=str, default='evaluation_results.csv',
                        help='Where to save the evaluation CSV')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.method == "lstm":
        evaluate_lstm(config, args.model_path, args.save_csv)
    elif args.method == "dynamic":
        evaluate_dynamic(config, args.save_csv)

    print("Evaluation complete.")
