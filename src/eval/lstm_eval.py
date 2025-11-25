import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from typing import List, Dict

import pandas as pd
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.dataloader as dataloader_module
from models.fault_lstm_classifier import LSTMClassifier


def topk_accuracy(outputs, targets, topk):
    max_k = max(topk)
    _, preds = outputs.topk(max_k, dim=1, largest=True, sorted=True)
    metrics = []
    for k in topk:
        hits = (preds[:, :k] == targets.view(-1, 1)).any(dim=1).float().mean().item()
        metrics.append(hits)
    return metrics


def evaluate(model, dataloader, criterion, device, topk):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    accum = [0.0 for _ in topk]

    with torch.no_grad():
        for x, _, p in dataloader:
            x = x.to(device).to(torch.float64)
            targets = torch.argmax(p, dim=1).to(device)

            outputs = model(x)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            accs = topk_accuracy(outputs, targets, topk)
            for i, acc in enumerate(accs):
                accum[i] += acc * batch_size

    avg_loss = total_loss / total_samples
    avg_accs = [val / total_samples for val in accum]
    return avg_loss, avg_accs


def run_single_evaluation(config, model_path, device, topk) -> List[Dict]:
    data_cfg = config["data"]
    train_dataset, val_dataset = dataloader_module.load_dataset_from_folder(
        data_dir=data_cfg["path"], config=data_cfg
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

    train_loader = torch.utils.data.DataLoader(
        scaled_train,
        batch_size=config["train"]["batch"],
        shuffle=False
    )
    val_loader = torch.utils.data.DataLoader(
        scaled_val,
        batch_size=config["train"]["batch"],
        shuffle=False
    )

    model = LSTMClassifier(config=config["model"]).to(device).to(torch.float64)
    model.load_state_dict(torch.load(model_path, map_location=device))
    criterion = nn.CrossEntropyLoss()

    train_loss, train_accs = evaluate(model, train_loader, criterion, device, topk)
    val_loss, val_accs = evaluate(model, val_loader, criterion, device, topk)

    rows = []
    for name, loss_val, accs in [("train", train_loss, train_accs), ("val", val_loss, val_accs)]:
        row = {"dataset": name, "loss": loss_val}
        for idx, k in enumerate(topk):
            row[f"top{k}_acc"] = accs[idx]
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate LSTM classifier with optional multi-window support.")
    parser.add_argument("--config", type=str, default="config_lstm_classifier.json")
    parser.add_argument("--model_path", type=str, default="best_model.pth",
                        help="Checkpoint path for single evaluation.")
    parser.add_argument("--model_paths", type=str, nargs='*', default=None,
                        help="List of checkpoints aligning with --window_lengths.")
    parser.add_argument("--window_lengths", type=int, nargs='*', default=None,
                        help="Window sizes to evaluate; requires --model_paths with same length.")
    parser.add_argument("--topk", type=int, nargs='+', default=[1, 2, 3, 4, 5],
                        help="Top-k accuracies to report.")
    parser.add_argument("--output_csv", type=str, default="eval_results.csv")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        base_config = json.load(f)

    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    topk = tuple(args.topk)

    if args.window_lengths:
        if not args.model_paths or len(args.model_paths) != len(args.window_lengths):
            raise ValueError("Provide the same number of --model_paths as --window_lengths.")
        aggregate_rows = []
        for window_size, model_path in zip(args.window_lengths, args.model_paths):
            cfg = deepcopy(base_config)
            cfg["data"]["window_size"] = window_size
            rows = run_single_evaluation(cfg, model_path, device, topk)
            for row in rows:
                row["window_size"] = window_size
            aggregate_rows.extend(rows)
        df = pd.DataFrame(aggregate_rows)
    else:
        rows = run_single_evaluation(base_config, args.model_path, device, topk)
        if "window_size" in base_config["data"]:
            for row in rows:
                row["window_size"] = base_config["data"]["window_size"]
        df = pd.DataFrame(rows)

    df.to_csv(args.output_csv, index=False)
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
