import torch
import torch.nn as nn
import json
import os
import sys
import argparse
import logging
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.dataloader as dataloader_module
from models.fault_lstm_classifier import LSTMClassifier


def topk_accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy for specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)   # [batch, maxk]
    pred = pred.t()                              # [maxk, batch]
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append((correct_k / batch_size).item())
    return res


def evaluate(model, dataloader, criterion, device, topk=(1,5)):
    model.eval()
    total_loss, total_samples = 0, 0
    topk_sums = [0 for _ in topk]

    with torch.no_grad():
        for x, _, p in dataloader:
            x = x.to(device).to(torch.float64)
            target = torch.argmax(p, dim=1).to(device)

            outputs = model(x)
            loss = criterion(outputs, target)

            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            accs = topk_accuracy(outputs, target, topk)
            for i, acc in enumerate(accs):
                topk_sums[i] += acc * batch_size

    avg_loss = total_loss / total_samples
    avg_accs = [s / total_samples for s in topk_sums]
    return avg_loss, avg_accs


def main(config_path, model_path, output_csv):
    with open(config_path, 'r') as f:
        config = json.load(f)

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset
    data_cfg = config["data"]
    train_dataset, val_dataset = dataloader_module.load_dataset_from_folder(
        data_dir=data_cfg["path"], config=data_cfg
    )

    scaled_train = dataloader_module.ScaledDataset(train_dataset, pca_dim=data_cfg.get("pca_dim", 2), fit_scalers=True)
    scaled_val = dataloader_module.ScaledDataset(val_dataset, pca_dim=data_cfg.get("pca_dim", 2))
    scaled_val.set_preprocessing_params(scaled_train.get_preprocessing_params())

    train_loader = torch.utils.data.DataLoader(scaled_train, batch_size=config["train"]["batch"], shuffle=False)
    val_loader = torch.utils.data.DataLoader(scaled_val, batch_size=config["train"]["batch"], shuffle=False)

    # load model
    model = LSTMClassifier(config=config["model"]).to(device).to(torch.float64)
    model.load_state_dict(torch.load(model_path, map_location=device))
    criterion = nn.CrossEntropyLoss()

    # evaluate
    train_loss, train_accs = evaluate(model, train_loader, criterion, device, topk=(1,2,3,4,5))
    val_loss, val_accs = evaluate(model, val_loader, criterion, device, topk=(1,2,3,4,5))

    # save to csv
    results = {
        "dataset": ["train", "val"],
        "loss": [train_loss, val_loss]
    }
    for i, k in enumerate([1,2,3,4,5]):
        results[f"top{k}_acc"] = [train_accs[i], val_accs[i]]

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    logging.info(f"Results saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LSTM Classifier")
    parser.add_argument("--config", type=str, default="config_lstm_classifier.json")
    parser.add_argument("--model_path", type=str, default="best_model.pth")
    parser.add_argument("--output_csv", type=str, default="eval_results.csv")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    main(args.config, args.model_path, args.output_csv)
