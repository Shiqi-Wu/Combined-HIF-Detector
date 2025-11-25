#!/usr/bin/env python3
"""
Plot top-k accuracy trends for multiple evaluation methods.

Each method should provide a CSV file with columns:
    dataset (train/val), window_length, top1_acc, top2_acc, top3_acc, ...

Usage example:
    python tools/plot_topk_trends.py \
        --inputs results/compare_lstm_window.csv LSTM \
                 results/compare_dynamic_window.csv Dynamic \
                 results/compare_koopman_window.csv Koopman \
        --topk 1 2 3 \
        --output ./figures/topk_trends.png
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def parse_inputs(args: argparse.Namespace) -> List[Tuple[Path, str]]:
    if len(args.inputs) % 2 != 0:
        raise ValueError("Inputs must be provided as pairs: <csv_path> <label>")
    pairs = []
    for i in range(0, len(args.inputs), 2):
        csv_path = Path(args.inputs[i])
        label = args.inputs[i + 1]
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        pairs.append((csv_path, label))
    return pairs


def load_trend(csv_path: Path, label: str, topk: List[int]) -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    missing = [k for k in topk if f"top{k}_acc" not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")
    return {
        dataset: sub.set_index("window_length")[ [f"top{k}_acc" for k in topk] ]
        for dataset, sub in df.groupby("dataset")
    }


def plot_trends(data: Dict[str, Dict[str, pd.DataFrame]], topk: List[int],
                output: Path, title: str) -> None:
    datasets = sorted({ds for method in data.values() for ds in method.keys()})
    num_rows = len(datasets)
    fig, axes = plt.subplots(num_rows, 1, figsize=(8, 4 * num_rows), sharex=True)
    if num_rows == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        for method_label, method_data in data.items():
            if dataset not in method_data:
                continue
            df = method_data[dataset]
            for k in topk:
                ax.plot(df.index, df[f"top{k}_acc"], marker='o', label=f"{method_label} top{k}")
        ax.set_title(f"{dataset} split")
        ax.set_ylabel("Accuracy")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Window length")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Saved plot to {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot top-k trends across methods.")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Pairs of <csv_path> <method_label> ...")
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 2, 3],
                        help="Top-k metrics to plot (columns must exist).")
    parser.add_argument("--output", type=str, default="./figures/topk_trends.png",
                        help="Output image path.")
    parser.add_argument("--title", type=str, default="Top-k accuracy vs window length",
                        help="Plot title.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_pairs = parse_inputs(args)
    data = {}
    for csv_path, label in input_pairs:
        data[label] = load_trend(csv_path, label, args.topk)

    plot_trends(data, args.topk, Path(args.output), args.title)


if __name__ == "__main__":
    main()

