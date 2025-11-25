#!/usr/bin/env python3
"""
Plot top-k accuracy vs window size for multiple methods.

Produces a 2x3 grid (by default) where the first row shows train metrics
and the second row shows validation metrics; columns correspond to top-k
values (top1/top2/top3). Each subplot overlays all provided methods.

Example:
    python tools/plot_method_comparison.py \
        --inputs results/lstm_multi.csv LSTM \
                 results/dynamic_window.csv Dynamic \
                 results/compare_koopman_window.csv Koopman \
        --topk 1 2 3 \
        --rows train val \
        --output figures/method_compare.png
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def parse_inputs(args: argparse.Namespace) -> List[Tuple[Path, str]]:
    if len(args.inputs) % 2 != 0:
        raise ValueError("Provide inputs as pairs: <csv_path> <label>")
    pairs = []
    for i in range(0, len(args.inputs), 2):
        csv_path = Path(args.inputs[i])
        label = args.inputs[i + 1]
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        pairs.append((csv_path, label))
    return pairs


def normalize_dataframe(df: pd.DataFrame, topk: List[int]) -> pd.DataFrame:
    if "window_size" in df.columns:
        df = df.rename(columns={"window_size": "window"})
    elif "window_length" in df.columns:
        df = df.rename(columns={"window_length": "window"})
    else:
        raise ValueError("CSV must contain column 'window_size' or 'window_length'.")

    available_topk = [k for k in topk if f"top{k}_acc" in df.columns]
    if not available_topk:
        raise ValueError("CSV must contain at least one of the requested top-k columns.")
    df = df[["dataset", "window"] + [f"top{k}_acc" for k in available_topk]]
    df.attrs["available_topk"] = available_topk

    return df


def load_data(pairs: List[Tuple[Path, str]], topk: List[int]) -> Dict[str, pd.DataFrame]:
    data = {}
    global_topk = set()
    for path, label in pairs:
        df = pd.read_csv(path)
        df = normalize_dataframe(df, topk)
        global_topk.update(df.attrs.get("available_topk", []))
        df["dataset"] = df["dataset"].str.lower()
        data[label] = df
    if not global_topk:
        raise ValueError("None of the inputs contained the requested top-k columns.")
    return data, sorted(global_topk)


def plot_comparison(data: Dict[str, pd.DataFrame], rows: List[str], topk: List[int],
                    output: Path) -> None:
    num_rows = len(rows)
    num_cols = len(topk)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 3.5 * num_rows), sharex=True)
    if num_rows == 1:
        axes = [axes]

    for r_idx, row_name in enumerate(rows):
        row_axes = axes[r_idx]
        if num_cols == 1:
            row_axes = [row_axes]
        for c_idx, k in enumerate(topk):
            ax = row_axes[c_idx]
            for label, df in data.items():
                subset = df[df["dataset"] == row_name.lower()]
                if subset.empty:
                    continue
                column = f"top{k}_acc"
                if column not in subset.columns:
                    continue
                ax.plot(subset["window"], subset[column], marker='o', label=label)
            ax.set_title(f"{row_name.capitalize()} - Top-{k}")
            ax.set_ylabel("Accuracy")
            ax.grid(True, linestyle="--", alpha=0.3)
            if r_idx == num_rows - 1:
                ax.set_xlabel("Window size")
            if r_idx == 0 and c_idx == num_cols - 1:
                ax.legend()

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Saved comparison plot to {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot multi-method top-k comparisons.")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Pairs of <csv_path> <method_label> ...")
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 2, 3],
                        help="Top-k columns to plot (requires top{k}_acc in CSV).")
    parser.add_argument("--rows", nargs="+", default=["train", "val"],
                        help="Row order to plot (e.g., train val).")
    parser.add_argument("--output", type=str, default="./figures/method_compare.png",
                        help="Output image path.")
    return parser.parse_args()


def main():
    args = parse_args()
    pairs = parse_inputs(args)
    data, available_topk = load_data(pairs, args.topk)
    plot_comparison(data, args.rows, available_topk, Path(args.output))


if __name__ == "__main__":
    main()
