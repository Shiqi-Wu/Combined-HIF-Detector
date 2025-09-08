#!/usr/bin/env python3
"""
Visualization: Compare Dynamic vs LSTM classifier across window sizes
- Subplot 1: Top-1 accuracy
- Subplot 2: Top-2 accuracy
- Subplot 3: Top-3 accuracy
- Subplot 4: Δ accuracy (Dynamic - LSTM), for Top-1, Top-2, Top-3
"""

import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", palette="muted", font_scale=1.2)


def load_results(base_dir="checkpoints", classifiers=("known_control_classifier", "lstm_classifier"), split="val"):
    records = []

    for clf in classifiers:
        pattern = os.path.join(base_dir, clf, "*", "eval_results.csv")
        for path in glob.glob(pattern):
            m = re.search(rf"{clf}/(\d+)/eval_results\.csv", path)
            if not m:
                continue
            window_size = int(m.group(1))

            df = pd.read_csv(path)
            val_row = df[df["dataset"] == split].iloc[0].to_dict()

            record = {
                "classifier": "dynamic" if "known_control" in clf else "lstm",
                "window_size": window_size,
            }
            for k in range(1, 6):
                record[f"top{k}_acc"] = val_row[f"top{k}_acc"]
            records.append(record)

    return pd.DataFrame(records)


def plot_comparison(df, save_path="comparison.png"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    # ---- 子图1-3：Top-1,2,3 Accuracy ----
    for i, k in enumerate([1, 2, 3]):
        ax = axes[i]
        sns.lineplot(
            data=df,
            x="window_size",
            y=f"top{k}_acc",
            hue="classifier",
            marker="o",
            ax=ax
        )
        ax.set_title(f"Top-{k} Accuracy (Validation)")
        ax.set_xlabel("Window Size")
        ax.set_ylabel("Accuracy")
        ax.legend(title="Classifier")

    # ---- 子图4：Delta (Dynamic - LSTM) ----
    df_pivot = df.pivot(index="window_size", columns="classifier")
    records = []
    for k in [1, 2, 3]:
        dynamic_vals = df_pivot[f"top{k}_acc"]["dynamic"]
        lstm_vals = df_pivot[f"top{k}_acc"]["lstm"]
        delta = dynamic_vals - lstm_vals
        for ws, d in delta.items():
            records.append({"window_size": ws, "metric": f"Top-{k}", "delta": d})
    df_delta = pd.DataFrame(records)

    ax = axes[3]
    sns.lineplot(
        data=df_delta,
        x="window_size",
        y="delta",
        hue="metric",
        marker="o",
        ax=ax
    )
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_title("Δ Accuracy (Dynamic - LSTM)")
    ax.set_xlabel("Window Size")
    ax.set_ylabel("Δ Accuracy")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved comparison plot to {save_path}")


if __name__ == "__main__":
    df = load_results(base_dir="./checkpoints", split="train")
    print(df)

    plot_comparison(df, save_path="./figures/lstm_dynamic_comparison_train.png")
