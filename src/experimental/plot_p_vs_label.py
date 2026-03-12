#!/usr/bin/env python3
"""
Visualize predicted probability vectors (p) vs true labels.

Input CSV format:
class_0, class_1, ..., class_5, label
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === Modify this path if needed ===
csv_path = Path("results/p_analysis.csv")

# === Load data ===
df = pd.read_csv(csv_path)
n_classes = len([c for c in df.columns if c.startswith("class_")])
probs = df[[f"class_{i}" for i in range(n_classes)]].values
labels = df["label"].values.astype(int)
preds = probs.argmax(axis=1)

print(f"Loaded {len(df)} samples, {n_classes} classes.")
print(f"Prediction collapse ratio (class 0 count): {(preds==0).mean():.3f}")

# === 1. Probability heatmap ===
plt.figure(figsize=(10, 6))
sns.heatmap(probs, cmap="viridis", cbar=True)
plt.xlabel("Predicted class index")
plt.ylabel("Sample index")
plt.title("Predicted probability heatmap (rows = samples)")
plt.tight_layout()
plt.savefig("p_heatmap.png", dpi=200)
plt.close()
print("Saved: p_heatmap.png")

# === 2. True vs Predicted scatter ===
plt.figure(figsize=(6, 6))
plt.scatter(labels, preds, alpha=0.6, s=30)
plt.plot([0, n_classes - 1], [0, n_classes - 1], "r--", label="Perfect prediction")
plt.xlabel("True label")
plt.ylabel("Predicted label")
plt.title("True vs Predicted Class")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("p_scatter_true_vs_pred.png", dpi=200)
plt.close()
print("Saved: p_scatter_true_vs_pred.png")

# === 3. Average predicted probabilities by true label ===
mean_probs = np.zeros((n_classes, n_classes))
for i in range(n_classes):
    subset = df[df["label"] == i]
    if len(subset) > 0:
        mean_probs[i] = subset[[f"class_{j}" for j in range(n_classes)]].mean().values

plt.figure(figsize=(8, 6))
sns.heatmap(mean_probs, annot=True, fmt=".2f", cmap="magma_r",
            xticklabels=[f"class_{j}" for j in range(n_classes)],
            yticklabels=[f"true_{i}" for i in range(n_classes)])
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.title("Average predicted probability per true class")
plt.tight_layout()
plt.savefig("p_meanprob_heatmap.png", dpi=200)
plt.close()
print("Saved: p_meanprob_heatmap.png")

print("\nVisualization complete:")
print(" - p_heatmap.png")
print(" - p_scatter_true_vs_pred.png")
print(" - p_meanprob_heatmap.png")
