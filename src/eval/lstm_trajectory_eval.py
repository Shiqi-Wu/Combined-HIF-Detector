#!/usr/bin/env python3
"""
LSTM Trajectory-Level Evaluation Script

This script evaluates LSTM models at the trajectory level by:
1. Using the same data loading structure as training (FileTrackingTrajectoryDataset + ScaledDataset)
2. Making predictions on multiple windows from each trajectory
3. Using majority voting to determine the final trajectory classification
4. Evaluating results on the full dataset and generating comprehensive plots

Key features:
- Consistent with training data pipeline
- Trajectory-level accuracy assessment
- Sliding window approach for robust predictions
- Majority voting mechanism
- Full dataset evaluation (no k-fold)
- Comprehensive visualization and statistics
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fault_lstm_classifier import LSTMClassifier
from utils.dataloader import load_full_dataset, ScaledDataset

class LSTMTrajectoryEvaluator:
    def __init__(self, results_dir: str, data_dir: str, preprocessing_params_path: str = None,
                 window_size: int = 30, stride: int = None, device: str = 'cuda'):
        self.results_dir = Path(results_dir)
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        self.device = device

        if preprocessing_params_path and os.path.exists(preprocessing_params_path):
            with open(preprocessing_params_path, 'rb') as f:
                self.preprocessing_params = pickle.load(f)
            self.scaled_dataset = None  # will be loaded later
        else:
            print("Fitting new scalers + PCA from scratch.")
            self.preprocessing_params = None
            self.scaled_dataset = None

        self.labels = []
        self.file_names = []

    def load_dataset(self):
        print("\n📦 Loading full trajectory dataset...")
        config = {'window_size': 3317, 'sample_step': 1}
        base_dataset, labels, file_names = load_full_dataset(str(self.data_dir), config)

        # Fit scalers and PCA if no preprocessing parameters loaded
        if self.preprocessing_params is None:
            scaled_dataset = ScaledDataset(base_dataset, pca_dim=2, fit_scalers=True)
            self.preprocessing_params = scaled_dataset.get_preprocessing_params()
        else:
            scaled_dataset = ScaledDataset(base_dataset, pca_dim=2)
            scaled_dataset.set_preprocessing_params(self.preprocessing_params)

        self.scaled_dataset = scaled_dataset
        self.labels = labels
        self.file_names = file_names

        print(f"✅ Loaded and scaled {len(self.scaled_dataset)} full trajectories.")

    def create_sliding_windows(self, trajectory: torch.Tensor) -> List[torch.Tensor]:
        windows = []
        for i in range(0, len(trajectory) - self.window_size + 1, self.stride):
            windows.append(trajectory[i:i + self.window_size])
        if not windows:
            windows = [trajectory]  # short trajectory fallback
        return windows

    def load_model(self, model_path: str = None) -> Union[None, torch.nn.Module]:
        if model_path is not None:
            candidates = [Path(model_path)]
        else:
            # Try default + fold-specific paths
            candidates = [
                self.results_dir / "best_model.pth",
                self.results_dir / "latest_checkpoint.pth"
            ]
            for i in range(5):
                candidates.append(self.results_dir / f"fold_{i}" / "best_model.pth")
                candidates.append(self.results_dir / f"fold_{i}" / f"best_model_fold_{i}.pth")

            for path in candidates:
                if path.exists():
                    checkpoint = torch.load(path, map_location=self.device)
                    model_config = checkpoint.get('model_config', {
                    'input_size': 2, 'hidden_size': 128,
                    'num_layers': 2, 'num_classes': 6,
                    'dropout': 0.2, 'bidirectional': True
                })
                model = LSTMClassifier(**model_config).to(self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                print(f"✅ Loaded model from {path}")
                return model

        print("❌ No model found in default or fold paths.")
        return None


    def predict_trajectory(self, model: torch.nn.Module, windows: List[torch.Tensor]) -> Tuple[int, np.ndarray]:
        preds, probs = [], []
        with torch.no_grad():
            for w in windows:
                x = w.unsqueeze(0).to(torch.float64).to(self.device)  # (1, T, D)
                out = model(x)
                prob = torch.softmax(out, dim=1).squeeze()
                preds.append(torch.argmax(prob).item())
                probs.append(prob.cpu().numpy())
        vote = Counter(preds).most_common(1)[0][0]
        conf = np.mean(probs, axis=0)
        return vote, conf

    def load_single_model(self, model_path: Path, model_config: Dict) -> Tuple[nn.Module, Dict]:
        """
        Load a single model from checkpoint
        
        Args:
            model_path: Path to model checkpoint
            model_config: Model configuration dictionary
            
        Returns:
            Tuple of (model, checkpoint_info)
        """
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create model
        model = LSTMClassifier(**model_config).to(torch.float64)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        checkpoint_info = {
            'epoch': checkpoint['epoch'],
            'val_accuracy': checkpoint['val_accuracy']
        }
        
        return model, checkpoint_info

    def evaluate(self, model_config: Dict, model_path: str = None) -> Dict:
        model, _ = self.load_single_model(Path(model_path), model_config)
        if model is None:
            return {}

        print("\n🚀 Evaluating trajectories...")
        pred_labels, true_labels, confidences = [], [], []

        for idx in tqdm(range(len(self.scaled_dataset))):
            x, u, p = self.scaled_dataset[idx]
            true_label = np.argmax(p)
            windows = self.create_sliding_windows(x)
            pred, conf = self.predict_trajectory(model, windows)
            pred_labels.append(pred)
            true_labels.append(true_label)
            confidences.append(conf)

        acc = accuracy_score(true_labels, pred_labels)
        report = classification_report(true_labels, pred_labels, output_dict=True)

        return {
            'accuracy': acc,
            'true_labels': true_labels,
            'pred_labels': pred_labels,
            'confidences': confidences,
            'report': report
        }

    def plot_and_save_results(self, result: Dict, save_dir: str):
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        cm = confusion_matrix(result['true_labels'], result['pred_labels'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(save_dir / "confusion_matrix.png", dpi=300)
        plt.close()

        report = result["report"]
        classes = [str(i) for i in range(6)]
        metrics = ['precision', 'recall', 'f1-score']
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, metric in enumerate(metrics):
            values = [report.get(cls, {}).get(metric, 0.0) for cls in classes]
            axes[i].bar(classes, values)
            axes[i].set_ylim(0, 1)
            axes[i].set_title(metric)
        plt.tight_layout()
        plt.savefig(save_dir / "classwise_performance.png", dpi=300)
        plt.close()

        with open(save_dir / "trajectory_evaluation_results.json", 'w') as f:
            json.dump(convert_nested(result), f, indent=2)
        print(f"✅ Results saved to {save_dir}")


def tensor_to_builtin(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj

def convert_nested(obj):
    if isinstance(obj, dict):
        return {k: convert_nested(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_nested(v) for v in obj]
    else:
        return tensor_to_builtin(obj)




def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='LSTM Trajectory-Level Evaluation')
    
    # Required arguments
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing trained models')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing trajectory data (.npy files)')
    
    # Optional arguments
    parser.add_argument('--preprocessing_params', type=str,
                       default='/home/shiqi_w/code/Combined-HIF-detector/preprocessing_params_fold.pkl',
                       help='Path to preprocessing parameters pickle file')
    parser.add_argument('--save_dir', type=str, default='./evaluations/trajectory_lstm',
                       help='Directory to save evaluation results and plots')
    parser.add_argument('--window_size', type=int, default=30,
                       help='Size of sliding windows')
    parser.add_argument('--stride', type=int, default=None,
                       help='Stride for sliding windows (default: same as window_size)')
    parser.add_argument('--max_folds', type=int, default=5,
                       help='Maximum number of folds to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for inference (cuda/cpu)')
    
    args = parser.parse_args()

    model_config = {
        'input_size': 2,
        'hidden_size': 128,
        'num_layers': 2,
        'num_classes': 6,
        'dropout': 0.2,
        'bidirectional': True
    }
    
    # Create evaluator
    evaluator = LSTMTrajectoryEvaluator(
        results_dir=args.results_dir,
        data_dir=args.data_dir,
        preprocessing_params_path=args.preprocessing_params,
        window_size=args.window_size,
        stride=args.stride,
        device=args.device
    )
    
    # Run evaluation
    model_path = os.path.join(args.results_dir, "fold_1/best_model_fold_1.pth")
    evaluator.load_dataset()
    result = evaluator.evaluate(model_config=model_config, model_path=model_path)
    evaluator.plot_and_save_results(result, args.save_dir)




if __name__ == "__main__":
    main()
