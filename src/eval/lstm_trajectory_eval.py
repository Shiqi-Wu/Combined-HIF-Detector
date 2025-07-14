#!/usr/bin/env python3
"""
LSTM Trajectory-Level Evaluation Script

This script evaluates LSTM models at the trajectory level by:
1. Using the same data loading structure as training (FileTrackingTrajectoryDataset + ScaledDataset)
2. Making predictions on multiple windows from each trajectory
3. Using majority voting to determine the final trajectory classification
4. Evaluating results across all folds and generating comprehensive plots

Key features:
- Consistent with training data pipeline
- Trajectory-level accuracy assessment
- Sliding window approach for robust predictions
- Majority voting mechanism
- Cross-fold evaluation and visualization
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
from utils.dataloader import load_full_dataset, create_kfold_dataloaders, ScaledDataset

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TrajectoryDataset:
    """
    Dataset class for handling trajectory-level data
    """
    
    def __init__(self, data_dir: str, preprocessing_params_path: str):
        """
        Initialize trajectory dataset
        
        Args:
            data_dir: Directory containing .npy files
            preprocessing_params_path: Path to preprocessing parameters
        """
        self.data_dir = data_dir
        self.trajectories = []
        self.labels = []
        self.file_names = []
        
        # Load preprocessing parameters
        with open(preprocessing_params_path, 'rb') as f:
            self.preprocessing_params = pickle.load(f)
        
        self._load_trajectories()
    
    def _load_trajectories(self):
        """Load all trajectory files"""
        data_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.npy')])
        
        print(f"Found {len(data_files)} .npy files in {self.data_dir}")
        if len(data_files) == 0:
            print("No .npy files found!")
            return
        
        print(f"First few files: {data_files[:5]}")
        print(f"Loading {len(data_files)} trajectory files...")
        
        for file_name in tqdm(data_files):
            file_path = os.path.join(self.data_dir, file_name)
            
            try:
                # Load trajectory data (allow pickle for object arrays)
                data = np.load(file_path, allow_pickle=True)
                print(f"Loaded {file_name}: shape {data.shape}, dtype {data.dtype}")
                
                # Handle different data formats
                if data.ndim == 0:
                    # If it's a 0-d array (scalar), try to extract the actual data
                    if hasattr(data.item(), 'shape'):
                        data = data.item()
                        print(f"  Extracted data from scalar: shape {data.shape}, dtype {data.dtype}")
                    else:
                        print(f"  Warning: Scalar data without shape attribute: {data.item()}")
                        continue
                
                # Ensure data is at least 2D (time_steps, features)
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                    print(f"  Reshaped 1D data to 2D: {data.shape}")
                elif data.ndim > 2:
                    print(f"  Warning: High-dimensional data ({data.ndim}D), flattening to 2D")
                    data = data.reshape(data.shape[0], -1)
                
                # Check if data has reasonable size
                if data.shape[0] < 10:
                    print(f"  Warning: Very short trajectory ({data.shape[0]} time steps)")
                
                # Extract class from filename (assuming format like "Case_X_Y_Z.npy")
                if 'Case_' in file_name:
                    parts = file_name.replace('.npy', '').split('_')
                    if len(parts) >= 2:
                        try:
                            class_id = int(parts[1]) - 2  # Convert to 0-based indexing (Case_2 -> class 0)
                            print(f"  File {file_name}: extracted class_id = {class_id}")
                            
                            if 0 <= class_id <= 5:  # Valid class range
                                self.trajectories.append(data)
                                self.labels.append(class_id)
                                self.file_names.append(file_name)
                            else:
                                print(f"  Warning: Invalid class_id {class_id}, skipping")
                        except ValueError as e:
                            print(f"  Error parsing class from {file_name}: {e}")
                    else:
                        print(f"  Unexpected filename format: {file_name}")
                else:
                    print(f"  Skipping file (no 'Case_' in name): {file_name}")
                    
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
                continue
        
        print(f"Loaded {len(self.trajectories)} trajectories")
        print(f"Class distribution: {Counter(self.labels)}")
        
        if len(self.trajectories) == 0:
            print("WARNING: No trajectories were loaded successfully!")
            print("Please check the data format and file structure.")
    
    def create_windows(self, trajectory: np.ndarray, window_size: int, stride: int = None) -> List[np.ndarray]:
        """
        Create sliding windows from a trajectory
        
        Args:
            trajectory: Input trajectory of shape (length, features)
            window_size: Size of each window
            stride: Stride for sliding window (default: window_size for non-overlapping)
            
        Returns:
            List of windows
        """
        if stride is None:
            stride = window_size
        
        # Handle different data formats
        if not isinstance(trajectory, np.ndarray):
            trajectory = np.array(trajectory)
        
        # Ensure trajectory is at least 1D
        if trajectory.ndim == 0:
            print(f"Warning: 0-dimensional trajectory encountered")
            return []
        
        # If trajectory is 1D, reshape to 2D (n_samples, 1)
        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(-1, 1)
        
        # Check if trajectory is long enough
        if len(trajectory) < window_size:
            print(f"Warning: Trajectory length {len(trajectory)} is shorter than window_size {window_size}")
            if len(trajectory) > 0:
                return [trajectory]  # Return the whole trajectory as a single window
            else:
                return []
        
        windows = []
        for i in range(0, len(trajectory) - window_size + 1, stride):
            window = trajectory[i:i + window_size]
            windows.append(window)
        
        return windows
    
    def preprocess_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to trajectory data
        
        Args:
            trajectory: Raw trajectory data
            
        Returns:
            Preprocessed trajectory
        """
        # Ensure trajectory is a numpy array
        if not isinstance(trajectory, np.ndarray):
            trajectory = np.array(trajectory)
        
        # Handle different data formats
        if trajectory.ndim == 0:
            print("Warning: 0-dimensional trajectory in preprocessing")
            return np.array([[0, 0]])  # Return default 2D array
        
        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(-1, 1)
        
        # Apply same preprocessing as training
        processed = trajectory.copy().astype(np.float64)
        
        try:
            # Apply saved preprocessing transformations
            if 'pca' in self.preprocessing_params:
                pca = self.preprocessing_params['pca']
                if hasattr(pca, 'transform'):
                    # Ensure the input has the right number of features
                    if processed.shape[1] != pca.n_features_in_:
                        print(f"Warning: Feature mismatch. Data has {processed.shape[1]} features, PCA expects {pca.n_features_in_}")
                        # Pad or truncate as needed
                        if processed.shape[1] < pca.n_features_in_:
                            # Pad with zeros
                            padding = np.zeros((processed.shape[0], pca.n_features_in_ - processed.shape[1]))
                            processed = np.hstack([processed, padding])
                        else:
                            # Truncate
                            processed = processed[:, :pca.n_features_in_]
                    
                    processed = pca.transform(processed)
            
            if 'scaler' in self.preprocessing_params:
                scaler = self.preprocessing_params['scaler']
                if hasattr(scaler, 'transform'):
                    processed = scaler.transform(processed)
                    
        except Exception as e:
            print(f"Warning: Error applying preprocessing: {e}")
            # Return original data if preprocessing fails
            if processed.ndim == 1:
                processed = processed.reshape(-1, 1)
        
        # Ensure we return at least a 2D array with 2 features (for PCA)
        if processed.shape[1] < 2:
            # Duplicate the feature or add zeros
            if processed.shape[1] == 1:
                processed = np.hstack([processed, processed])
            else:
                processed = np.hstack([processed, np.zeros((processed.shape[0], 2 - processed.shape[1]))])
        
        return processed
    
    def get_trajectory_data(self, idx: int) -> Tuple[np.ndarray, int, str]:
        """
        Get trajectory data by index
        
        Returns:
            trajectory: Preprocessed trajectory data
            label: Class label
            filename: Original filename
        """
        trajectory = self.trajectories[idx]
        label = self.labels[idx]
        filename = self.file_names[idx]
        
        # Preprocess trajectory
        processed_trajectory = self.preprocess_trajectory(trajectory)
        
        return processed_trajectory, label, filename
    
    def __len__(self):
        return len(self.trajectories)


class LSTMTrajectoryEvaluator:
    """
    LSTM Trajectory-Level Evaluator
    """
    
    def __init__(self, results_dir: str, data_dir: str, preprocessing_params_path: str,
                 window_size: int = 30, stride: int = None, device: str = 'cuda'):
        """
        Initialize evaluator
        
        Args:
            results_dir: Directory containing trained models
            data_dir: Directory containing trajectory data
            preprocessing_params_path: Path to preprocessing parameters
            window_size: Size of sliding windows
            stride: Stride for sliding windows
            device: Device to use for inference
        """
        self.results_dir = Path(results_dir)
        self.data_dir = data_dir
        self.preprocessing_params_path = preprocessing_params_path
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load trajectory dataset
        self.dataset = TrajectoryDataset(data_dir, preprocessing_params_path)
        
        # Results storage
        self.fold_results = {}
        self.models = {}
        
        print(f"Initialized evaluator:")
        print(f"  Results directory: {self.results_dir}")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Window size: {self.window_size}")
        print(f"  Stride: {self.stride}")
        print(f"  Device: {self.device}")
        print(f"  Loaded {len(self.dataset)} trajectories")
    
    def load_fold_model(self, fold_idx: int) -> Optional[nn.Module]:
        """
        Load trained model for a specific fold
        
        Args:
            fold_idx: Fold index
            
        Returns:
            Loaded model or None if not found
        """
        # Try different possible model file names
        possible_paths = [
            self.results_dir / f"fold_{fold_idx + 1}" / "best_model.pth",
            self.results_dir / f"fold_{fold_idx + 1}" / f"best_model_fold_{fold_idx + 1}.pth",
            self.results_dir / f"best_model_fold_{fold_idx + 1}.pth",
        ]
        
        for model_path in possible_paths:
            if model_path.exists():
                try:
                    print(f"Loading model from {model_path}")
                    checkpoint = torch.load(model_path, map_location='cpu')
                    
                    # Extract model config
                    model_config = checkpoint.get('model_config', {
                        'input_size': 2,  # Default PCA dimension
                        'hidden_size': 128,
                        'num_layers': 2,
                        'num_classes': 6,
                        'dropout': 0.2,
                        'bidirectional': True
                    })
                    
                    # Create and load model
                    model = LSTMClassifier(**model_config)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model = model.to(self.device).eval()
                    
                    print(f"Successfully loaded model for fold {fold_idx + 1}")
                    return model
                    
                except Exception as e:
                    print(f"Error loading model from {model_path}: {e}")
                    continue
        
        print(f"No model found for fold {fold_idx + 1}")
        return None
    
    def predict_trajectory(self, model: nn.Module, trajectory: np.ndarray) -> Tuple[int, np.ndarray, List[int]]:
        """
        Predict class for a trajectory using sliding windows and majority voting
        
        Args:
            model: Trained LSTM model
            trajectory: Input trajectory data
            
        Returns:
            predicted_class: Final predicted class (majority vote)
            confidence_scores: Average confidence scores across all windows
            window_predictions: List of predictions for each window
        """
        # Create windows
        windows = self.dataset.create_windows(trajectory, self.window_size, self.stride)
        
        if len(windows) == 0:
            # Trajectory too short, use the whole trajectory
            if len(trajectory) > 0:
                windows = [trajectory]
            else:
                return 0, np.zeros(6), [0]  # Default prediction
        
        window_predictions = []
        all_confidences = []
        
        model.eval()
        with torch.no_grad():
            for window in windows:
                # Convert to tensor and add batch dimension
                window_tensor = torch.FloatTensor(window).unsqueeze(0).to(self.device)
                
                # Make prediction
                output = model(window_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item()
                
                window_predictions.append(predicted_class)
                all_confidences.append(probabilities.cpu().numpy()[0])
        
        # Majority voting
        vote_counts = Counter(window_predictions)
        predicted_class = vote_counts.most_common(1)[0][0]
        
        # Average confidence scores
        confidence_scores = np.mean(all_confidences, axis=0)
        
        return predicted_class, confidence_scores, window_predictions
    
    def evaluate_fold(self, fold_idx: int) -> Dict:
        """
        Evaluate all trajectories for a specific fold
        
        Args:
            fold_idx: Fold index
            
        Returns:
            Dictionary containing evaluation results
        """
        model = self.load_fold_model(fold_idx)
        if model is None:
            return {}
        
        results = {
            'fold_idx': fold_idx,
            'trajectory_results': [],
            'predictions': [],
            'true_labels': [],
            'confidence_scores': [],
            'window_predictions': []
        }
        
        print(f"Evaluating fold {fold_idx + 1} on {len(self.dataset)} trajectories...")
        
        for traj_idx in tqdm(range(len(self.dataset))):
            trajectory, true_label, filename = self.dataset.get_trajectory_data(traj_idx)
            
            # Make prediction
            pred_class, confidence, window_preds = self.predict_trajectory(model, trajectory)
            
            # Store results
            traj_result = {
                'trajectory_idx': traj_idx,
                'filename': filename,
                'true_label': true_label,
                'predicted_label': pred_class,
                'confidence_scores': confidence.tolist(),
                'num_windows': len(window_preds),
                'window_predictions': window_preds,
                'correct': pred_class == true_label
            }
            
            results['trajectory_results'].append(traj_result)
            results['predictions'].append(pred_class)
            results['true_labels'].append(true_label)
            results['confidence_scores'].append(confidence)
            results['window_predictions'].append(window_preds)
        
        # Calculate metrics
        if len(results['true_labels']) == 0:
            print(f"No trajectories to evaluate for fold {fold_idx + 1}")
            return {}
        
        accuracy = accuracy_score(results['true_labels'], results['predictions'])
        
        results['accuracy'] = accuracy
        
        # Only generate classification report if we have predictions
        try:
            results['classification_report'] = classification_report(
                results['true_labels'], 
                results['predictions'], 
                target_names=[f'Class {i}' for i in range(6)],
                output_dict=True,
                zero_division=0,
                labels=list(range(6))  # Explicitly specify all possible labels
            )
        except Exception as e:
            print(f"Error generating classification report: {e}")
            results['classification_report'] = {}
        
        print(f"Fold {fold_idx + 1} accuracy: {accuracy:.4f}")
        
        return results
    
    def evaluate_all_folds(self, max_folds: int = 5) -> Dict:
        """
        Evaluate all available folds
        
        Args:
            max_folds: Maximum number of folds to evaluate
            
        Returns:
            Dictionary containing results for all folds
        """
        print("Starting evaluation across all folds...")
        
        for fold_idx in range(max_folds):
            print(f"\n{'='*50}")
            print(f"Evaluating Fold {fold_idx + 1}/{max_folds}")
            print(f"{'='*50}")
            
            fold_results = self.evaluate_fold(fold_idx)
            if fold_results:
                self.fold_results[fold_idx] = fold_results
        
        print(f"\nCompleted evaluation of {len(self.fold_results)} folds")
        return self.fold_results
    
    def plot_fold_comparison(self, save_path: Optional[str] = None):
        """Plot accuracy comparison across folds"""
        if not self.fold_results:
            print("No results to plot")
            return
        
        fold_accuracies = []
        fold_labels = []
        
        for fold_idx, results in self.fold_results.items():
            fold_accuracies.append(results['accuracy'])
            fold_labels.append(f'Fold {fold_idx + 1}')
        
        plt.figure(figsize=(10, 6))
        
        # Bar plot
        bars = plt.bar(fold_labels, fold_accuracies, alpha=0.7, color='skyblue', edgecolor='navy')
        
        # Add value labels on bars
        for bar, acc in zip(bars, fold_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add average line
        avg_accuracy = np.mean(fold_accuracies)
        plt.axhline(y=avg_accuracy, color='red', linestyle='--', alpha=0.7,
                   label=f'Average: {avg_accuracy:.3f}')
        
        plt.title('Trajectory-Level Accuracy Across Folds', fontsize=16, fontweight='bold')
        plt.xlabel('Fold', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        std_accuracy = np.std(fold_accuracies)
        plt.text(0.02, 0.98, f'Mean ± Std: {avg_accuracy:.3f} ± {std_accuracy:.3f}',
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="lightgray", alpha=0.7), fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Fold comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrices(self, save_dir: Optional[str] = None):
        """Plot confusion matrices for each fold"""
        if not self.fold_results:
            print("No results to plot")
            return
        
        n_folds = len(self.fold_results)
        cols = min(3, n_folds)
        rows = (n_folds + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_folds == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        class_names = [f'Class {i}' for i in range(6)]
        
        for i, (fold_idx, results) in enumerate(self.fold_results.items()):
            ax = axes[i] if n_folds > 1 else axes[0]
            
            cm = confusion_matrix(results['true_labels'], results['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=class_names, yticklabels=class_names)
            ax.set_title(f'Fold {fold_idx + 1}\nAccuracy: {results["accuracy"]:.3f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
        
        # Hide unused subplots
        for i in range(n_folds, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, 'confusion_matrices.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrices saved to {save_path}")
        
        plt.show()
    
    def plot_class_wise_performance(self, save_path: Optional[str] = None):
        """Plot class-wise performance metrics across folds"""
        if not self.fold_results:
            print("No results to plot")
            return
        
        # Collect class-wise metrics
        class_metrics = {}
        for fold_idx, results in self.fold_results.items():
            report = results['classification_report']
            for class_name in [f'Class {i}' for i in range(6)]:
                if class_name not in class_metrics:
                    class_metrics[class_name] = {'precision': [], 'recall': [], 'f1-score': []}
                
                if class_name in report:
                    class_metrics[class_name]['precision'].append(report[class_name]['precision'])
                    class_metrics[class_name]['recall'].append(report[class_name]['recall'])
                    class_metrics[class_name]['f1-score'].append(report[class_name]['f1-score'])
                else:
                    class_metrics[class_name]['precision'].append(0)
                    class_metrics[class_name]['recall'].append(0)
                    class_metrics[class_name]['f1-score'].append(0)
        
        # Create plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        metrics = ['precision', 'recall', 'f1-score']
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Prepare data for plotting
            classes = list(class_metrics.keys())
            mean_scores = [np.mean(class_metrics[cls][metric]) for cls in classes]
            std_scores = [np.std(class_metrics[cls][metric]) for cls in classes]
            
            # Bar plot with error bars
            bars = ax.bar(classes, mean_scores, yerr=std_scores, capsize=5, 
                         alpha=0.7, color='lightcoral', edgecolor='darkred')
            
            # Add value labels
            for bar, mean_val in zip(bars, mean_scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(f'{metric.capitalize()} Across Classes', fontsize=14, fontweight='bold')
            ax.set_xlabel('Class', fontsize=12)
            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class-wise performance plot saved to {save_path}")
        
        plt.show()
    
    def plot_window_analysis(self, save_path: Optional[str] = None):
        """Plot analysis of window-level predictions"""
        if not self.fold_results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Collect window statistics
        all_num_windows = []
        all_window_accuracy = []
        all_confidence_variance = []
        
        for fold_idx, results in self.fold_results.items():
            for traj_result in results['trajectory_results']:
                num_windows = traj_result['num_windows']
                window_preds = traj_result['window_predictions']
                true_label = traj_result['true_label']
                confidence = np.array(traj_result['confidence_scores'])
                
                all_num_windows.append(num_windows)
                
                # Window-level accuracy
                window_acc = np.mean([pred == true_label for pred in window_preds])
                all_window_accuracy.append(window_acc)
                
                # Confidence variance
                conf_var = np.var(confidence)
                all_confidence_variance.append(conf_var)
        
        # Plot 1: Distribution of number of windows per trajectory
        axes[0, 0].hist(all_num_windows, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Windows per Trajectory')
        axes[0, 0].set_xlabel('Number of Windows')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Window-level accuracy distribution
        axes[0, 1].hist(all_window_accuracy, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Window-Level Accuracy Distribution')
        axes[0, 1].set_xlabel('Window Accuracy')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Confidence variance distribution
        axes[1, 0].hist(all_confidence_variance, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_title('Confidence Variance Distribution')
        axes[1, 0].set_xlabel('Confidence Variance')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Relationship between num windows and accuracy
        axes[1, 1].scatter(all_num_windows, all_window_accuracy, alpha=0.6, color='purple')
        axes[1, 1].set_title('Windows Count vs Window Accuracy')
        axes[1, 1].set_xlabel('Number of Windows')
        axes[1, 1].set_ylabel('Window Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Window analysis plot saved to {save_path}")
        
        plt.show()
    
    def save_results(self, save_dir: str):
        """Save all results to files"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = save_dir / 'trajectory_evaluation_results.json'
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for fold_idx, results in self.fold_results.items():
            json_results[f'fold_{fold_idx}'] = {
                'fold_idx': results['fold_idx'],
                'accuracy': results['accuracy'],
                'classification_report': results['classification_report'],
                'trajectory_results': []
            }
            
            for traj_result in results['trajectory_results']:
                json_traj = traj_result.copy()
                json_traj['confidence_scores'] = [float(x) for x in json_traj['confidence_scores']]
                json_results[f'fold_{fold_idx}']['trajectory_results'].append(json_traj)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Detailed results saved to {results_file}")
        
        # Save summary CSV
        summary_data = []
        for fold_idx, results in self.fold_results.items():
            summary_data.append({
                'fold': fold_idx + 1,
                'accuracy': results['accuracy'],
                'num_trajectories': len(results['trajectory_results'])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = save_dir / 'trajectory_evaluation_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        
        print(f"Summary saved to {summary_file}")
        
        # Calculate and save overall statistics
        all_accuracies = [results['accuracy'] for results in self.fold_results.values()]
        overall_stats = {
            'mean_accuracy': np.mean(all_accuracies),
            'std_accuracy': np.std(all_accuracies),
            'min_accuracy': np.min(all_accuracies),
            'max_accuracy': np.max(all_accuracies),
            'num_folds': len(all_accuracies)
        }
        
        stats_file = save_dir / 'overall_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(overall_stats, f, indent=2)
        
        print(f"Overall statistics saved to {stats_file}")
        print(f"Mean accuracy: {overall_stats['mean_accuracy']:.4f} ± {overall_stats['std_accuracy']:.4f}")
    
    def run_full_evaluation(self, save_dir: str, max_folds: int = 5):
        """
        Run complete trajectory-level evaluation
        
        Args:
            save_dir: Directory to save results and plots
            max_folds: Maximum number of folds to evaluate
        """
        print("Starting trajectory-level LSTM evaluation...")
        print(f"Window size: {self.window_size}")
        print(f"Stride: {self.stride}")
        print(f"Save directory: {save_dir}")
        
        # Create save directory
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluate all folds
        self.evaluate_all_folds(max_folds)
        
        if not self.fold_results:
            print("No results obtained. Check if model files exist.")
            return
        
        # Generate plots
        print("\nGenerating plots...")
        
        # Fold comparison
        self.plot_fold_comparison(save_dir / 'fold_accuracy_comparison.png')
        
        # Confusion matrices
        self.plot_confusion_matrices(save_dir)
        
        # Class-wise performance
        self.plot_class_wise_performance(save_dir / 'class_wise_performance.png')
        
        # Window analysis
        self.plot_window_analysis(save_dir / 'window_analysis.png')
        
        # Save results
        self.save_results(save_dir)
        
        print(f"\nEvaluation completed! Results saved to {save_dir}")


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
    evaluator.run_full_evaluation(
        save_dir=args.save_dir,
        max_folds=args.max_folds
    )


if __name__ == "__main__":
    main()
