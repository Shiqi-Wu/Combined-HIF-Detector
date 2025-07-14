#!/usr/bin/env python3
"""
State-Based Dynamic System Classifier Evaluation

This script implements fault classification using state signals x as control inputs:
- Linear dynamic system modeling: x[t+1] = K*x[t] + B(p)*x[t]
- Uses state signals x as both input and "control" for classification
- Classifies by finding the class that minimizes prediction error

The approach:
- Computes a shared system matrix K using all data
- Estimates class-specific B matrices for each fault type using x as "control"
- Uses state signals x in place of control signals u
- Classifies by finding the class that minimizes prediction error
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, top_k_accuracy_score

import torch
from torch.utils.data import Dataset

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dynamic_system_classifier import compute_K_and_B_per_class
from utils.dataloader import load_full_dataset, create_kfold_dataloaders, ScaledDataset

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class DynamicSystemDataset(Dataset):
    """Dataset wrapper for dynamic system modeling"""
    
    def __init__(self, scaled_dataset):
        self.scaled_dataset = scaled_dataset
        
    def __len__(self):
        return len(self.scaled_dataset)
    
    def __getitem__(self, idx):
        return self.scaled_dataset[idx]


def convert_to_json_serializable(obj):
    """Convert numpy types and other non-serializable objects to JSON-compatible types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    else:
        return obj


def evaluate_state_based_classifier(x_test, et_test, K_list, B_list, verbose=True):
    """
    Evaluate state-based dynamic system classifier
    
    Args:
        x_test: Test state data
        et_test: Test labels
        K_list: List of system matrices (one per class)
        B_list: List of input matrices (one per class)
        verbose: Whether to print progress
        
    Returns:
        predictions: Predicted class indices
        residuals: Prediction residuals for each class
        errors_dict: Dictionary containing error analysis
    """
    num_classes = len(K_list)
    num_samples = len(x_test)
    
    predictions = []
    all_residuals = []
    
    # For each test sample
    for i, (x, et) in enumerate(zip(x_test, et_test)):
        if verbose and i % 100 == 0:
            print(f"Processing sample {i}/{num_samples}")
            
        sample_residuals = []
        
        # For each class, compute prediction error using x as both state and "control"
        for class_idx in range(num_classes):
            K = K_list[class_idx]
            B = B_list[class_idx]
            
            # Predict next states: x[t+1] = K*x[t] + B*x[t]
            x_pred = x[:-1] @ K.T + x[:-1] @ B.T
            
            # Compute residual
            residual = np.mean((x[1:] - x_pred) ** 2)
            sample_residuals.append(residual)
        
        # Classify based on minimum residual
        predicted_class = np.argmin(sample_residuals)
        predictions.append(predicted_class)
        all_residuals.append(sample_residuals)
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    all_residuals = np.array(all_residuals)
    
    # Compute error statistics
    errors_dict = {}
    for class_idx in range(num_classes):
        class_residuals = all_residuals[:, class_idx]
        errors_dict[f'class_{class_idx}'] = {
            'mean_error': float(np.mean(class_residuals)),
            'std_error': float(np.std(class_residuals)),
            'min_error': float(np.min(class_residuals)),
            'max_error': float(np.max(class_residuals)),
            'median_error': float(np.median(class_residuals))
        }
    
    return predictions, all_residuals, errors_dict


def compute_top_k_accuracy(y_true, all_residuals, k_values=[1, 2, 3, 5]):
    """
    Compute top-k accuracy by ranking classes by prediction error (lower is better)
    """
    # Sort classes by residual (ascending order - lower error is better)
    sorted_indices = np.argsort(all_residuals, axis=1)
    
    top_k_accuracies = {}
    for k in k_values:
        if k <= all_residuals.shape[1]:
            # Check if true class is in top-k predictions
            correct = 0
            for i, true_class in enumerate(y_true):
                if true_class in sorted_indices[i, :k]:
                    correct += 1
            accuracy = correct / len(y_true)
            top_k_accuracies[f'top_{k}'] = float(accuracy)
    
    return top_k_accuracies


def analyze_residual_distribution(all_residuals, y_true, class_names=None):
    """
    Analyze the distribution of residuals for each class
    """
    num_classes = all_residuals.shape[1]
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    analysis = {}
    
    for class_idx in range(num_classes):
        residuals = all_residuals[:, class_idx]
        
        # Split by true class
        class_analysis = {}
        for true_class in range(num_classes):
            mask = y_true == true_class
            if np.sum(mask) > 0:
                class_residuals = residuals[mask]
                class_analysis[f'true_class_{true_class}'] = {
                    'mean': float(np.mean(class_residuals)),
                    'std': float(np.std(class_residuals)),
                    'min': float(np.min(class_residuals)),
                    'max': float(np.max(class_residuals)),
                    'median': float(np.median(class_residuals)),
                    'count': int(np.sum(mask))
                }
        
        analysis[f'pred_class_{class_idx}'] = class_analysis
    
    return analysis


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title="Confusion Matrix"):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm


def plot_residual_distributions(all_residuals, y_true, class_names, save_path):
    """Plot residual distributions for each class"""
    num_classes = len(class_names)
    
    fig, axes = plt.subplots(2, (num_classes + 1) // 2, figsize=(16, 10))
    if num_classes == 1:
        axes = [axes]
    axes = axes.flatten()
    
    for class_idx in range(num_classes):
        ax = axes[class_idx]
        residuals = all_residuals[:, class_idx]
        
        # Plot histogram
        ax.hist(residuals, bins=30, alpha=0.7, density=True, label='All samples')
        
        # Plot by true class
        colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
        for true_class in range(num_classes):
            mask = y_true == true_class
            if np.sum(mask) > 0:
                class_residuals = residuals[mask]
                ax.hist(class_residuals, bins=20, alpha=0.5, density=True, 
                       label=f'True {class_names[true_class]}', color=colors[true_class])
        
        ax.set_title(f'Residuals for Predicted {class_names[class_idx]}')
        ax.set_xlabel('Residual Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(num_classes, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_residual_comparison(all_residuals, y_true, class_names, save_path):
    """Plot residual comparison across classes"""
    plt.figure(figsize=(12, 8))
    
    # Create box plot
    residual_data = []
    labels = []
    
    for class_idx in range(len(class_names)):
        residuals = all_residuals[:, class_idx]
        residual_data.append(residuals)
        labels.append(f'Pred {class_names[class_idx]}')
    
    plt.boxplot(residual_data, labels=labels)
    plt.title('Residual Distribution Comparison Across Predicted Classes')
    plt.xlabel('Predicted Class')
    plt.ylabel('Residual Value')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_ranking_analysis(all_residuals, y_true, class_names, save_path):
    """Plot error ranking analysis"""
    num_samples, num_classes = all_residuals.shape
    
    # For each sample, rank classes by residual
    rankings = np.argsort(all_residuals, axis=1)
    
    # Find where true class appears in ranking
    true_class_ranks = []
    for i, true_class in enumerate(y_true):
        rank = np.where(rankings[i] == true_class)[0][0] + 1  # 1-indexed
        true_class_ranks.append(rank)
    
    plt.figure(figsize=(12, 6))
    
    # Plot histogram of true class ranks
    plt.subplot(1, 2, 1)
    plt.hist(true_class_ranks, bins=range(1, num_classes + 2), alpha=0.7, edgecolor='black')
    plt.title('Distribution of True Class Ranks\n(Based on Residual Ordering)')
    plt.xlabel('Rank of True Class (1=best)')
    plt.ylabel('Number of Samples')
    plt.grid(True, alpha=0.3)
    
    # Plot cumulative accuracy
    plt.subplot(1, 2, 2)
    cumulative_acc = []
    for k in range(1, num_classes + 1):
        acc = np.mean(np.array(true_class_ranks) <= k)
        cumulative_acc.append(acc)
    
    plt.plot(range(1, num_classes + 1), cumulative_acc, 'bo-', linewidth=2, markersize=8)
    plt.title('Cumulative Top-K Accuracy')
    plt.xlabel('K (Top-K)')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Add text annotations
    for k, acc in enumerate(cumulative_acc):
        plt.annotate(f'{acc:.3f}', (k+1, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return true_class_ranks, cumulative_acc


def plot_class_wise_accuracy(y_true, y_pred, class_names, save_path):
    """Plot class-wise accuracy analysis"""
    num_classes = len(class_names)
    
    # Compute per-class metrics
    class_accuracies = []
    class_counts = []
    
    for class_idx in range(num_classes):
        mask = y_true == class_idx
        if np.sum(mask) > 0:
            class_pred = y_pred[mask]
            accuracy = np.mean(class_pred == class_idx)
            class_accuracies.append(accuracy)
            class_counts.append(np.sum(mask))
        else:
            class_accuracies.append(0)
            class_counts.append(0)
    
    plt.figure(figsize=(12, 6))
    
    # Plot class-wise accuracy
    plt.subplot(1, 2, 1)
    bars = plt.bar(range(num_classes), class_accuracies, alpha=0.7)
    plt.title('Per-Class Accuracy')
    plt.xlabel('True Class')
    plt.ylabel('Accuracy')
    plt.xticks(range(num_classes), class_names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add text annotations
    for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Plot class distribution
    plt.subplot(1, 2, 2)
    bars = plt.bar(range(num_classes), class_counts, alpha=0.7, color='orange')
    plt.title('Class Distribution in Test Set')
    plt.xlabel('True Class')
    plt.ylabel('Number of Samples')
    plt.xticks(range(num_classes), class_names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add text annotations
    for i, (bar, count) in enumerate(zip(bars, class_counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return class_accuracies, class_counts


def save_evaluation_results(results_dict, save_path):
    """Save evaluation results to JSON file"""
    # Convert to JSON-serializable format
    serializable_results = convert_to_json_serializable(results_dict)
    
    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def create_results_csv(results_dict, save_path):
    """Create a CSV summary of results"""
    summary_data = []
    
    # Overall metrics
    summary_data.append({
        'Metric': 'Overall Accuracy',
        'Value': results_dict['overall_accuracy'],
        'Description': 'Overall classification accuracy'
    })
    
    # Top-k accuracies
    for k, acc in results_dict['top_k_accuracy'].items():
        summary_data.append({
            'Metric': f'Top-{k.split("_")[1]} Accuracy',
            'Value': acc,
            'Description': f'Top-{k.split("_")[1]} classification accuracy'
        })
    
    # Per-class accuracies
    for i, acc in enumerate(results_dict['class_wise_accuracy']):
        summary_data.append({
            'Metric': f'Class {i} Accuracy',
            'Value': acc,
            'Description': f'Accuracy for class {i}'
        })
    
    # Mean residual per class
    for class_name, stats in results_dict['error_analysis'].items():
        summary_data.append({
            'Metric': f'{class_name.replace("_", " ").title()} Mean Error',
            'Value': stats['mean_error'],
            'Description': f'Mean prediction error for {class_name}'
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(save_path, index=False)


def main():
    parser = argparse.ArgumentParser(description='State-Based Dynamic System Classifier Evaluation')
    parser.add_argument('--fold', type=int, required=True, help='Fold number to evaluate')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--output_dir', type=str, default='evaluations/state_based_classifier', 
                       help='Output directory for evaluation results')
    parser.add_argument('--shared_K', action='store_true', help='Use shared K matrix across classes')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--train_window_size', type=int, default=300, 
                       help='Window size for training (fitting K and B matrices)')
    parser.add_argument('--test_window_size', type=int, default=300,
                       help='Window size for testing (classification)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir) / f'fold_{args.fold}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting state-based classifier evaluation for fold {args.fold}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    print("Loading dataset...")
    print(f"Training window size (for K,B fitting): {args.train_window_size}")
    print(f"Testing window size (for classification): {args.test_window_size}")
    
    try:
        # Training data configuration (for fitting K and B matrices)
        train_data_config = {
            'window_size': args.train_window_size,
            'sample_step': 1,
            'batch_size': 64
        }
        
        # Testing data configuration (for classification)
        test_data_config = {
            'window_size': args.test_window_size,
            'sample_step': 1,
            'batch_size': 64
        }
        
        # Load training dataset (for fitting K and B matrices)
        train_dataset, train_file_labels = load_full_dataset(args.data_dir, train_data_config)
        train_fold_dataloaders, train_test_loader = create_kfold_dataloaders(
            train_dataset, train_file_labels, train_data_config, k_folds=5, random_state=42
        )
        
        # Load testing dataset (for classification)
        test_dataset, test_file_labels = load_full_dataset(args.data_dir, test_data_config)
        test_fold_dataloaders, test_test_loader = create_kfold_dataloaders(
            test_dataset, test_file_labels, test_data_config, k_folds=5, random_state=42
        )
        
        # Get training data (for fitting K and B - use train window size)
        train_data = []
        for x, u, p in train_test_loader.dataset:
            train_data.append((x, u, p))
        
        x_train = [item[0] for item in train_data]
        u_train = [item[1] for item in train_data]  # Will be replaced with x for state-based
        et_train = [item[2] for item in train_data]
        
        # Get test data (for classification - use test window size)
        test_data = []
        for x, u, p in test_test_loader.dataset:
            test_data.append((x, u, p))
        
        x_test = [item[0] for item in test_data]
        u_test = [item[1] for item in test_data]
        et_test = [item[2] for item in test_data]
        
        print(f"Loaded training set (window={args.train_window_size}): {len(x_train)} samples")
        print(f"Loaded test set (window={args.test_window_size}): {len(x_test)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Get class names
    num_classes = len(et_test[0])
    class_names = [f'Class {i}' for i in range(num_classes)]
    
    # Convert labels to class indices
    y_true = np.array([np.argmax(et) for et in et_test])
    
    print(f"Dataset info:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Test samples: {len(x_test)}")
    print(f"  State dimension: {x_test[0].shape[1]}")
    print(f"  Class distribution: {np.bincount(y_true)}")
    
    # Compute K and B matrices using x as "control"
    print("Computing system matrices K and B using state signals as control...")
    print(f"Using training data with window size {args.train_window_size} for K,B fitting")
    try:
        # Use x in place of u for the computation (state-based approach)
        x_as_control_train = x_train  # Use state signals as "control" signals
        K_list, B_list, B_mse_list = compute_K_and_B_per_class(
            x_train, x_as_control_train, et_train, shared_K=args.shared_K
        )
        print(f"Computed {len(K_list)} K matrices and {len(B_list)} B matrices")
        print(f"B matrix MSE per class: {B_mse_list}")
    except Exception as e:
        print(f"Error computing system matrices: {e}")
        return
    
    # Evaluate classifier
    print("Evaluating state-based classifier...")
    print(f"Using test data with window size {args.test_window_size} for classification")
    try:
        predictions, all_residuals, errors_dict = evaluate_state_based_classifier(
            x_test, et_test, K_list, B_list, verbose=True
        )
        
        # Compute metrics
        overall_accuracy = accuracy_score(y_true, predictions)
        top_k_accuracy = compute_top_k_accuracy(y_true, all_residuals)
        
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
        for k, acc in top_k_accuracy.items():
            print(f"{k.replace('_', '-').title()} Accuracy: {acc:.4f}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return
    
    # Generate classification report
    try:
        class_report = classification_report(y_true, predictions, 
                                           target_names=class_names, 
                                           output_dict=True, zero_division=0)
        
        # Extract per-class accuracies
        class_accuracies = [class_report[class_name]['recall'] for class_name in class_names]
        
    except Exception as e:
        print(f"Error generating classification report: {e}")
        class_report = {}
        class_accuracies = [0] * num_classes
    
    # Analyze residual distributions
    print("Analyzing residual distributions...")
    try:
        residual_analysis = analyze_residual_distribution(all_residuals, y_true, class_names)
    except Exception as e:
        print(f"Error analyzing residuals: {e}")
        residual_analysis = {}
    
    # Create visualizations
    print("Creating visualizations...")
    try:
        # Confusion matrix
        cm = plot_confusion_matrix(y_true, predictions, class_names, 
                                 output_dir / 'confusion_matrix.png',
                                 'State-Based Classifier Confusion Matrix')
        
        # Residual distributions
        plot_residual_distributions(all_residuals, y_true, class_names,
                                  output_dir / 'residual_distributions.png')
        
        # Residual comparison
        plot_residual_comparison(all_residuals, y_true, class_names,
                               output_dir / 'residual_comparison.png')
        
        # Error ranking analysis
        true_class_ranks, cumulative_acc = plot_error_ranking_analysis(
            all_residuals, y_true, class_names, output_dir / 'error_ranking_analysis.png'
        )
        
        # Class-wise accuracy
        class_accs, class_counts = plot_class_wise_accuracy(
            y_true, predictions, class_names, output_dir / 'class_wise_accuracy.png'
        )
        
        print("All visualizations created successfully")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        cm = confusion_matrix(y_true, predictions)
        true_class_ranks = []
        cumulative_acc = []
        class_accs = class_accuracies
        class_counts = list(np.bincount(y_true))
    
    # Compile results
    results_dict = {
        'fold': args.fold,
        'evaluation_type': 'state_based_classifier',
        'overall_accuracy': float(overall_accuracy),
        'top_k_accuracy': top_k_accuracy,
        'class_wise_accuracy': [float(acc) for acc in class_accuracies],
        'class_counts': [int(count) for count in class_counts],
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
        'error_analysis': errors_dict,
        'residual_analysis': residual_analysis,
        'true_class_ranks': [int(rank) for rank in true_class_ranks],
        'cumulative_accuracy': [float(acc) for acc in cumulative_acc],
        'system_matrices': {
            'shared_K': args.shared_K,
            'num_K_matrices': len(K_list),
            'num_B_matrices': len(B_list),
            'B_mse_per_class': [float(mse) for mse in B_mse_list]
        }
    }
    
    # Save results
    print("Saving results...")
    try:
        # Save detailed results as JSON
        save_evaluation_results(results_dict, output_dir / 'evaluation_results.json')
        
        # Save summary as CSV
        create_results_csv(results_dict, output_dir / 'evaluation_summary.csv')
        
        print(f"Results saved to {output_dir}")
        print(f"  - evaluation_results.json: Detailed results")
        print(f"  - evaluation_summary.csv: Summary metrics")
        print(f"  - *.png: Visualization plots")
        
    except Exception as e:
        print(f"Error saving results: {e}")
    
    print("State-based classifier evaluation completed successfully!")


if __name__ == '__main__':
    main()
