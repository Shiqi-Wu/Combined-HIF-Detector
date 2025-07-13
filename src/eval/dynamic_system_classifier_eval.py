#!/usr/bin/env python3
"""
Dynamic System Classifier Evaluation

This script implements a novel approach for fault classification that combines:
1. Linear dynamic system modeling (x[t+1] = K*x[t] + B(p)*u[t])
2. Seq2Seq model for control signal generation
3. Residual-based classification

The approach:
- Computes a shared system matrix K using all data
- Estimates class-specific B matrices for each fault type
- Uses a seq2seq model to generate control signals u from state signals x
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
from models.dynamic_system_classifier import DynamicSystemClassifier, compute_K_and_B_per_class
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
        x, u, p = self.scaled_dataset[idx]
        return x, u, p


def load_preprocessing_params(preprocessing_params_path: str) -> Dict:
    """Load preprocessing parameters"""
    print(f"Loading preprocessing parameters from {preprocessing_params_path}")
    
    if not os.path.exists(preprocessing_params_path):
        raise FileNotFoundError(f"Preprocessing parameters not found: {preprocessing_params_path}")
    
    with open(preprocessing_params_path, 'rb') as f:
        params = pickle.load(f)
    
    return params


def load_and_prepare_data(data_dir: str, data_config: Dict, preprocessing_params_path: str,
                         k_folds: int = 5, fold_idx: int = 0, random_state: int = 42) -> Tuple:
    """Load and preprocess data for evaluation"""
    print("\n===== Loading Dataset =====")
    
    # Load full dataset
    dataset, file_labels = load_full_dataset(data_dir, data_config)
    print(f"Loaded dataset with {len(dataset)} samples from {len(file_labels)} files")
    
    # Create K-fold splits
    fold_dataloaders, test_loader = create_kfold_dataloaders(
        dataset, file_labels, data_config, k_folds, random_state
    )
    
    # Load preprocessing parameters
    preprocessing_params = load_preprocessing_params(preprocessing_params_path)
    
    # Apply preprocessing to test set
    test_scaled = ScaledDataset(test_loader.dataset, pca_dim=2, fit_scalers=False)
    test_scaled.set_preprocessing_params(preprocessing_params)
    test_dataset = DynamicSystemDataset(test_scaled)
    
    # Apply preprocessing to selected fold
    train_loader, val_loader = fold_dataloaders[fold_idx]
    
    train_scaled = ScaledDataset(train_loader.dataset, pca_dim=2, fit_scalers=False)
    train_scaled.set_preprocessing_params(preprocessing_params)
    train_dataset = DynamicSystemDataset(train_scaled)
    
    val_scaled = ScaledDataset(val_loader.dataset, pca_dim=2, fit_scalers=False)
    val_scaled.set_preprocessing_params(preprocessing_params)
    val_dataset = DynamicSystemDataset(val_scaled)
    
    print(f"Data loaded for fold {fold_idx + 1}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    # Print examples from each dataset
    print("\nExample data:")
    
    # Print first sample from train dataset
    x_train, u_train, p_train = train_dataset[0]
    print(f"  Train example:")
    print(f"    State vector shape: {x_train.shape}")
    print(f"    Control vector shape: {u_train.shape}")
    print(f"    Class label: {p_train}")
    print(f"    First state vector: {x_train[0]}")
    
    # Print first sample from validation dataset
    x_val, u_val, p_val = val_dataset[0]
    print(f"  Validation example:")
    print(f"    State vector shape: {x_val.shape}")
    print(f"    Control vector shape: {u_val.shape}")
    print(f"    Class label: {p_val}")
    print(f"    First state vector: {x_val[0]}")
    
    # Print first sample from test dataset
    x_test, u_test, p_test = test_dataset[0]
    print(f"  Test example:")
    print(f"    State vector shape: {x_test.shape}")
    print(f"    Control vector shape: {u_test.shape}")
    print(f"    Class label: {p_test}")
    print(f"    First state vector: {x_test[0]}")
    
    return train_dataset, val_dataset, test_dataset


def plot_confusion_matrix(results: Dict, dataset_name: str, num_classes: int,
                         save_path: Optional[str] = None, save_data_path: Optional[str] = None) -> Dict:
    """Plot confusion matrix"""
    true_labels = results['true_labels']
    predicted_labels = results['predictions']
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Prepare data for saving
    cm_data = {
        'confusion_matrix': cm.tolist(),
        'true_labels': true_labels.tolist() if hasattr(true_labels, 'tolist') else list(true_labels),
        'predicted_labels': predicted_labels.tolist() if hasattr(predicted_labels, 'tolist') else list(predicted_labels),
        'class_names': [f"Class {i}" for i in range(num_classes)],
        'dataset_name': dataset_name
    }
    
    # Save data to JSON
    if save_data_path:
        with open(save_data_path, 'w') as f:
            json.dump(cm_data, f, indent=2)
        print(f"Confusion matrix data saved to {save_data_path}")
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=[f"Class {i}" for i in range(num_classes)],
               yticklabels=[f"Class {i}" for i in range(num_classes)])
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    
    return cm_data


def plot_prediction_errors(results: Dict, dataset_name: str,
                          save_path: Optional[str] = None, save_data_path: Optional[str] = None) -> Dict:
    """Plot prediction errors distribution"""
    errors = results['prediction_errors']
    true_labels = results['true_labels']
    predictions = results['predictions']
    
    # Create DataFrame for plotting
    plot_data = []
    for i, (true_class, pred_class, error_vec) in enumerate(zip(true_labels, predictions, errors)):
        for class_idx, error in enumerate(error_vec):
            plot_data.append({
                'Sample': int(i),
                'True_Class': f"True_Class_{int(true_class)}",
                'Predicted_Class': f"Class_{int(class_idx)}",
                'Error': float(error),  # Convert to Python float
                'Is_Correct': bool(class_idx == true_class)
            })
    
    df = pd.DataFrame(plot_data)
    
    # Prepare aggregated data for saving
    error_data = {
        'raw_data': plot_data,
        'dataset_name': dataset_name,
        'summary_stats': {
            'error_by_true_class': {k: {sk: float(sv) for sk, sv in v.items()} for k, v in df.groupby('True_Class')['Error'].describe().to_dict().items()},
            'error_by_predicted_class': {k: {sk: float(sv) for sk, sv in v.items()} for k, v in df.groupby('Predicted_Class')['Error'].describe().to_dict().items()},
            'error_by_correctness': {k: {sk: float(sv) for sk, sv in v.items()} for k, v in df.groupby('Is_Correct')['Error'].describe().to_dict().items()},
            'error_heatmap': {k: {sk: float(sv) for sk, sv in v.items()} for k, v in df.groupby(['True_Class', 'Predicted_Class'])['Error'].mean().unstack().fillna(0).to_dict().items()}
        }
    }
    
    # Save data to JSON
    if save_data_path:
        with open(save_data_path, 'w') as f:
            json.dump(error_data, f, indent=2)
        print(f"Prediction errors data saved to {save_data_path}")
    
    # Plot
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Error distribution by true class
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df, x='True_Class', y='Error')
    plt.title('Error Distribution by True Class')
    plt.xticks(rotation=45)
    plt.yscale('log')
    
    # Subplot 2: Error distribution by predicted class
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df, x='Predicted_Class', y='Error')
    plt.title('Error Distribution by Predicted Class')
    plt.xticks(rotation=45)
    plt.yscale('log')
    
    # Subplot 3: Correct vs incorrect predictions
    plt.subplot(2, 2, 3)
    sns.boxplot(data=df, x='Is_Correct', y='Error')
    plt.title('Error Distribution: Correct vs Incorrect')
    plt.yscale('log')
    
    # Subplot 4: Error heatmap
    plt.subplot(2, 2, 4)
    pivot_data = df.groupby(['True_Class', 'Predicted_Class'])['Error'].mean().unstack()
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Average Error Heatmap')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error analysis plot saved to {save_path}")
    
    plt.show()
    
    return error_data


def calculate_top_k_accuracy(prediction_errors: np.ndarray, true_labels: np.ndarray, 
                             k_values: List[int] = [1, 2, 3]) -> Dict[int, float]:
    """
    Calculate top-k accuracy based on prediction errors
    
    Args:
        prediction_errors: Array of shape (n_samples, n_classes) with prediction errors
        true_labels: Array of true class labels
        k_values: List of k values to calculate top-k accuracy for
        
    Returns:
        Dictionary mapping k to top-k accuracy
    """
    # Convert errors to probabilities (lower error = higher probability)
    # Use negative errors and apply softmax
    prob_scores = -prediction_errors
    
    top_k_accuracies = {}
    for k in k_values:
        if k <= prob_scores.shape[1]:  # Ensure k doesn't exceed number of classes
            top_k_acc = top_k_accuracy_score(true_labels, prob_scores, k=k, labels=range(prob_scores.shape[1]))
            top_k_accuracies[k] = top_k_acc
        else:
            top_k_accuracies[k] = top_k_accuracies.get(1, 0.0)  # If k > n_classes, use top-1
    
    return top_k_accuracies


def plot_top_k_accuracy_comparison(all_results: Dict[str, Dict], 
                                  k_values: List[int] = [1, 2, 3],
                                  save_path: Optional[str] = None, save_data_path: Optional[str] = None) -> Dict:
    """
    Plot top-k accuracy comparison across different datasets
    
    Args:
        all_results: Dictionary with results for different datasets
        k_values: List of k values to plot
        save_path: Path to save the plot
        save_data_path: Path to save the data
    """
    # Prepare data for plotting
    plot_data = []
    top_k_data = {}
    
    for dataset_name, results in all_results.items():
        top_k_accs = calculate_top_k_accuracy(
            results['prediction_errors'], 
            results['true_labels'], 
            k_values
        )
        
        top_k_data[dataset_name] = top_k_accs
        
        for k, accuracy in top_k_accs.items():
            plot_data.append({
                'Dataset': dataset_name.capitalize(),
                'K': f'Top-{int(k)}',
                'Accuracy': float(accuracy)  # Convert to Python float
            })
    
    df = pd.DataFrame(plot_data)
    
    # Prepare data for saving
    topk_save_data = {
        'top_k_accuracies': {k: {sk: float(sv) for sk, sv in v.items()} for k, v in top_k_data.items()},
        'plot_data': plot_data,
        'k_values': [int(k) for k in k_values],
        'summary': {
            'best_performance': {
                'overall_best': {k: (float(v) if isinstance(v, (int, float, np.number)) else str(v)) for k, v in df.loc[df['Accuracy'].idxmax()].to_dict().items()},
                'best_by_dataset': {k: float(v) for k, v in df.groupby('Dataset')['Accuracy'].max().to_dict().items()},
                'best_by_k': {k: float(v) for k, v in df.groupby('K')['Accuracy'].max().to_dict().items()}
            }
        }
    }
    
    # Save data to JSON
    if save_data_path:
        with open(save_data_path, 'w') as f:
            json.dump(topk_save_data, f, indent=2)
        print(f"Top-K accuracy data saved to {save_data_path}")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Bar plot
    plt.subplot(2, 1, 1)
    sns.barplot(data=df, x='K', y='Accuracy', hue='Dataset')
    plt.title('Top-K Accuracy Comparison Across Datasets')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    ax = plt.gca()
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=9)
    
    plt.legend(title='Dataset')
    
    # Line plot
    plt.subplot(2, 1, 2)
    for dataset_name in all_results.keys():
        dataset_data = df[df['Dataset'] == dataset_name.capitalize()]
        k_nums = [int(k.split('-')[1]) for k in dataset_data['K']]
        accuracies = dataset_data['Accuracy'].values
        plt.plot(k_nums, accuracies, marker='o', label=dataset_name.capitalize(), linewidth=2)
    
    plt.title('Top-K Accuracy Trends')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend(title='Dataset')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Top-K accuracy plot saved to {save_path}")
    
    plt.show()
    
    return topk_save_data


def plot_error_ranking_analysis(results: Dict, dataset_name: str,
                               save_path: Optional[str] = None, save_data_path: Optional[str] = None) -> Dict:
    """
    Plot analysis of error rankings for each sample
    
    Args:
        results: Results dictionary containing prediction errors and true labels
        dataset_name: Name of the dataset
        save_path: Path to save the plot
        save_data_path: Path to save the data
    """
    errors = results['prediction_errors']
    true_labels = results['true_labels']
    
    # Calculate ranking of true class for each sample
    true_class_ranks = []
    for i, (error_vec, true_class) in enumerate(zip(errors, true_labels)):
        # Rank classes by error (ascending order, so rank 1 = lowest error = best)
        ranked_classes = np.argsort(error_vec)
        true_class_rank = np.where(ranked_classes == true_class)[0][0] + 1  # +1 for 1-based ranking
        true_class_ranks.append(true_class_rank)
    
    true_class_ranks = np.array(true_class_ranks)
    
    # Calculate cumulative accuracy by rank
    max_rank = len(errors[0])
    cumulative_acc = []
    for k in range(1, max_rank + 1):
        acc_at_k = np.mean(true_class_ranks <= k)
        cumulative_acc.append(acc_at_k)
    
    # Calculate ranking by true class
    rank_by_class = {}
    for true_class in range(len(errors[0])):
        class_mask = true_labels == true_class
        if np.any(class_mask):
            rank_by_class[f'Class_{int(true_class)}'] = [int(r) for r in true_class_ranks[class_mask].tolist()]
    
    # Calculate error gaps
    error_gaps = []
    for error_vec in errors:
        sorted_errors = np.sort(error_vec)
        if len(sorted_errors) >= 2:
            gap = float(sorted_errors[1] - sorted_errors[0])  # Convert to Python float
            error_gaps.append(gap)
    
    # Prepare data for saving
    ranking_data = {
        'dataset_name': dataset_name,
        'true_class_ranks': [int(r) for r in true_class_ranks.tolist()],
        'cumulative_accuracy': {
            'ranks': list(range(1, max_rank + 1)),
            'accuracies': [float(acc) for acc in cumulative_acc]
        },
        'rank_by_class': rank_by_class,
        'error_gaps': error_gaps,
        'summary_stats': {
            'rank_distribution': {
                'mean_rank': float(np.mean(true_class_ranks)),
                'median_rank': float(np.median(true_class_ranks)),
                'std_rank': float(np.std(true_class_ranks)),
                'rank_1_accuracy': float(np.mean(true_class_ranks == 1)),
                'top_3_accuracy': float(np.mean(true_class_ranks <= 3))
            },
            'error_gap_stats': {
                'mean_gap': float(np.mean(error_gaps)) if error_gaps else 0.0,
                'median_gap': float(np.median(error_gaps)) if error_gaps else 0.0,
                'std_gap': float(np.std(error_gaps)) if error_gaps else 0.0
            }
        }
    }
    
    # Save data to JSON
    if save_data_path:
        with open(save_data_path, 'w') as f:
            json.dump(ranking_data, f, indent=2)
        print(f"Error ranking analysis data saved to {save_data_path}")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Subplot 1: Histogram of true class rankings
    axes[0, 0].hist(true_class_ranks, bins=range(1, len(errors[0]) + 2), alpha=0.7, edgecolor='black')
    axes[0, 0].set_title(f'Distribution of True Class Rankings - {dataset_name}')
    axes[0, 0].set_xlabel('Rank of True Class')
    axes[0, 0].set_ylabel('Number of Samples')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Subplot 2: Cumulative accuracy by rank
    axes[0, 1].plot(range(1, max_rank + 1), cumulative_acc, marker='o', linewidth=2)
    axes[0, 1].set_title(f'Cumulative Top-K Accuracy - {dataset_name}')
    axes[0, 1].set_xlabel('K (Rank)')
    axes[0, 1].set_ylabel('Cumulative Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # Subplot 3: Ranking by true class
    if rank_by_class:
        rank_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in rank_by_class.items()]))
        rank_df_melted = rank_df.melt(var_name='True_Class', value_name='Rank')
        rank_df_melted = rank_df_melted.dropna()
        
        sns.boxplot(data=rank_df_melted, x='True_Class', y='Rank', ax=axes[1, 0])
        axes[1, 0].set_title(f'Ranking Distribution by True Class - {dataset_name}')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Subplot 4: Error gap analysis
    if error_gaps:
        axes[1, 1].hist(error_gaps, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title(f'Error Gap Distribution - {dataset_name}')
        axes[1, 1].set_xlabel('Error Gap (2nd Best - Best)')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error ranking analysis plot saved to {save_path}")
    
    plt.show()
    
    return ranking_data


def run_evaluation(data_dir: str, seq2seq_model_path: str, preprocessing_params_path: str,
                  data_config: Dict, seq2seq_model_config: Dict, 
                  fold_idx: int = 0, k_folds: int = 5, random_state: int = 42,
                  save_dir: Optional[str] = None) -> Dict:
    """Run complete evaluation pipeline"""
    print("="*60)
    print("STARTING DYNAMIC SYSTEM CLASSIFIER EVALUATION")
    print("="*60)
    
    # Create save directory
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    train_dataset, val_dataset, test_dataset = load_and_prepare_data(
        data_dir, data_config, preprocessing_params_path, k_folds, fold_idx, random_state
    )
    
    # Create classifier
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = DynamicSystemClassifier(device=device)
    
    # Load seq2seq model
    classifier.load_seq2seq_model(seq2seq_model_path, seq2seq_model_config)
    
    # Train system model on training data
    print("\n===== Training Dynamic System Model =====")
    classifier.fit_dynamic_system(train_dataset, shared_K=True, verbose=True)
    
    # Evaluate on all datasets
    results = {}
    
    # Training set evaluation
    print("\n===== Evaluating on Training Set =====")
    results['train'] = classifier.evaluate_dataset(train_dataset, verbose=True)
    
    # Validation set evaluation
    print("\n===== Evaluating on Validation Set =====")
    results['val'] = classifier.evaluate_dataset(val_dataset, verbose=True)
    
    # Test set evaluation
    print("\n===== Evaluating on Test Set =====")
    results['test'] = classifier.evaluate_dataset(test_dataset, verbose=True)
    
    # Plot results and collect plot data
    all_plot_data = {}
    
    for dataset_name, result in results.items():
        # Confusion matrix
        if save_dir:
            cm_path = Path(save_dir) / f'confusion_matrix_{dataset_name}.png'
            cm_data_path = Path(save_dir) / f'confusion_matrix_{dataset_name}_data.json'
        else:
            cm_path = None
            cm_data_path = None
        cm_data = plot_confusion_matrix(result, dataset_name.capitalize(), classifier.num_classes, cm_path, cm_data_path)
        
        # Error analysis
        if save_dir:
            error_path = Path(save_dir) / f'error_analysis_{dataset_name}.png'
            error_data_path = Path(save_dir) / f'error_analysis_{dataset_name}_data.json'
        else:
            error_path = None
            error_data_path = None
        error_data = plot_prediction_errors(result, dataset_name.capitalize(), error_path, error_data_path)
        
        # Error ranking analysis
        if save_dir:
            ranking_path = Path(save_dir) / f'error_ranking_{dataset_name}.png'
            ranking_data_path = Path(save_dir) / f'error_ranking_{dataset_name}_data.json'
        else:
            ranking_path = None
            ranking_data_path = None
        ranking_data = plot_error_ranking_analysis(result, dataset_name.capitalize(), ranking_path, ranking_data_path)
        
        # Collect all plot data for this dataset
        all_plot_data[dataset_name] = {
            'confusion_matrix': cm_data,
            'error_analysis': error_data,
            'ranking_analysis': ranking_data
        }
    
    # Top-K accuracy comparison across all datasets
    if save_dir:
        topk_path = Path(save_dir) / 'top_k_accuracy_comparison.png'
        topk_data_path = Path(save_dir) / 'top_k_accuracy_comparison_data.json'
    else:
        topk_path = None
        topk_data_path = None
    topk_data = plot_top_k_accuracy_comparison(results, k_values=[1, 2, 3, 4, 5, 6], save_path=topk_path, save_data_path=topk_data_path)
    
    all_plot_data['top_k_comparison'] = topk_data
    
    # Save results
    if save_dir:
        results_file = Path(save_dir) / 'evaluation_results.json'
        all_plot_data_file = Path(save_dir) / 'all_plot_data.json'
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for dataset_name, result in results.items():
            # Calculate top-k accuracies
            top_k_accs = calculate_top_k_accuracy(
                result['prediction_errors'], 
                result['true_labels'], 
                k_values=[1, 2, 3, 4, 5, 6]
            )
            
            json_results[dataset_name] = {
                'accuracy': float(result['accuracy']),
                'top_k_accuracies': {f'top_{k}': float(acc) for k, acc in top_k_accs.items()},
                'classification_report': result['classification_report']
            }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"Results saved to {results_file}")
        
        # Save all plot data in one comprehensive file
        with open(all_plot_data_file, 'w') as f:
            json.dump(all_plot_data, f, indent=2)
        print(f"All plot data saved to {all_plot_data_file}")
    
    # Print summary including top-k accuracies
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for dataset_name, result in results.items():
        top_k_accs = calculate_top_k_accuracy(
            result['prediction_errors'], 
            result['true_labels'], 
            k_values=[1, 2, 3, 5]
        )
        print(f"{dataset_name.capitalize()} Results:")
        print(f"  Top-1 Accuracy: {result['accuracy']:.4f}")
        for k, acc in top_k_accs.items():
            if k > 1:
                print(f"  Top-{k} Accuracy: {acc:.4f}")
        print()
    print("="*60)
    
    return results


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Dynamic System Classifier Evaluation')
    
    # Required arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing original data')
    parser.add_argument('--seq2seq_model', type=str, required=True,
                       help='Path to trained seq2seq model')
    
    # Optional arguments
    parser.add_argument('--preprocessing_params', type=str, 
                       default='/home/shiqi_w/code/Combined-HIF-detector/preprocessing_params_fold.pkl',
                       help='Path to preprocessing parameters pickle file')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save evaluation results and plots')
    parser.add_argument('--fold_idx', type=int, default=0,
                       help='Fold index to use for evaluation')
    parser.add_argument('--k_folds', type=int, default=5,
                       help='Number of folds')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed')
    
    # Data parameters
    parser.add_argument('--window_size', type=int, default=30,
                       help='Window size')
    parser.add_argument('--sample_step', type=int, default=1,
                       help='Sampling step')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    
    # Seq2seq model parameters
    parser.add_argument('--state_dim', type=int, default=2,
                       help='State dimension (after PCA)')
    parser.add_argument('--control_dim', type=int, default=2,
                       help='Control dimension')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--bidirectional', action='store_true', default=True,
                       help='Use bidirectional LSTM')
    
    # Top-K accuracy parameters
    parser.add_argument('--top_k_values', nargs='+', type=int, default=[1, 2, 3, 5],
                       help='K values for top-k accuracy calculation')
    
    args = parser.parse_args()
    
    # Configuration
    data_config = {
        'window_size': args.window_size,
        'sample_step': args.sample_step,
        'batch_size': args.batch_size
    }
    
    seq2seq_model_config = {
        'state_dim': args.state_dim,
        'control_dim': args.control_dim,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'bidirectional': args.bidirectional
    }
    
    # Run evaluation
    results = run_evaluation(
        data_dir=args.data_dir,
        seq2seq_model_path=args.seq2seq_model,
        preprocessing_params_path=args.preprocessing_params,
        data_config=data_config,
        seq2seq_model_config=seq2seq_model_config,
        fold_idx=args.fold_idx,
        k_folds=args.k_folds,
        random_state=args.random_state,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
