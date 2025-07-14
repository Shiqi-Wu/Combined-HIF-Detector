#!/usr/bin/env python3
"""
Known Control Dynamic System Classifier Evaluation

This script implements fault classification using known control signals:
- Linear dynamic system modeling: x[t+1] = K*x[t] + B(p)*u[t]
- Uses ground truth control signals u for classification
- Classifies by finding the class that minimizes prediction error

The approach:
- Computes a shared system matrix K using all data
- Estimates class-specific B matrices for each fault type
- Uses known control signals u from dataset
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
        x, u, p = self.scaled_dataset[idx]
        return x, u, p


class KnownControlClassifier:
    """
    Dynamic System-based Fault Classifier using known control signals
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.K_matrices = None
        self.B_matrices = None
        self.B_mse_list = None
        self.num_classes = 6
        self.class_names = ['Fault_1', 'Fault_2', 'Fault_3', 'Fault_4', 'Fault_5', 'Fault_6']
        
    def fit_dynamic_system(self, train_dataset, shared_K=True, verbose=True):
        """Fit the linear dynamic system parameters"""
        if verbose:
            print("Fitting dynamic system parameters with known control signals...")
        
        # Extract data from dataset
        x_data = []
        u_data = []
        et_data = []
        
        for i in range(len(train_dataset)):
            x_batch, u_batch, p_batch = train_dataset[i]
            x_data.append(x_batch.numpy())
            u_data.append(u_batch.numpy())
            et_data.append([p_batch.numpy()])
        
        # Compute K and B matrices
        self.K_matrices, self.B_matrices, self.B_mse_list = compute_K_and_B_per_class(
            x_data, u_data, et_data, shared_K=shared_K
        )
        
        if verbose:
            print(f"Fitted {len(self.K_matrices)} K matrices and {len(self.B_matrices)} B matrices")
            print(f"Training MSE per class: {[f'{mse:.6f}' for mse in self.B_mse_list]}")
    
    def predict_single_sequence(self, x_sequence, u_sequence):
        """Predict class for a single sequence using known control signals"""
        if self.K_matrices is None or self.B_matrices is None:
            raise ValueError("Dynamic system not fitted. Call fit_dynamic_system first.")
        
        # Compute prediction errors for each class
        prediction_errors = []
        
        x_current = x_sequence[:-1]  # x[0:T-1]
        x_next = x_sequence[1:]      # x[1:T]
        u_current = u_sequence[:-1]  # u[0:T-1] - use known control signals
        
        for class_idx in range(self.num_classes):
            K = self.K_matrices[class_idx]
            B = self.B_matrices[class_idx]
            
            # Predict next states: x[t+1] = K*x[t] + B*u[t]
            x_pred = x_current @ K + u_current @ B
            
            # Compute prediction error
            error = np.mean((x_next - x_pred) ** 2)
            prediction_errors.append(error)
        
        # Class with minimum error
        predicted_class = np.argmin(prediction_errors)
        
        return predicted_class, prediction_errors
    
    def evaluate_dataset(self, dataset, verbose=True):
        """Evaluate the classifier on a dataset using known control signals"""
        if verbose:
            print(f"Evaluating on dataset with {len(dataset)} samples using known control signals...")
        
        # Extract all data
        all_x = []
        all_u = []
        all_labels = []
        
        for i in range(len(dataset)):
            x_batch, u_batch, p_batch = dataset[i]
            all_x.append(x_batch.numpy())
            all_u.append(u_batch.numpy())  # Use known control signals
            
            # Convert one-hot to class index
            if p_batch.dim() > 0 and len(p_batch.shape) > 0:
                if p_batch.shape[-1] > 1:
                    label = torch.argmax(p_batch).item()
                else:
                    label = p_batch.item()
            else:
                label = 0
            all_labels.append(label)
        
        all_x = np.array(all_x)
        all_u = np.array(all_u)
        all_labels = np.array(all_labels)
        
        # Make predictions
        predictions = []
        prediction_errors = []
        
        for i in range(len(all_x)):
            pred_class, errors = self.predict_single_sequence(all_x[i], all_u[i])
            predictions.append(pred_class)
            prediction_errors.append(errors)
        
        predictions = np.array(predictions)
        prediction_errors = np.array(prediction_errors)
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, predictions)
        
        # Classification report
        class_report = classification_report(
            all_labels, predictions, 
            target_names=self.class_names, 
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(all_labels, predictions)
        
        results = {
            'accuracy': accuracy,
            'predictions': predictions,
            'true_labels': all_labels,
            'prediction_errors': prediction_errors,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix
        }
        
        if verbose:
            print(f"Accuracy: {accuracy:.4f}")
        
        return results


def load_preprocessing_params(preprocessing_params_path: str) -> Dict:
    """Load preprocessing parameters"""
    print(f"Loading preprocessing parameters from {preprocessing_params_path}")
    
    if not os.path.exists(preprocessing_params_path):
        raise FileNotFoundError(f"Preprocessing parameters not found: {preprocessing_params_path}")
    
    with open(preprocessing_params_path, 'rb') as f:
        params = pickle.load(f)
    
    return params


def load_and_prepare_data(data_dir: str, train_data_config: Dict, test_data_config: Dict, 
                         preprocessing_params_path: str, k_folds: int = 5, fold_idx: int = 0, 
                         random_state: int = 42) -> Tuple:
    """Load and preprocess data for evaluation with separate train/test window sizes"""
    print("\n===== Loading Dataset =====")
    print(f"Training window size: {train_data_config['window_size']}")
    print(f"Testing window size: {test_data_config['window_size']}")
    
    # Load dataset for training (fitting K and B matrices)
    train_dataset, train_file_labels = load_full_dataset(data_dir, train_data_config)
    print(f"Loaded training dataset with {len(train_dataset)} samples from {len(train_file_labels)} files")
    
    # Load dataset for testing (classification)
    test_dataset_full, test_file_labels = load_full_dataset(data_dir, test_data_config)
    print(f"Loaded testing dataset with {len(test_dataset_full)} samples from {len(test_file_labels)} files")
    
    # Create K-fold splits for training data
    train_fold_dataloaders, train_test_loader = create_kfold_dataloaders(
        train_dataset, train_file_labels, train_data_config, k_folds, random_state
    )
    
    # Create K-fold splits for testing data (for evaluation)
    test_fold_dataloaders, test_test_loader = create_kfold_dataloaders(
        test_dataset_full, test_file_labels, test_data_config, k_folds, random_state
    )
    
    # Load preprocessing parameters
    preprocessing_params = load_preprocessing_params(preprocessing_params_path)
    
    # Apply preprocessing to training data (for fitting K and B)
    train_loader, val_loader = train_fold_dataloaders[fold_idx]
    
    train_scaled = ScaledDataset(train_loader.dataset, pca_dim=2, fit_scalers=False)
    train_scaled.set_preprocessing_params(preprocessing_params)
    train_dataset_processed = DynamicSystemDataset(train_scaled)
    
    train_val_scaled = ScaledDataset(val_loader.dataset, pca_dim=2, fit_scalers=False)
    train_val_scaled.set_preprocessing_params(preprocessing_params)
    train_val_dataset = DynamicSystemDataset(train_val_scaled)
    
    # Apply preprocessing to testing data (for classification)
    test_train_loader, test_val_loader = test_fold_dataloaders[fold_idx]
    
    test_train_scaled = ScaledDataset(test_train_loader.dataset, pca_dim=2, fit_scalers=False)
    test_train_scaled.set_preprocessing_params(preprocessing_params)
    test_train_dataset = DynamicSystemDataset(test_train_scaled)
    
    test_val_scaled = ScaledDataset(test_val_loader.dataset, pca_dim=2, fit_scalers=False)
    test_val_scaled.set_preprocessing_params(preprocessing_params)
    test_val_dataset = DynamicSystemDataset(test_val_scaled)
    
    test_test_scaled = ScaledDataset(test_test_loader.dataset, pca_dim=2, fit_scalers=False)
    test_test_scaled.set_preprocessing_params(preprocessing_params)
    test_test_dataset = DynamicSystemDataset(test_test_scaled)
    
    print(f"Data loaded for fold {fold_idx + 1}")
    print(f"  Training data (for fitting K,B): Train={len(train_dataset_processed)}, Val={len(train_val_dataset)}")
    print(f"  Testing data (for classification): Train={len(test_train_dataset)}, Val={len(test_val_dataset)}, Test={len(test_test_dataset)}")
    
    return (train_dataset_processed, train_val_dataset, 
            test_train_dataset, test_val_dataset, test_test_dataset)


# Import plotting functions from the main evaluation script
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
    plt.title(f'Confusion Matrix - {dataset_name} (Known Control)')
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
                'Error': float(error),
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
    plt.title(f'Error Distribution by True Class - {dataset_name} (Known Control)')
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
    """Calculate top-k accuracy based on prediction errors"""
    prob_scores = -prediction_errors
    
    top_k_accuracies = {}
    for k in k_values:
        if k <= prob_scores.shape[1]:
            top_k_acc = top_k_accuracy_score(true_labels, prob_scores, k=k, labels=range(prob_scores.shape[1]))
            top_k_accuracies[k] = top_k_acc
        else:
            top_k_accuracies[k] = top_k_accuracies.get(1, 0.0)
    
    return top_k_accuracies


def plot_top_k_accuracy_comparison(all_results: Dict[str, Dict], 
                                  k_values: List[int] = [1, 2, 3],
                                  save_path: Optional[str] = None, save_data_path: Optional[str] = None) -> Dict:
    """Plot top-k accuracy comparison across different datasets"""
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
                'Accuracy': float(accuracy)
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
    plt.title('Top-K Accuracy Comparison Across Datasets (Known Control)')
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
    """Plot analysis of error rankings for each sample"""
    errors = results['prediction_errors']
    true_labels = results['true_labels']
    
    # Calculate ranking of true class for each sample
    true_class_ranks = []
    for i, (error_vec, true_class) in enumerate(zip(errors, true_labels)):
        ranked_classes = np.argsort(error_vec)
        true_class_rank = np.where(ranked_classes == true_class)[0][0] + 1
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
            gap = float(sorted_errors[1] - sorted_errors[0])
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
    axes[0, 0].set_title(f'Distribution of True Class Rankings - {dataset_name} (Known Control)')
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


def save_results_to_csv(all_results: Dict, save_dir: str):
    """Save classification results to CSV file"""
    csv_data = []
    
    for dataset_name, results in all_results.items():
        # Calculate top-k accuracies
        top_k_accs = calculate_top_k_accuracy(
            results['prediction_errors'], 
            results['true_labels'], 
            k_values=[1, 2, 3, 4, 5, 6]
        )
        
        # Add row to CSV data
        row = {
            'Dataset': dataset_name.capitalize(),
            'Method': 'Known_Control_Classifier',
            'Top_1_Accuracy': float(results['accuracy']),
            'Top_2_Accuracy': float(top_k_accs.get(2, 0.0)),
            'Top_3_Accuracy': float(top_k_accs.get(3, 0.0)),
            'Top_4_Accuracy': float(top_k_accs.get(4, 0.0)),
            'Top_5_Accuracy': float(top_k_accs.get(5, 0.0)),
            'Top_6_Accuracy': float(top_k_accs.get(6, 0.0)),
            'Total_Samples': len(results['true_labels'])
        }
        csv_data.append(row)
    
    # Save to CSV
    df = pd.DataFrame(csv_data)
    csv_path = Path(save_dir) / 'known_control_classification_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"Classification results saved to {csv_path}")
    
    return df


def run_evaluation(data_dir: str, preprocessing_params_path: str,
                  train_data_config: Dict, test_data_config: Dict, 
                  fold_idx: int = 0, k_folds: int = 5, 
                  random_state: int = 42, save_dir: Optional[str] = None) -> Dict:
    """Run complete evaluation pipeline with separate train/test window sizes"""
    print("="*60)
    print("KNOWN CONTROL DYNAMIC SYSTEM CLASSIFIER EVALUATION")
    print("="*60)
    print(f"Training window size (for K,B fitting): {train_data_config['window_size']}")
    print(f"Testing window size (for classification): {test_data_config['window_size']}")
    
    # Create save directory
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    (train_dataset, train_val_dataset, 
     test_train_dataset, test_val_dataset, test_test_dataset) = load_and_prepare_data(
        data_dir, train_data_config, test_data_config, preprocessing_params_path, 
        k_folds, fold_idx, random_state
    )
    
    # Create classifier
    classifier = KnownControlClassifier()
    
    # Train system model on training data (with training window size)
    print("\n===== Training Dynamic System Model =====")
    print(f"Using training window size: {train_data_config['window_size']} for fitting K and B matrices")
    classifier.fit_dynamic_system(train_dataset, shared_K=True, verbose=True)
    
    # Evaluate on all datasets (with testing window size)
    results = {}
    
    # Training set evaluation (using test window size for classification)
    print(f"\n===== Evaluating on Training Set (window_size={test_data_config['window_size']}) =====")
    results['train'] = classifier.evaluate_dataset(test_train_dataset, verbose=True)
    
    # Validation set evaluation
    print(f"\n===== Evaluating on Validation Set (window_size={test_data_config['window_size']}) =====")
    results['val'] = classifier.evaluate_dataset(test_val_dataset, verbose=True)
    
    # Test set evaluation
    print(f"\n===== Evaluating on Test Set (window_size={test_data_config['window_size']}) =====")
    results['test'] = classifier.evaluate_dataset(test_test_dataset, verbose=True)
    
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
        
        # Save results to CSV
        save_results_to_csv(results, save_dir)
    
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
    parser = argparse.ArgumentParser(description='Known Control Dynamic System Classifier Evaluation')
    
    # Required arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing original data')
    
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
    parser.add_argument('--train_window_size', type=int, default=300,
                       help='Window size for training (fitting K and B matrices)')
    parser.add_argument('--test_window_size', type=int, default=300,
                       help='Window size for testing (classification)')
    parser.add_argument('--sample_step', type=int, default=1,
                       help='Sampling step')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    
    args = parser.parse_args()
    
    # Configuration
    train_data_config = {
        'window_size': args.train_window_size,
        'sample_step': args.sample_step,
        'batch_size': args.batch_size
    }
    
    test_data_config = {
        'window_size': args.test_window_size,
        'sample_step': args.sample_step,
        'batch_size': args.batch_size
    }
    
    # Run evaluation
    results = run_evaluation(
        data_dir=args.data_dir,
        preprocessing_params_path=args.preprocessing_params,
        train_data_config=train_data_config,
        test_data_config=test_data_config,
        fold_idx=args.fold_idx,
        k_folds=args.k_folds,
        random_state=args.random_state,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
