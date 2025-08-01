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
import utils.dataloader as dataloader_module

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


def load_train_val_data(data_dir: str, train_data_config: Dict, val_data_config: Dict, preprocessing_params_path: str = None):
    print(f"Loading data from: {data_dir}")
    
    # Load full dataset
    train_dataset, _ = dataloader_module.load_dataset_from_folder(data_dir, train_data_config)
    _, val_dataset = dataloader_module.load_dataset_from_folder(data_dir, val_data_config)
        
    # Scale datasets
    scaled_train = dataloader_module.ScaledDataset(train_dataset, pca_dim=train_data_config.get("pca_dim", 2), fit_scalers=True)
    scaled_val = dataloader_module.ScaledDataset(val_dataset, pca_dim=val_data_config.get("pca_dim", 2))

    scaled_val.set_preprocessing_params(scaled_train.get_preprocessing_params())

    if preprocessing_params_path is not None:
        # Load preprocessing parameters if provided
        preprocessing_params = load_preprocessing_params(preprocessing_params_path)
        scaled_train.set_preprocessing_params(preprocessing_params)
        scaled_val.set_preprocessing_params(preprocessing_params)


    # Create dynamic system datasets
    train_dynamic_dataset = DynamicSystemDataset(scaled_train)
    val_dynamic_dataset = DynamicSystemDataset(scaled_val)

    return train_dynamic_dataset, val_dynamic_dataset


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


def evaluate(config_path: str = 'config.json'):
    """Evaluate known control classifier and save top-k accuracy results"""

    # === Load config ===
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    data_dir = config['data_dir']
    # preprocessing_params_path = config['preprocessing_params']
    save_path = config.get('save_path', 'evaluation_results.json')
    
    
    # Data config
    train_config = config['train_data_config']
    val_config = config['val_data_config']
    k_values = config.get('k_values', [1, 2, 3])

    # === Load and preprocess data ===
    train_dataset, val_dataset = load_train_val_data(
        data_dir=data_dir,
        train_data_config=train_config,
        val_data_config=val_config,
        # preprocessing_params_path=preprocessing_params_path
    )

    # === Initialize and train classifier ===
    classifier = KnownControlClassifier()
    classifier.fit_dynamic_system(train_dataset, shared_K=True)

    # === Evaluate on train and val ===
    print("\n[Train Evaluation]")
    train_results = classifier.evaluate_dataset(train_dataset)

    print("\n[Validation Evaluation]")
    val_results = classifier.evaluate_dataset(val_dataset)

    # === Compute Top-K Accuracies ===
    train_topk = calculate_top_k_accuracy(train_results['prediction_errors'], train_results['true_labels'], k_values)
    val_topk = calculate_top_k_accuracy(val_results['prediction_errors'], val_results['true_labels'], k_values)

    # === Save results ===
    results = {
        'train': {
            'eval_len': len(train_results['true_labels']),
            'top_k_accuracies': {f'top_{k}': float(v) for k, v in train_topk.items()}
        },
        'val': {
            'eval_len': len(val_results['true_labels']),
            'top_k_accuracies': {f'top_{k}': float(v) for k, v in val_topk.items()}
        }
    }

    # === Save results ===
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation results saved to {save_path}")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Known Control Classifier")
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration file')
    args = parser.parse_args()
    
    evaluate(config_path=args.config)
    print("Evaluation complete.")