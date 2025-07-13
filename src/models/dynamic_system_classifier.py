#!/usr/bin/env python3
"""
Seq2Seq Evaluation Script with Dynamic System Modeling

This script implements a hybrid approach for fault classification:
1. Learn linear dynamics: x[t+1] = K*x[t] + B(p)*u[t] + noise
2. Use seq2seq model to generate u from x (since u is unknown in practice)
3. Classify by comparing prediction error: ||x[1:] - (K*x[:-1] + B(p)*u_gen[:-1])||

The approach combines:
- Linear system identification for each fault class
- Seq2seq neural network for control signal generation
- Residual-based fault classification
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.seq2seq_lstm import Seq2SeqLSTM
from utils.dataloader import load_full_dataset, create_kfold_dataloaders, ScaledDataset

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def compute_K_and_B_per_class(x_dataset, u_dataset, et_dataset, shared_K=True):
    """
    Compute system matrix K (shared or per-class) and B matrix per class.

    Args:
        x_dataset: list of np.ndarray, each (T, n_x)
        u_dataset: list of np.ndarray, each (T, n_u)
        et_dataset: list of metadata, where et_data[0] is one-hot class label
        shared_K (bool): If True, estimate one global K; else estimate K_i for each class.

    Returns:
        K_list: list of np.ndarray (n_x, n_x), one per class
        B_list: list of np.ndarray (n_x, n_u), one per class
        B_mse_list: list of float, MSE per class
    """
    num_classes = len(et_dataset[0][0])
    class_data = {i: {'X': [], 'Y': [], 'U': []} for i in range(num_classes)}

    # Organize data by class
    for x, u, et in zip(x_dataset, u_dataset, et_dataset):
        class_idx = int(np.argmax(et[0]))
        class_data[class_idx]['X'].append(x[:-1])
        class_data[class_idx]['Y'].append(x[1:])
        class_data[class_idx]['U'].append(u[:-1])

    if shared_K:
        # Compute global K
        X_all = []
        Y_all = []
        for i in range(num_classes):
            if len(class_data[i]['X']) > 0:  # Check if class has data
                X_all.append(np.vstack(class_data[i]['X']))
                Y_all.append(np.vstack(class_data[i]['Y']))
        
        if len(X_all) > 0:
            X_cat = np.vstack(X_all)
            Y_cat = np.vstack(Y_all)
            K_global, _, _, _ = np.linalg.lstsq(X_cat, Y_cat, rcond=None)
        else:
            # Fallback to identity matrix
            n_x = x_dataset[0].shape[1]
            K_global = np.eye(n_x)
        
        K_list = [K_global.copy() for _ in range(num_classes)]
    else:
        K_list = []

    B_list = []
    B_mse_list = []

    for i in range(num_classes):
        if len(class_data[i]['X']) == 0:
            # No data for this class, use default values
            n_x = x_dataset[0].shape[1] if x_dataset else 2
            n_u = u_dataset[0].shape[1] if u_dataset else 2
            
            if not shared_K:
                K_list.append(np.eye(n_x))
            B_list.append(np.zeros((n_x, n_u)))
            B_mse_list.append(float('inf'))
            continue

        X_i = np.vstack(class_data[i]['X'])
        Y_i = np.vstack(class_data[i]['Y'])
        U_i = np.vstack(class_data[i]['U'])

        if shared_K:
            K_i = K_list[i]
        else:
            try:
                K_i, _, _, _ = np.linalg.lstsq(X_i, Y_i, rcond=None)
                K_list.append(K_i)
            except np.linalg.LinAlgError:
                # Fallback to identity matrix
                K_i = np.eye(X_i.shape[1])
                K_list.append(K_i)

        # Compute residual and estimate B
        delta_Y = Y_i - X_i @ K_i
        try:
            B_T, _, _, _ = np.linalg.lstsq(U_i, delta_Y, rcond=None)
            B = B_T.T
        except np.linalg.LinAlgError:
            # Fallback to zero matrix
            B = np.zeros((X_i.shape[1], U_i.shape[1]))

        # Compute prediction error
        Y_pred = X_i @ K_i + U_i @ B.T
        mse = np.mean((Y_i - Y_pred) ** 2)

        B_list.append(B)
        B_mse_list.append(mse)

    return K_list, B_list, B_mse_list


class DynamicSystemClassifier:
    """
    Hybrid classifier combining linear system identification and seq2seq modeling
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the classifier
        
        Args:
            device: Device to run models on
        """
        self.device = device
        self.K_matrices = None
        self.B_matrices = None
        self.B_mse_list = None
        self.seq2seq_model = None
        self.num_classes = 6
        self.class_names = ['Normal', 'Fault_1', 'Fault_2', 'Fault_3', 'Fault_4', 'Fault_5']
        
    def fit_dynamic_system(self, train_dataset, shared_K=True, verbose=True):
        """
        Fit the linear dynamic system parameters
        
        Args:
            train_dataset: Training dataset
            shared_K: Whether to use shared K matrix across classes
            verbose: Whether to print progress
        """
        if verbose:
            print("Fitting dynamic system parameters...")
        
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
    
    def load_seq2seq_model(self, model_path, model_config):
        """
        Load trained seq2seq model
        
        Args:
            model_path: Path to trained model checkpoint
            model_config: Model configuration dictionary
        """
        print(f"Loading seq2seq model from {model_path}")
        
        # Create model
        self.seq2seq_model = Seq2SeqLSTM(**model_config).to(torch.float64)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        self.seq2seq_model.load_state_dict(checkpoint['model_state_dict'])
        self.seq2seq_model.to(self.device)
        self.seq2seq_model.eval()
        
        print("Seq2seq model loaded successfully")
    
    def generate_control_signals(self, x_sequences):
        """
        Generate control signals using seq2seq model
        
        Args:
            x_sequences: State sequences [batch_size, seq_len, state_dim]
            
        Returns:
            Generated control signals [batch_size, seq_len, control_dim]
        """
        if self.seq2seq_model is None:
            raise ValueError("Seq2seq model not loaded. Call load_seq2seq_model first.")
        
        self.seq2seq_model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x_sequences, dtype=torch.float64, device=self.device)
            u_generated = self.seq2seq_model.predict_control_for_inference(x_tensor)
        
        return u_generated.cpu().numpy()
    
    def predict_single_sequence(self, x_sequence, u_generated=None):
        """
        Predict class for a single sequence using residual-based classification
        
        Args:
            x_sequence: State sequence [seq_len, state_dim]
            u_generated: Generated control sequence [seq_len, control_dim] (optional)
            
        Returns:
            Tuple of (predicted_class, prediction_errors)
        """
        if self.K_matrices is None or self.B_matrices is None:
            raise ValueError("Dynamic system not fitted. Call fit_dynamic_system first.")
        
        if u_generated is None:
            # Generate control signals
            u_generated = self.generate_control_signals(x_sequence[None, ...])[0]
        
        # Compute prediction errors for each class
        prediction_errors = []
        
        x_current = x_sequence[:-1]  # x[0:T-1]
        x_next = x_sequence[1:]      # x[1:T]
        u_current = u_generated[:-1] # u[0:T-1]
        
        for class_idx in range(self.num_classes):
            K = self.K_matrices[class_idx]
            B = self.B_matrices[class_idx]
            
            # Predict next states: x[t+1] = K*x[t] + B*u[t]
            x_pred = x_current @ K.T + u_current @ B.T
            
            # Compute prediction error
            error = np.mean((x_next - x_pred) ** 2)
            prediction_errors.append(error)
        
        # Class with minimum error
        predicted_class = np.argmin(prediction_errors)
        
        return predicted_class, prediction_errors
    
    def predict_batch(self, x_sequences, u_sequences=None, verbose=True):
        """
        Predict classes for a batch of sequences
        
        Args:
            x_sequences: State sequences [batch_size, seq_len, state_dim]
            u_sequences: Control sequences [batch_size, seq_len, control_dim] (optional)
            verbose: Whether to show progress
            
        Returns:
            Tuple of (predictions, all_errors)
        """
        if isinstance(x_sequences, torch.Tensor):
            x_sequences = x_sequences.numpy()
        
        if u_sequences is None:
            if verbose:
                print("Generating control signals...")
            u_sequences = self.generate_control_signals(x_sequences)
        elif isinstance(u_sequences, torch.Tensor):
            u_sequences = u_sequences.numpy()
        
        predictions = []
        all_errors = []
        
        iterator = tqdm(range(len(x_sequences)), desc="Predicting") if verbose else range(len(x_sequences))
        
        for i in iterator:
            pred_class, errors = self.predict_single_sequence(x_sequences[i], u_sequences[i])
            predictions.append(pred_class)
            all_errors.append(errors)
        
        return np.array(predictions), np.array(all_errors)
    
    def evaluate_dataset(self, dataset, batch_size=32, verbose=True):
        """
        Evaluate the classifier on a dataset
        
        Args:
            dataset: Dataset to evaluate
            batch_size: Batch size for processing
            verbose: Whether to show progress
            
        Returns:
            Dictionary with evaluation results
        """
        if verbose:
            print(f"Evaluating on dataset with {len(dataset)} samples...")
        
        # Extract all data
        all_x = []
        all_u = []
        all_labels = []
        
        for i in range(len(dataset)):
            x_batch, u_batch, p_batch = dataset[i]
            all_x.append(x_batch.numpy())
            all_u.append(u_batch.numpy())
            
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
        predictions, prediction_errors = self.predict_batch(all_x, all_u, verbose=verbose)
        
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

