import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import sys
import argparse
import pandas as pd
import json
from datetime import datetime

# Handle wandb import gracefully
WANDB_AVAILABLE = False
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not installed. Logging will be done to CSV files only.")
    wandb = None


# Add parent directory to path to import dataloader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.dataloader as dataloader_module

class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for power grid fault detection
    
    Args:
        input_size: Number of input features (state signal dimensions)
        hidden_size: Number of LSTM hidden units
        num_layers: Number of LSTM layers
        num_classes: Number of output classes (fault types)
        dropout: Dropout probability
        bidirectional: Whether to use bidirectional LSTM
    """
    
    def __init__(self, config):
        super(LSTMClassifier, self).__init__()

        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.num_classes = config['num_classes']
        self.bidirectional = config['bidirectional']
        self.dropout = config['dropout']
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # Calculate the size of LSTM output
        lstm_output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size

        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes)

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(self.hidden_size)

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            output: Classification logits of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # Initialize hidden state for bidirectional LSTM
        # For bidirectional: num_layers * 2 (forward + backward)
        # For unidirectional: num_layers * 1
        h0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            batch_size,
            self.hidden_size,
            dtype=x.dtype
        ).to(x.device)
        
        c0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            batch_size,
            self.hidden_size,
            dtype=x.dtype
        ).to(x.device)
        
        # LSTM forward pass
        # For bidirectional LSTM:
        # - Input: x.shape = (batch_size, seq_len, input_size)
        # - Output: lstm_out.shape = (batch_size, seq_len, hidden_size * 2)
        #   where the last dimension contains [forward_output, backward_output]
        # - Hidden states: hn.shape = (num_layers * 2, batch_size, hidden_size)
        #   where layers are ordered as: [forward_layer1, backward_layer1, forward_layer2, backward_layer2, ...]
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Use the last output of the sequence
        # For bidirectional LSTM, this contains both forward and backward information
        # Shape: (batch_size, hidden_size * 2) where:
        # - First hidden_size elements: forward LSTM output at time T
        # - Last hidden_size elements: backward LSTM output at time 1 (processed from T to 1)
        last_output = lstm_out[:, -1, :]  # Shape: (batch_size, lstm_output_size)
        
        # Fully connected layers
        out = F.relu(self.fc1(last_output))
        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

if __name__ == "__main__":
    data_config = {
        'sample_step': 1,
        'window_size': 30,
        'batch_size': 2
    }
    data_dir = "./data"

    # Load datasets
    train_dataset, val_dataset = dataloader_module.load_dataset_from_folder(data_dir, data_config)

    # Wrap with ScaledDataset and fit scalers/PCA on train set
    scaled_train = dataloader_module.ScaledDataset(train_dataset, pca_dim=2, fit_scalers=True)
    scaled_val = dataloader_module.ScaledDataset(val_dataset, pca_dim=2)
    # Load preprocessing params from train to val
    scaled_val.set_preprocessing_params(scaled_train.get_preprocessing_params())

    # Create DataLoaders
    train_loader = DataLoader(scaled_train, batch_size=data_config['batch_size'], shuffle=True)
    val_loader = DataLoader(scaled_val, batch_size=data_config['batch_size'], shuffle=False)

    # Initialize model, loss function, optimizer
    model_config = {
        'input_size': 2,
        'hidden_size': 128,
        'num_layers': 2,
        'num_classes': 6,
        'dropout': 0.2,
        'bidirectional': True
    }

    model = LSTMClassifier(config=model_config)
    print(model)
    # Get a batch of data
    for batch in train_loader:
        x, u, p = batch
        print("State shape:", x.shape)
        print("Inputs shape:", u.shape)
        print("Position shape:", p.shape)
        # Forward pass
        outputs = model(x)
        print("Model output shape:", outputs.shape)
        print("Model outputs:", outputs)
        break  # Only print for the first batch