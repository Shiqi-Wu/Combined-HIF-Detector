"""
LSTM Sequence Generation Model for Control Signal Prediction
This module implements a sequence-to-sequence model with feedback mechanism
where u(t) depends on x(0:t) and u(0:t-1)

Features:
- Encoder-Decoder architecture with attention mechanism
- Teacher forcing during training
- Autoregressive generation during inference
- Distributed training support with HuggingFace Accelerate
- Comprehensive evaluation metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import sys
import time
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm.auto import tqdm
import math

# HuggingFace Accelerate imports
try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    ACCELERATE_AVAILABLE = True
except ImportError:
    print("Warning: accelerate not installed. Please install with 'pip install accelerate'")
    ACCELERATE_AVAILABLE = False

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataloader import load_full_dataset

# Handle wandb import gracefully
WANDB_AVAILABLE = False
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not installed. Logging will be done to local files only.")
    wandb = None


class SequenceDataset(Dataset):
    """Dataset for sequence-to-sequence learning"""
    
    def __init__(self, x_sequences, u_sequences, input_length, target_length):
        """
        Initialize dataset
        
        Args:
            x_sequences: State sequences (N, seq_len, x_dim)
            u_sequences: Control sequences (N, seq_len, u_dim)
            input_length: Length of input sequence
            target_length: Length of target sequence to predict
        """
        self.x_sequences = x_sequences
        self.u_sequences = u_sequences
        self.input_length = input_length
        self.target_length = target_length
        
        # Create valid sequence indices
        self.valid_indices = []
        for i in range(len(x_sequences)):
            seq_len = x_sequences[i].shape[0]
            if seq_len >= input_length + target_length:
                # Generate all possible subsequences from this sequence
                for start_idx in range(seq_len - input_length - target_length + 1):
                    self.valid_indices.append((i, start_idx))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        seq_idx, start_idx = self.valid_indices[idx]
        
        # Extract input sequences
        x_input = self.x_sequences[seq_idx][start_idx:start_idx + self.input_length]
        u_input = self.u_sequences[seq_idx][start_idx:start_idx + self.input_length]
        
        # Extract target sequence
        u_target = self.u_sequences[seq_idx][start_idx + self.input_length:start_idx + self.input_length + self.target_length]
        
        return (
            torch.FloatTensor(x_input),
            torch.FloatTensor(u_input),
            torch.FloatTensor(u_target)
        )


class AttentionMechanism(nn.Module):
    """Attention mechanism for sequence-to-sequence model"""
    
    def __init__(self, hidden_size):
        super(AttentionMechanism, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, decoder_hidden, encoder_outputs):
        """
        Apply attention mechanism
        
        Args:
            decoder_hidden: Current decoder hidden state (batch_size, hidden_size)
            encoder_outputs: All encoder outputs (batch_size, seq_len, hidden_size)
            
        Returns:
            context: Context vector (batch_size, hidden_size)
            attention_weights: Attention weights (batch_size, seq_len)
        """
        seq_len = encoder_outputs.size(1)
        
        # Repeat decoder hidden state for each encoder output
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Concatenate decoder hidden state with encoder outputs
        combined = torch.cat([decoder_hidden, encoder_outputs], dim=2)
        
        # Calculate attention scores
        energy = torch.tanh(self.attention(combined))
        attention_weights = F.softmax(self.v(energy).squeeze(2), dim=1)
        
        # Apply attention weights to encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights


class LSTMSeq2SeqModel(nn.Module):
    """
    LSTM Sequence-to-Sequence model with feedback mechanism
    Architecture: Encoder-Decoder with attention and feedback
    """
    
    def __init__(self, x_dim, u_dim, hidden_size=256, num_layers=2, dropout=0.2):
        """
        Initialize the model
        
        Args:
            x_dim: Dimension of state input x
            u_dim: Dimension of control input/output u
            hidden_size: Hidden size of LSTM layers
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(LSTMSeq2SeqModel, self).__init__()
        
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder: processes x and u sequences
        self.encoder = nn.LSTM(
            input_size=x_dim + u_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Project bidirectional encoder output to decoder hidden size
        self.encoder_projection = nn.Linear(hidden_size * 2, hidden_size)
        
        # Decoder: generates u sequences with feedback
        self.decoder = nn.LSTM(
            input_size=x_dim + u_dim + hidden_size,  # x + u + context
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = AttentionMechanism(hidden_size)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, u_dim)
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x_seq, u_seq, u_target=None, teacher_forcing_ratio=0.5):
        """
        Forward pass
        
        Args:
            x_seq: State sequence (batch_size, input_len, x_dim)
            u_seq: Control sequence (batch_size, input_len, u_dim)
            u_target: Target control sequence (batch_size, target_len, u_dim)
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            outputs: Predicted control sequences (batch_size, target_len, u_dim)
            attention_weights: Attention weights for visualization
        """
        batch_size = x_seq.size(0)
        input_len = x_seq.size(1)
        target_len = u_target.size(1) if u_target is not None else input_len
        
        # Encode input sequences
        encoder_input = torch.cat([x_seq, u_seq], dim=2)
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(encoder_input)
        
        # Project bidirectional encoder outputs
        encoder_outputs = self.encoder_projection(encoder_outputs)
        
        # Initialize decoder hidden state
        decoder_hidden = encoder_hidden[-1:].repeat(self.num_layers, 1, 1)
        decoder_cell = encoder_cell[-1:].repeat(self.num_layers, 1, 1)
        
        # Decoder outputs
        outputs = []
        attention_weights_list = []
        
        # Initialize decoder input with the last control input
        decoder_input_u = u_seq[:, -1:, :]  # (batch_size, 1, u_dim)
        
        # Extend x_seq for decoder (assuming x continues or repeats last value)
        if target_len > 0:
            # For simplicity, repeat the last x value for the target sequence
            last_x = x_seq[:, -1:, :].repeat(1, target_len, 1)
        
        for t in range(target_len):
            # Get current x value
            current_x = last_x[:, t:t+1, :]
            
            # Calculate attention context
            context, attention_weights = self.attention(
                decoder_hidden[-1], encoder_outputs
            )
            attention_weights_list.append(attention_weights)
            
            # Prepare decoder input: [current_x, previous_u, context]
            context_expanded = context.unsqueeze(1)
            decoder_input = torch.cat([current_x, decoder_input_u, context_expanded], dim=2)
            
            # Decoder forward pass
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder(
                decoder_input, (decoder_hidden, decoder_cell)
            )
            
            # Generate u(t)
            u_pred = self.output_projection(decoder_output)
            outputs.append(u_pred)
            
            # Decide whether to use teacher forcing
            if u_target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Use ground truth as next input (teacher forcing)
                decoder_input_u = u_target[:, t:t+1, :]
            else:
                # Use predicted output as next input (autoregressive)
                decoder_input_u = u_pred
        
        outputs = torch.cat(outputs, dim=1)
        attention_weights = torch.stack(attention_weights_list, dim=1)
        
        return outputs, attention_weights
    
    def generate_sequence(self, x_seq, u_seq, target_len, temperature=1.0):
        """
        Generate control sequence autoregressively
        
        Args:
            x_seq: State sequence (batch_size, input_len, x_dim)
            u_seq: Control sequence (batch_size, input_len, u_dim)
            target_len: Length of sequence to generate
            temperature: Temperature for sampling (1.0 = deterministic)
            
        Returns:
            Generated control sequence
        """
        self.eval()
        with torch.no_grad():
            # Create dummy target for forward pass
            batch_size = x_seq.size(0)
            dummy_target = torch.zeros(batch_size, target_len, self.u_dim).to(x_seq.device)
            
            # Generate without teacher forcing
            outputs, attention_weights = self.forward(
                x_seq, u_seq, dummy_target, teacher_forcing_ratio=0.0
            )
            
            # Apply temperature if needed
            if temperature != 1.0:
                outputs = outputs / temperature
            
            return outputs, attention_weights


class AcceleratedSeq2SeqTrainer:
    """
    Accelerated trainer for sequence-to-sequence model
    """
    
    def __init__(self, model_config, train_config, accelerator=None):
        """
        Initialize trainer
        
        Args:
            model_config: Model configuration dictionary
            train_config: Training configuration dictionary
            accelerator: Accelerator instance (optional)
        """
        if not ACCELERATE_AVAILABLE:
            raise ImportError("accelerate is required for distributed training")
        
        self.model_config = model_config
        self.train_config = train_config
        
        # Initialize accelerator
        if accelerator is not None:
            self.accelerator = accelerator
        else:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 1),
                mixed_precision=train_config.get('mixed_precision', 'fp16'),
                log_with=["wandb"] if train_config.get('use_wandb', False) and WANDB_AVAILABLE else None,
                project_dir=train_config.get('results_dir', './results'),
            )
        
        # Set random seed
        if train_config.get('seed'):
            set_seed(train_config['seed'])
        
        # Initialize wandb
        if (self.accelerator.is_main_process and 
            train_config.get('use_wandb', False) and WANDB_AVAILABLE):
            self.accelerator.init_trackers(
                project_name=train_config.get('wandb_project', 'seq2seq-lstm-training'),
                config={**model_config, **train_config},
                init_kwargs={"wandb": {"name": f"seq2seq_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}}
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
        
        # Create model
        self.model = LSTMSeq2SeqModel(**model_config)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config.get('weight_decay', 1e-4)
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        # Performance metrics
        self.performance_metrics = {
            'throughput': [],
            'training_time': []
        }
    
    def prepare_data_loaders(self, train_dataset, val_dataset):
        """Prepare data loaders"""
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        start_time = time.time()
        
        if self.accelerator.is_main_process:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        else:
            progress_bar = train_loader
        
        for batch_idx, (x_seq, u_input, u_target) in enumerate(progress_bar):
            with self.accelerator.accumulate(self.model):
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass with teacher forcing
                outputs, attention_weights = self.model(
                    x_seq, u_input, u_target, 
                    teacher_forcing_ratio=self.train_config.get('teacher_forcing_ratio', 0.5)
                )
                
                # Calculate loss
                loss = self.criterion(outputs, u_target)
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if self.train_config.get('grad_clip'):
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.train_config['grad_clip']
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            if self.accelerator.is_main_process and hasattr(progress_bar, 'set_postfix'):
                current_lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'Loss': f'{total_loss/num_batches:.4f}',
                    'LR': f'{current_lr:.6f}'
                })
        
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - start_time
        
        # Calculate throughput
        total_samples = len(train_loader.dataset)
        samples_per_sec = total_samples / epoch_time
        self.performance_metrics['throughput'].append(samples_per_sec)
        self.performance_metrics['training_time'].append(epoch_time)
        
        return avg_loss
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for x_seq, u_input, u_target in val_loader:
                # Forward pass without teacher forcing
                outputs, _ = self.model(
                    x_seq, u_input, u_target, teacher_forcing_ratio=0.0
                )
                
                loss = self.criterion(outputs, u_target)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def create_scheduler(self, num_training_steps):
        """Create learning rate scheduler"""
        from torch.optim.lr_scheduler import OneCycleLR
        
        return OneCycleLR(
            self.optimizer,
            max_lr=self.train_config['learning_rate'],
            total_steps=num_training_steps,
            pct_start=0.3,
            anneal_strategy='cos'
        )
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save checkpoint"""
        if not self.accelerator.is_main_process:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'train_history': self.train_history,
            'model_config': self.model_config,
            'train_config': self.train_config
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(
            self.train_config['results_dir'],
            'latest_checkpoint.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(
                self.train_config['results_dir'],
                'best_model.pth'
            )
            torch.save(checkpoint, best_path)
            self.accelerator.print(f"New best model saved with validation loss: {val_loss:.4f}")
    
    def train(self, train_dataset, val_dataset):
        """Main training loop"""
        # Prepare data loaders
        train_loader, val_loader = self.prepare_data_loaders(train_dataset, val_dataset)
        
        # Calculate training steps
        num_epochs = self.train_config['num_epochs']
        gradient_accumulation_steps = self.train_config.get('gradient_accumulation_steps', 1)
        num_training_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        
        # Create scheduler
        self.scheduler = self.create_scheduler(num_training_steps)
        
        # Prepare with accelerator
        self.model, self.optimizer, train_loader, val_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, train_loader, val_loader, self.scheduler
        )
        
        # Training parameters
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.train_config.get('patience', 20)
        
        if self.accelerator.is_main_process:
            self.accelerator.print(f"Starting training on {self.accelerator.num_processes} processes")
            self.accelerator.print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
            # Create results directory
            os.makedirs(self.train_config['results_dir'], exist_ok=True)
        
        # Training loop
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss = self.validate_epoch(val_loader)
            
            # Record metrics
            current_lr = self.scheduler.get_last_lr()[0]
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['learning_rates'].append(current_lr)
            
            # Early stopping
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Logging
            if self.accelerator.is_main_process:
                self.accelerator.print(f"Epoch [{epoch+1}/{num_epochs}]:")
                self.accelerator.print(f"  Train Loss: {train_loss:.4f}")
                self.accelerator.print(f"  Val Loss: {val_loss:.4f}")
                self.accelerator.print(f"  Learning Rate: {current_lr:.6f}")
                if self.performance_metrics['throughput']:
                    self.accelerator.print(f"  Throughput: {self.performance_metrics['throughput'][-1]:.2f} samples/sec")
                self.accelerator.print("-" * 60)
                
                # Log to wandb
                if self.use_wandb:
                    log_dict = {
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'learning_rate': current_lr,
                        'best_val_loss': best_val_loss
                    }
                    if self.performance_metrics['throughput']:
                        log_dict['throughput'] = self.performance_metrics['throughput'][-1]
                    
                    self.accelerator.log(log_dict)
            
            # Early stopping
            if patience_counter >= patience:
                if self.accelerator.is_main_process:
                    self.accelerator.print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Training completion
        if self.accelerator.is_main_process:
            self.accelerator.print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
            
            # Save training history
            history_path = os.path.join(self.train_config['results_dir'], 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(self.train_history, f, indent=2)
            
            if self.use_wandb:
                self.accelerator.end_training()
        
        return best_val_loss
    
    def evaluate_model(self, test_dataset, visualize=True):
        """Evaluate trained model"""
        if not self.accelerator.is_main_process:
            return
        
        # Load best model
        best_model_path = os.path.join(self.train_config['results_dir'], 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location='cpu')
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state_dict'])
        
        # Create test loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=False,
            num_workers=4
        )
        
        self.model.eval()
        total_mse = 0
        total_mae = 0
        num_samples = 0
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for x_seq, u_input, u_target in test_loader:
                x_seq = x_seq.to(self.accelerator.device)
                u_input = u_input.to(self.accelerator.device)
                u_target = u_target.to(self.accelerator.device)
                
                # Generate sequences
                outputs, attention_weights = self.model.generate_sequence(
                    x_seq, u_input, u_target.size(1)
                )
                
                # Calculate metrics
                mse = F.mse_loss(outputs, u_target)
                mae = F.l1_loss(outputs, u_target)
                
                total_mse += mse.item() * x_seq.size(0)
                total_mae += mae.item() * x_seq.size(0)
                num_samples += x_seq.size(0)
                
                # Store for visualization
                predictions.append(outputs.cpu().numpy())
                targets.append(u_target.cpu().numpy())
        
        avg_mse = total_mse / num_samples
        avg_mae = total_mae / num_samples
        
        print(f"Test Results:")
        print(f"  MSE: {avg_mse:.6f}")
        print(f"  MAE: {avg_mae:.6f}")
        print(f"  RMSE: {math.sqrt(avg_mse):.6f}")
        
        if visualize:
            predictions = np.concatenate(predictions, axis=0)
            targets = np.concatenate(targets, axis=0)
            
            # Plot some example sequences
            self.plot_predictions(predictions, targets)
        
        return avg_mse, avg_mae
    
    def plot_predictions(self, predictions, targets, num_examples=5):
        """Plot prediction examples"""
        fig, axes = plt.subplots(num_examples, 1, figsize=(15, 3*num_examples))
        if num_examples == 1:
            axes = [axes]
        
        for i in range(min(num_examples, len(predictions))):
            # Plot first dimension of u
            axes[i].plot(targets[i, :, 0], label='Target', alpha=0.8)
            axes[i].plot(predictions[i, :, 0], label='Predicted', alpha=0.8)
            axes[i].set_title(f'Example {i+1}: Control Signal Prediction')
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Control Value')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.train_config['results_dir'], 'prediction_examples.png'), dpi=300)
        plt.show()


def create_datasets_from_data(dataset, input_length=30, target_length=10, train_ratio=0.8):
    """Create sequence datasets from loaded data"""
    x_sequences = []
    u_sequences = []
    
    for i in range(len(dataset)):
        x_seq, u_seq, _ = dataset[i]
        x_sequences.append(x_seq.numpy())
        u_sequences.append(u_seq.numpy())
    
    # Convert to numpy arrays
    x_sequences = np.array(x_sequences)
    u_sequences = np.array(u_sequences)
    
    # Split into train and validation
    num_train = int(len(x_sequences) * train_ratio)
    
    train_x = x_sequences[:num_train]
    train_u = u_sequences[:num_train]
    val_x = x_sequences[num_train:]
    val_u = u_sequences[num_train:]
    
    # Create datasets
    train_dataset = SequenceDataset(train_x, train_u, input_length, target_length)
    val_dataset = SequenceDataset(val_x, val_u, input_length, target_length)
    
    return train_dataset, val_dataset


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='LSTM Sequence-to-Sequence Training')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--input_length', type=int, default=30, help='Input sequence length')
    parser.add_argument('--target_length', type=int, default=10, help='Target sequence length')
    parser.add_argument('--window_size', type=int, default=50, help='Window size for data loading')
    parser.add_argument('--sample_step', type=int, default=1, help='Sample step for data loading')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation')
    parser.add_argument('--mixed_precision', type=str, default='fp16', help='Mixed precision')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, help='Teacher forcing ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Logging parameters
    parser.add_argument('--results_dir', type=str, default='./results_seq2seq', help='Results directory')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb logging')
    parser.add_argument('--wandb_project', type=str, default='seq2seq-lstm-training', help='Wandb project')
    
    args = parser.parse_args()
    
    # Configuration
    data_config = {
        'window_size': args.window_size,
        'sample_step': args.sample_step,
        'batch_size': args.batch_size
    }
    
    model_config = {
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout
    }
    
    train_config = {
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'grad_clip': args.grad_clip,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'mixed_precision': args.mixed_precision,
        'teacher_forcing_ratio': args.teacher_forcing_ratio,
        'seed': args.seed,
        'results_dir': args.results_dir,
        'use_wandb': args.use_wandb,
        'wandb_project': args.wandb_project
    }
    
    # Load data
    print("Loading dataset...")
    dataset, file_labels = load_full_dataset(args.data_dir, data_config)
    
    # Get dimensions from sample
    sample_x, sample_u, sample_p = dataset[0]
    x_dim = sample_x.shape[1]
    u_dim = sample_u.shape[1]
    
    model_config['x_dim'] = x_dim
    model_config['u_dim'] = u_dim
    
    print(f"Data dimensions: x_dim={x_dim}, u_dim={u_dim}")
    print(f"Total sequences: {len(dataset)}")
    
    # Create sequence datasets
    print("Creating sequence datasets...")
    train_dataset, val_dataset = create_datasets_from_data(
        dataset, args.input_length, args.target_length
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create trainer
    trainer = AcceleratedSeq2SeqTrainer(model_config, train_config)
    
    # Train model
    print("Starting training...")
    best_val_loss = trainer.train(train_dataset, val_dataset)
    
    # Evaluate model
    print("Evaluating model...")
    trainer.evaluate_model(val_dataset)
    
    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
