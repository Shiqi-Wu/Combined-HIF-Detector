"""
Multi-node distributed training framework for LSTM classifier using HuggingFace Accelerate

Features:
- HuggingFace Accelerate for simplified distributed training
- Multi-node and multi-GPU support
- Automatic mixed precision training
- Gradient accumulation
- Fault tolerance and resume capability
- Comprehensive logging and monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import time
import json
import argparse
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
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
from models.fault_lstm_classifier import LSTMClassifier
from utils.dataloader import load_full_dataset

# Handle wandb import gracefully
WANDB_AVAILABLE = False
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not installed. Logging will be done to local files only.")
    wandb = None


class AcceleratedLSTMTrainer:
    """
    Accelerated LSTM trainer using HuggingFace Accelerate framework
    Supports multi-node, multi-GPU training with automatic mixed precision
    """
    
    def __init__(self, model_config, train_config):
        """
        Initialize the accelerated trainer
        
        Args:
            model_config: Model configuration dictionary
            train_config: Training configuration dictionary
        """
        if not ACCELERATE_AVAILABLE:
            raise ImportError("accelerate is required for distributed training. Install with 'pip install accelerate'")
        
        self.model_config = model_config
        self.train_config = train_config
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 1),
            mixed_precision=train_config.get('mixed_precision', 'fp16'),
            log_with=["wandb"] if train_config.get('use_wandb', False) and WANDB_AVAILABLE else None,
            project_dir=train_config.get('results_dir', './results'),
        )
        
        # Set random seed for reproducibility
        if train_config.get('seed'):
            set_seed(train_config['seed'])
        
        # Initialize wandb if on main process
        if self.accelerator.is_main_process and train_config.get('use_wandb', False) and WANDB_AVAILABLE:
            self.accelerator.init_trackers(
                project_name=train_config.get('wandb_project', 'accelerated-lstm-training'),
                config={**model_config, **train_config},
                init_kwargs={"wandb": {"name": f"accelerated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}}
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
        
        # Create model
        self.model = LSTMClassifier(**model_config)
        
        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config.get('weight_decay', 1e-4)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Performance metrics
        self.performance_metrics = {
            'throughput': [],
            'memory_usage': [],
            'training_time': []
        }
    
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
    
    def prepare_data_loaders(self, train_dataset, val_dataset):
        """Prepare data loaders for distributed training"""
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
        total_correct = 0
        total_samples = 0
        start_time = time.time()
        
        # Create progress bar only on main process
        if self.accelerator.is_main_process:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        else:
            progress_bar = train_loader
        
        for batch_idx, batch in enumerate(progress_bar):
            x_batch, u_batch, p_batch = batch
            
            # Convert one-hot to class indices if needed
            if p_batch.dim() > 1 and p_batch.size(1) > 1:
                p_indices = torch.argmax(p_batch, dim=1)
            else:
                p_indices = p_batch.long()
            
            with self.accelerator.accumulate(self.model):
                # Forward pass
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, p_indices)
                
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
                self.optimizer.zero_grad()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += p_indices.size(0)
            total_correct += (predicted == p_indices).sum().item()
            
            # Update progress bar
            if self.accelerator.is_main_process and hasattr(progress_bar, 'set_postfix'):
                current_acc = total_correct / total_samples
                progress_bar.set_postfix({
                    'Loss': f'{total_loss/(batch_idx+1):.4f}',
                    'Acc': f'{current_acc:.4f}',
                    'LR': f'{self.scheduler.get_last_lr()[0]:.6f}'
                })
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples
        
        # Calculate throughput
        epoch_time = time.time() - start_time
        samples_per_sec = total_samples / epoch_time
        self.performance_metrics['throughput'].append(samples_per_sec)
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x_batch, u_batch, p_batch = batch
                
                # Convert one-hot to class indices if needed
                if p_batch.dim() > 1 and p_batch.size(1) > 1:
                    p_indices = torch.argmax(p_batch, dim=1)
                else:
                    p_indices = p_batch.long()
                
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, p_indices)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += p_indices.size(0)
                total_correct += (predicted == p_indices).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save checkpoint (only on main process)"""
        if not self.accelerator.is_main_process:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_accuracy': val_acc,
            'train_history': self.train_history,
            'model_config': self.model_config,
            'train_config': self.train_config,
            'accelerator_state': self.accelerator.get_state_dict()
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
            self.accelerator.print(f"New best model saved with validation accuracy: {val_acc:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint for resuming training"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Load model and optimizer states
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if 'accelerator_state' in checkpoint:
                self.accelerator.set_state_dict(checkpoint['accelerator_state'])
            
            self.train_history = checkpoint.get('train_history', self.train_history)
            
            if self.accelerator.is_main_process:
                self.accelerator.print(f"Checkpoint loaded from {checkpoint_path}")
                self.accelerator.print(f"Resuming from epoch {checkpoint['epoch']} with val_acc {checkpoint['val_accuracy']:.4f}")
            
            return checkpoint['epoch'], checkpoint['val_accuracy']
        
        return 0, 0.0
    
    def train(self, train_dataset, val_dataset):
        """Main training loop"""
        # Prepare data loaders
        train_loader, val_loader = self.prepare_data_loaders(train_dataset, val_dataset)
        
        # Calculate total training steps
        num_epochs = self.train_config['num_epochs']
        gradient_accumulation_steps = self.train_config.get('gradient_accumulation_steps', 1)
        num_training_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        
        # Create scheduler
        self.scheduler = self.create_scheduler(num_training_steps)
        
        # Prepare everything with accelerator
        self.model, self.optimizer, train_loader, val_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, train_loader, val_loader, self.scheduler
        )
        
        # Try to resume from checkpoint
        checkpoint_path = os.path.join(self.train_config['results_dir'], 'latest_checkpoint.pth')
        start_epoch, best_val_acc = self.load_checkpoint(checkpoint_path)
        
        # Training parameters
        patience_counter = 0
        patience = self.train_config.get('patience', 20)
        
        if self.accelerator.is_main_process:
            self.accelerator.print(f"Starting training on {self.accelerator.num_processes} processes")
            self.accelerator.print(f"Device: {self.accelerator.device}")
            self.accelerator.print(f"Mixed precision: {self.accelerator.mixed_precision}")
            self.accelerator.print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
            # Create results directory
            os.makedirs(self.train_config['results_dir'], exist_ok=True)
        
        # Wait for all processes
        self.accelerator.wait_for_everyone()
        
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Record metrics
            current_lr = self.scheduler.get_last_lr()[0]
            epoch_time = time.time() - start_time
            
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['learning_rates'].append(current_lr)
            self.performance_metrics['training_time'].append(epoch_time)
            
            # Early stopping check
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_acc, is_best)
            
            # Logging (main process only)
            if self.accelerator.is_main_process:
                self.accelerator.print(f"Epoch [{epoch+1}/{num_epochs}]:")
                self.accelerator.print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                self.accelerator.print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                self.accelerator.print(f"  Learning Rate: {current_lr:.6f}")
                self.accelerator.print(f"  Epoch Time: {epoch_time:.2f}s")
                if self.performance_metrics['throughput']:
                    self.accelerator.print(f"  Throughput: {self.performance_metrics['throughput'][-1]:.2f} samples/sec")
                self.accelerator.print("-" * 60)
                
                # Log to wandb
                if self.use_wandb:
                    log_dict = {
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'train_accuracy': train_acc,
                        'val_loss': val_loss,
                        'val_accuracy': val_acc,
                        'learning_rate': current_lr,
                        'epoch_time': epoch_time,
                        'best_val_accuracy': best_val_acc
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
            self.accelerator.print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
            
            # Save final training history
            history_path = os.path.join(self.train_config['results_dir'], 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(self.train_history, f, indent=2)
            
            # Save performance metrics
            metrics_path = os.path.join(self.train_config['results_dir'], 'performance_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
            
            if self.use_wandb:
                self.accelerator.end_training()
        
        return best_val_acc


def main():
    """Main training function using Accelerate framework"""
    parser = argparse.ArgumentParser(description='Accelerated LSTM Training')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--window_size', type=int, default=30, help='Window size')
    parser.add_argument('--sample_step', type=int, default=1, help='Sample step')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--bidirectional', action='store_true', default=True, help='Bidirectional LSTM')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size per device')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--mixed_precision', type=str, default='fp16', choices=['no', 'fp16', 'bf16'], help='Mixed precision training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Logging parameters
    parser.add_argument('--results_dir', type=str, default='./results_accelerated', help='Results directory')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb logging')
    parser.add_argument('--wandb_project', type=str, default='accelerated-lstm-training', help='Wandb project')
    
    args = parser.parse_args()
    
    # Configuration dictionaries
    model_config = {
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'num_classes': 6,  # Classes 2-7 (6 classes total)
        'dropout': args.dropout,
        'bidirectional': args.bidirectional
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
        'seed': args.seed,
        'results_dir': args.results_dir,
        'use_wandb': args.use_wandb,
        'wandb_project': args.wandb_project
    }
    
    data_config = {
        'data_dir': args.data_dir,
        'window_size': args.window_size,
        'sample_step': args.sample_step
    }
    
    # Load dataset
    print("Loading dataset...")
    dataset, file_labels = load_full_dataset(data_config['data_dir'], data_config)
    
    # Split into train and validation sets
    train_indices, val_indices = train_test_split(
        range(len(dataset)), 
        test_size=0.2, 
        random_state=args.seed,
        stratify=[file_labels[i] for i in range(len(dataset))]
    )
    
    # Create subset datasets
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Get input size from a sample
    sample_batch = dataset[0]
    input_size = sample_batch[0].shape[1]  # (seq_len, input_size)
    model_config['input_size'] = input_size
    
    print(f"Input size: {input_size}")
    print(f"Sequence length: {sample_batch[0].shape[0]}")
    
    # Create trainer
    trainer = AcceleratedLSTMTrainer(model_config, train_config)
    
    # Start training
    print("Starting accelerated training...")
    best_val_acc = trainer.train(train_dataset, val_dataset)
    
    print(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
