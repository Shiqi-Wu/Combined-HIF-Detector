"""
Distributed trainer for Seq2Seq LSTM using HuggingFace Accelerate
Supports K-fold validation with scaled datasets and multi-GPU training

Features:
- HuggingFace Accelerate for distributed training
- K-fold cross validation with shared preprocessing
- ScaledDataset support
- Teacher forcing scheduling
- Comprehensive metrics tracking
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
from models.seq2seq_lstm import Seq2SeqLSTM, Seq2SeqLoss
from utils.dataloader import load_full_dataset, create_kfold_dataloaders, create_preprocessed_kfold_dataloaders, ScaledDataset

# Handle wandb import gracefully
WANDB_AVAILABLE = False
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not installed. Logging will be done to local files only.")
    wandb = None


class AcceleratedSeq2SeqTrainer:
    """
    Accelerated Seq2Seq trainer using HuggingFace Accelerate framework
    Supports multi-GPU training with K-fold validation
    """
    
    def __init__(self, model_config, train_config, accelerator=None, fold_idx=None):
        """
        Initialize the accelerated seq2seq trainer
        
        Args:
            model_config: Model configuration dictionary
            train_config: Training configuration dictionary
            accelerator: Pre-initialized accelerator instance (optional)
            fold_idx: Current fold index for tracking (optional)
        """
        if not ACCELERATE_AVAILABLE:
            raise ImportError("accelerate is required for distributed training. Install with 'pip install accelerate'")
        
        self.model_config = model_config
        self.train_config = train_config
        self.fold_idx = fold_idx
        
        # Initialize accelerator or use provided one
        if accelerator is not None:
            self.accelerator = accelerator
            self.shared_accelerator = True
        else:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 1),
                mixed_precision='no',
                log_with=["wandb"] if train_config.get('use_wandb', False) and WANDB_AVAILABLE else None,
                project_dir=train_config.get('results_dir', './results'),
            )
            self.shared_accelerator = False
        
        # Set random seed for reproducibility
        if train_config.get('seed'):
            fold_seed = train_config['seed'] + (fold_idx if fold_idx is not None else 0)
            set_seed(fold_seed)
        
        # Initialize wandb only once per accelerator
        if (self.accelerator.is_main_process and 
            train_config.get('use_wandb', False) and 
            WANDB_AVAILABLE and 
            not self.shared_accelerator):
            self.accelerator.init_trackers(
                project_name=train_config.get('wandb_project', 'seq2seq-lstm-training'),
                config={**model_config, **train_config},
                init_kwargs={"wandb": {"name": f"seq2seq_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}}
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
        
        # Create model (fresh for each fold)
        self.model = Seq2SeqLSTM(**model_config)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config.get('weight_decay', 1e-4)
        )
        
        # Setup loss function
        self.criterion = Seq2SeqLoss(
            mse_weight=train_config.get('mse_weight', 1.0),
            l1_weight=train_config.get('l1_weight', 0.0),
            smoothness_weight=train_config.get('smoothness_weight', 0.0)
        )
        
        # Training history
        self.train_history = {
            'train_loss': [],
            'train_mse': [],
            'train_mae': [],
            'val_loss': [],
            'val_mse': [],
            'val_mae': [],
            'learning_rates': [],
            'teacher_forcing_ratios': []
        }
        
        # Performance metrics
        self.performance_metrics = {
            'throughput': [],
            'memory_usage': [],
            'training_time': []
        }
        
        # Teacher forcing scheduling
        self.initial_teacher_forcing_ratio = model_config.get('teacher_forcing_ratio', 0.5)
        self.teacher_forcing_decay = train_config.get('teacher_forcing_decay', 0.95)
        self.min_teacher_forcing_ratio = train_config.get('min_teacher_forcing_ratio', 0.1)
    
    def reset_for_new_fold(self, fold_idx):
        """Reset trainer state for a new fold while keeping the same accelerator"""
        self.fold_idx = fold_idx
        
        # Reset random seed with fold-specific seed
        if self.train_config.get('seed'):
            fold_seed = self.train_config['seed'] + fold_idx
            set_seed(fold_seed)
        
        # Create new model for this fold
        self.model = Seq2SeqLSTM(**self.model_config)
        
        # Create new optimizer for this fold
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.train_config['learning_rate'],
            weight_decay=self.train_config.get('weight_decay', 1e-4)
        )
        
        # Reset training history
        self.train_history = {
            'train_loss': [],
            'train_mse': [],
            'train_mae': [],
            'val_loss': [],
            'val_mse': [],
            'val_mae': [],
            'learning_rates': [],
            'teacher_forcing_ratios': []
        }
        
        # Reset performance metrics
        self.performance_metrics = {
            'throughput': [],
            'memory_usage': [],
            'training_time': []
        }
    
    def create_scheduler(self, optimizer):
        """Create learning rate scheduler"""
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=10, 
            factor=0.5,
            verbose=self.accelerator.is_main_process
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
    
    def update_teacher_forcing_ratio(self, epoch):
        """Update teacher forcing ratio based on training progress"""
        decay_factor = self.teacher_forcing_decay ** epoch
        new_ratio = max(
            self.initial_teacher_forcing_ratio * decay_factor,
            self.min_teacher_forcing_ratio
        )
        self.model.teacher_forcing_ratio = new_ratio
        return new_ratio
    
    def calculate_metrics(self, predictions, targets):
        """Calculate regression metrics"""
        predictions_np = predictions.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        mse = mean_squared_error(targets_np.flatten(), predictions_np.flatten())
        mae = mean_absolute_error(targets_np.flatten(), predictions_np.flatten())
        
        return mse, mae
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_mae = 0
        total_samples = 0
        start_time = time.time()
        
        # Update teacher forcing ratio
        tf_ratio = self.update_teacher_forcing_ratio(epoch)
        
        # Create progress bar only on main process
        if self.accelerator.is_main_process:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        else:
            progress_bar = train_loader
        
        for batch_idx, batch in enumerate(progress_bar):
            # Unpack batch: x_batch (state), u_batch (control), p_batch (class, not used)
            x_batch, u_batch, p_batch = batch
            
            # Ensure correct data types
            x_batch = x_batch.float()
            u_batch = u_batch.float()
            
            with self.accelerator.accumulate(self.model):
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass with teacher forcing
                predictions = self.model(x_batch, u_batch)
                
                # Calculate loss
                loss = self.criterion(predictions, u_batch)
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if self.train_config.get('grad_clip') and self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), 
                        self.train_config['grad_clip']
                    )
                
                # Optimizer step
                self.optimizer.step()
            
            # Calculate metrics
            mse, mae = self.calculate_metrics(predictions, u_batch)
            
            # Statistics
            total_loss += loss.item()
            total_mse += mse
            total_mae += mae
            total_samples += x_batch.size(0)
            
            # Update progress bar
            if self.accelerator.is_main_process and hasattr(progress_bar, 'set_postfix'):
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'Loss': f'{total_loss/(batch_idx+1):.4f}',
                    'MSE': f'{total_mse/(batch_idx+1):.4f}',
                    'MAE': f'{total_mae/(batch_idx+1):.4f}',
                    'TF': f'{tf_ratio:.3f}',
                    'LR': f'{current_lr:.6f}'
                })
        
        # Calculate average metrics
        avg_loss = total_loss / len(train_loader)
        avg_mse = total_mse / len(train_loader)
        avg_mae = total_mae / len(train_loader)
        
        # Calculate throughput
        epoch_time = time.time() - start_time
        samples_per_sec = total_samples / epoch_time
        self.performance_metrics['throughput'].append(samples_per_sec)
        
        return avg_loss, avg_mse, avg_mae, tf_ratio
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_mae = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x_batch, u_batch, p_batch = batch
                
                # Ensure correct data types
                x_batch = x_batch.float()
                u_batch = u_batch.float()
                
                # Forward pass without teacher forcing (inference mode)
                predictions = self.model.generate_sequence(x_batch, max_length=u_batch.size(1))
                
                # Calculate loss
                loss = self.criterion(predictions, u_batch)
                
                # Calculate metrics
                mse, mae = self.calculate_metrics(predictions, u_batch)
                
                total_loss += loss.item()
                total_mse += mse
                total_mae += mae
        
        avg_loss = total_loss / len(val_loader)
        avg_mse = total_mse / len(val_loader)
        avg_mae = total_mae / len(val_loader)
        
        return avg_loss, avg_mse, avg_mae
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save checkpoint (only on main process)"""
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
            'train_config': self.train_config,
            'fold_idx': self.fold_idx
        }
        
        # Create fold-specific checkpoint filename
        fold_suffix = f"_fold_{self.fold_idx + 1}" if self.fold_idx is not None else ""
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(
            self.train_config['results_dir'], 
            f'latest_checkpoint{fold_suffix}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(
                self.train_config['results_dir'], 
                f'best_model{fold_suffix}.pth'
            )
            torch.save(checkpoint, best_path)
            fold_info = f"Fold {self.fold_idx + 1} - " if self.fold_idx is not None else ""
            self.accelerator.print(f"{fold_info}New best model saved with validation loss: {val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint for resuming training"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Load model and optimizer states
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.train_history = checkpoint.get('train_history', self.train_history)
            
            if self.accelerator.is_main_process:
                fold_info = f"Fold {checkpoint.get('fold_idx', 0) + 1} - " if checkpoint.get('fold_idx') is not None else ""
                self.accelerator.print(f"{fold_info}Checkpoint loaded from {checkpoint_path}")
                self.accelerator.print(f"{fold_info}Resuming from epoch {checkpoint['epoch']} with val_loss {checkpoint['val_loss']:.4f}")
            
            return checkpoint['epoch'], checkpoint['val_loss']
        
        return 0, float('inf')
    
    def train(self, train_dataset, val_dataset):
        """Main training loop"""
        # Prepare data loaders
        train_loader, val_loader = self.prepare_data_loaders(train_dataset, val_dataset)
        
        # Create scheduler
        self.scheduler = self.create_scheduler(self.optimizer)
        
        # Prepare everything with accelerator
        self.model, self.optimizer, train_loader, val_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, train_loader, val_loader, self.scheduler
        )
        
        # Try to resume from checkpoint
        fold_suffix = f"_fold_{self.fold_idx + 1}" if self.fold_idx is not None else ""
        checkpoint_path = os.path.join(self.train_config['results_dir'], f'latest_checkpoint{fold_suffix}.pth')
        start_epoch, best_val_loss = self.load_checkpoint(checkpoint_path)
        
        # Training parameters
        patience_counter = 0
        patience = self.train_config.get('patience', 20)
        num_epochs = self.train_config['num_epochs']
        
        if self.accelerator.is_main_process:
            fold_info = f"Fold {self.fold_idx + 1}" if self.fold_idx is not None else "Training"
            self.accelerator.print(f"Starting {fold_info} on {self.accelerator.num_processes} processes")
            self.accelerator.print(f"Device: {self.accelerator.device}")
            self.accelerator.print(f"Mixed precision: {self.accelerator.mixed_precision}")
            
            # Model complexity analysis
            from models.seq2seq_lstm import analyze_model_complexity
            stats = analyze_model_complexity(self.accelerator.unwrap_model(self.model))
            self.accelerator.print(f"Model parameters: {stats['total_parameters']:,}")
            
            # Create results directory
            os.makedirs(self.train_config['results_dir'], exist_ok=True)
        
        # Wait for all processes
        self.accelerator.wait_for_everyone()
        
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            start_time = time.time()
            
            # Training
            train_loss, train_mse, train_mae, tf_ratio = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_mse, val_mae = self.validate_epoch(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Record metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - start_time
            
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_mse'].append(train_mse)
            self.train_history['train_mae'].append(train_mae)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_mse'].append(val_mse)
            self.train_history['val_mae'].append(val_mae)
            self.train_history['learning_rates'].append(current_lr)
            self.train_history['teacher_forcing_ratios'].append(tf_ratio)
            self.performance_metrics['training_time'].append(epoch_time)
            
            # Early stopping check
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Logging (main process only)
            if self.accelerator.is_main_process:
                fold_info = f"Fold {self.fold_idx + 1} - " if self.fold_idx is not None else ""
                self.accelerator.print(f"{fold_info}Epoch [{epoch+1}/{num_epochs}]:")
                self.accelerator.print(f"  Train Loss: {train_loss:.4f}, MSE: {train_mse:.4f}, MAE: {train_mae:.4f}")
                self.accelerator.print(f"  Val Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, MAE: {val_mae:.4f}")
                self.accelerator.print(f"  Learning Rate: {current_lr:.6f}")
                self.accelerator.print(f"  Teacher Forcing: {tf_ratio:.3f}")
                self.accelerator.print(f"  Epoch Time: {epoch_time:.2f}s")
                if self.performance_metrics['throughput']:
                    self.accelerator.print(f"  Throughput: {self.performance_metrics['throughput'][-1]:.2f} samples/sec")
                self.accelerator.print("-" * 70)
                
                # Log to wandb
                if self.use_wandb:
                    log_dict = {
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'train_mse': train_mse,
                        'train_mae': train_mae,
                        'val_loss': val_loss,
                        'val_mse': val_mse,
                        'val_mae': val_mae,
                        'learning_rate': current_lr,
                        'teacher_forcing_ratio': tf_ratio,
                        'epoch_time': epoch_time,
                        'best_val_loss': best_val_loss
                    }
                    if self.fold_idx is not None:
                        log_dict['fold'] = self.fold_idx + 1
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
            fold_info = f"Fold {self.fold_idx + 1} " if self.fold_idx is not None else ""
            self.accelerator.print(f"{fold_info}training completed. Best validation loss: {best_val_loss:.4f}")
            
            # Save final training history
            history_filename = f'training_history_fold_{self.fold_idx + 1}.json' if self.fold_idx is not None else 'training_history.json'
            history_path = os.path.join(self.train_config['results_dir'], history_filename)
            with open(history_path, 'w') as f:
                json.dump(self.train_history, f, indent=2)
            
            # Save performance metrics
            metrics_filename = f'performance_metrics_fold_{self.fold_idx + 1}.json' if self.fold_idx is not None else 'performance_metrics.json'
            metrics_path = os.path.join(self.train_config['results_dir'], metrics_filename)
            with open(metrics_path, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
        
        return best_val_loss


def main():
    """Main training function for K-fold Seq2Seq training"""
    parser = argparse.ArgumentParser(description='K-Fold Seq2Seq LSTM Training')

    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--window_size', type=int, default=30, help='Window size')
    parser.add_argument('--sample_step', type=int, default=1, help='Sampling step')
    parser.add_argument('--pca_dim', type=int, default=2, help='PCA dimension')

    # Model parameters
    parser.add_argument('--state_dim', type=int, default=68, help='State dimension')
    parser.add_argument('--control_dim', type=int, default=10, help='Control dimension')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--bidirectional', action='store_true', default=True, help='Use bidirectional LSTM')
    parser.add_argument('--use_attention', action='store_true', help='Use attention mechanism')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, help='Initial teacher forcing ratio')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Loss function parameters
    parser.add_argument('--mse_weight', type=float, default=1.0, help='MSE loss weight')
    parser.add_argument('--l1_weight', type=float, default=0.0, help='L1 loss weight')
    parser.add_argument('--smoothness_weight', type=float, default=0.0, help='Smoothness loss weight')

    # Teacher forcing scheduling
    parser.add_argument('--teacher_forcing_decay', type=float, default=0.95, help='Teacher forcing decay rate')
    parser.add_argument('--min_teacher_forcing_ratio', type=float, default=0.1, help='Minimum teacher forcing ratio')

    # Logging
    parser.add_argument('--results_dir', type=str, default='./results_seq2seq_kfold', help='Root directory to store fold results')
    parser.add_argument('--use_wandb', action='store_true', help='Use WandB logging')
    parser.add_argument('--wandb_project', type=str, default='seq2seq-kfold-training', help='WandB project name')

    # K-Fold
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds')

    args = parser.parse_args()

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision='no',
        log_with=["wandb"] if args.use_wandb and WANDB_AVAILABLE else None,
        project_dir=args.results_dir,
    )
    
    # Initialize wandb once for the entire K-fold experiment
    if accelerator.is_main_process and args.use_wandb and WANDB_AVAILABLE:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": {"name": f"seq2seq_kfold_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}}
        )
    
    # Step 1: Load full dataset
    data_config = {
        'window_size': args.window_size,
        'sample_step': args.sample_step,
        'batch_size': args.batch_size
    }

    print("\n===== Loading Dataset =====")
    dataset, file_labels = load_full_dataset(args.data_dir, data_config)

    # Step 2: Create K-Fold Dataloaders with shared preprocessing
    print("\n===== Creating K-Fold Splits with Shared Preprocessing =====")
    folds, preprocessing_params = create_preprocessed_kfold_dataloaders(
        dataset,
        file_labels,
        config=data_config,
        n_splits=args.k_folds,
        random_state=args.seed,
        pca_dim=args.pca_dim
    )

    # Step 3: Train across folds with shared accelerator
    fold_losses = []
    trainer = None  # Initialize trainer once
    
    for fold_idx, (train_loader, val_loader, test_loader) in enumerate(folds):
        print(f"\n========================\n Fold {fold_idx + 1}/{args.k_folds}\n========================")

        fold_result_dir = os.path.join(args.results_dir, f"fold_{fold_idx + 1}")
        os.makedirs(fold_result_dir, exist_ok=True)

        # Get dimensions from data
        sample_batch = train_loader.dataset[0]
        state_dim = sample_batch[0].shape[1]  # x_batch dimension
        control_dim = sample_batch[1].shape[1]  # u_batch dimension
        
        model_config = {
            'state_dim': state_dim,
            'control_dim': control_dim,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'bidirectional': args.bidirectional,
            'use_attention': args.use_attention,
            'teacher_forcing_ratio': args.teacher_forcing_ratio
        }
        
        train_config = {
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'patience': args.patience,
            'grad_clip': args.grad_clip,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'seed': args.seed,
            'results_dir': fold_result_dir,
            'use_wandb': args.use_wandb,
            'wandb_project': args.wandb_project,
            'mse_weight': args.mse_weight,
            'l1_weight': args.l1_weight,
            'smoothness_weight': args.smoothness_weight,
            'teacher_forcing_decay': args.teacher_forcing_decay,
            'min_teacher_forcing_ratio': args.min_teacher_forcing_ratio
        }

        if trainer is None:
            # Create trainer for first fold
            trainer = AcceleratedSeq2SeqTrainer(model_config, train_config, accelerator=accelerator, fold_idx=fold_idx)
        else:
            # Reset trainer for subsequent folds
            trainer.reset_for_new_fold(fold_idx)
            trainer.train_config['results_dir'] = fold_result_dir

        print(f"Model config: {model_config}")
        print(f"State dim: {state_dim}, Control dim: {control_dim}")
        
        best_val_loss = trainer.train(train_loader.dataset, val_loader.dataset)
        fold_losses.append(best_val_loss)
        
        # Log fold completion to wandb
        if accelerator.is_main_process and args.use_wandb and WANDB_AVAILABLE:
            accelerator.log({
                f'fold_{fold_idx + 1}_final_loss': best_val_loss,
                'completed_folds': fold_idx + 1
            })

    # Step 4: Summary
    print("\n===== K-Fold Summary =====")
    for i, loss in enumerate(fold_losses):
        print(f"Fold {i + 1}: {loss:.4f}")
    avg_loss = np.mean(fold_losses)
    std_loss = np.std(fold_losses)
    print(f"Average Loss: {avg_loss:.4f} ± {std_loss:.4f}")
    
    # Final logging to wandb
    if accelerator.is_main_process and args.use_wandb and WANDB_AVAILABLE:
        accelerator.log({
            'final_average_loss': avg_loss,
            'final_std_loss': std_loss,
            'all_fold_losses': fold_losses
        })
        accelerator.end_training()

    print("\n===== Training Complete =====")
    print(f"Results saved in: {args.results_dir}")


if __name__ == "__main__":
    main()
