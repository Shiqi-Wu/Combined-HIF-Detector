"""
Multi-node distributed training framework for LSTM classifier using HuggingFace Accelerate

Features:
- HuggingFace Accelerate for simplified distributed training
- Multi-node and multi-GPU support
- Gradient accumulation
- Fault tolerance and resume capability
- Comprehensive logging and monitoring
"""

import torch
torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import os
import sys
import time
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import math
import logging
from logging.handlers import RotatingFileHandler
import signal
import atexit
import traceback

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
from utils.dataloader import load_full_dataset, create_kfold_dataloaders, create_preprocessed_kfold_dataloaders, ScaledDataset

# Handle wandb import gracefully
WANDB_AVAILABLE = False
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not installed. Logging will be done to local files only.")
    wandb = None

def setup_logger(name, log_file=None, level=logging.INFO, console_output=True):
    """
    Set up a logger with both file and console handlers
    Enhanced with better error handling and forced flushing
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        console_output: Whether to output to console
    
    Returns:
        logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation and forced flushing
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        
        # Force immediate flushing
        class FlushingHandler(RotatingFileHandler):
            def emit(self, record):
                super().emit(record)
                self.flush()  # Force flush after each log message
        
        # Replace with flushing handler
        file_handler = FlushingHandler(
            log_file, 
            maxBytes=10*1024*1024,
            backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def setup_exception_handling(logger):
    """Setup global exception handling to ensure logs are captured"""
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions"""
        if issubclass(exc_type, KeyboardInterrupt):
            # Handle Ctrl+C gracefully
            logger.error("KeyboardInterrupt received - shutting down gracefully")
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        # Log the full traceback for other exceptions
        logger.critical("Uncaught exception occurred!", exc_info=(exc_type, exc_value, exc_traceback))
        logger.critical("Full traceback:")
        logger.critical(''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
        
        # Ensure all logs are flushed
        for handler in logger.handlers:
            handler.flush()
    
    def signal_handler(signum, frame):
        """Handle system signals"""
        logger.error(f"Received signal {signum} - attempting graceful shutdown")
        for handler in logger.handlers:
            handler.flush()
        sys.exit(1)
    
    def cleanup_on_exit():
        """Cleanup function called on normal exit"""
        logger.info("Program exiting - flushing all logs")
        for handler in logger.handlers:
            handler.flush()
    
    # Set up exception and signal handlers
    sys.excepthook = handle_exception
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(cleanup_on_exit)

class SafeLogger:
    """A wrapper around logger that ensures all messages are written immediately"""
    
    def __init__(self, logger):
        self.logger = logger
    
    def _log_and_flush(self, level, msg, *args, **kwargs):
        """Log message and immediately flush all handlers"""
        getattr(self.logger, level)(msg, *args, **kwargs)
        for handler in self.logger.handlers:
            handler.flush()
    
    def info(self, msg, *args, **kwargs):
        self._log_and_flush('info', msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        self._log_and_flush('error', msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        self._log_and_flush('warning', msg, *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        self._log_and_flush('debug', msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        self._log_and_flush('critical', msg, *args, **kwargs)

def log_system_info(logger):
    """Log system information"""
    logger.info("="*60)
    logger.info("SYSTEM INFORMATION")
    logger.info("="*60)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    logger.info(f"Accelerate available: {ACCELERATE_AVAILABLE}")
    logger.info(f"WandB available: {WANDB_AVAILABLE}")
    logger.info("="*60)

class AcceleratedLSTMTrainer:
    """
    Accelerated LSTM trainer using HuggingFace Accelerate framework
    Supports multi-node, multi-GPU training with automatic mixed precision
    """
    
    def __init__(self, model_config, train_config, accelerator=None, fold_idx=None):
        """
        Initialize the accelerated trainer
        
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
        
        # Setup logger
        self.setup_fold_logger()
        
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
        
        # Log initialization
        if self.accelerator.is_main_process:
            self.logger.info(f"Initializing trainer for fold {fold_idx + 1 if fold_idx is not None else 'N/A'}")
            self.logger.info(f"Process rank: {self.accelerator.process_index}/{self.accelerator.num_processes}")
            self.logger.info(f"Device: {self.accelerator.device}")
            self.logger.info(f"Mixed precision: {self.accelerator.mixed_precision}")
        
        # Set random seed for reproducibility (important for each fold)
        if train_config.get('seed'):
            # Add fold_idx to seed to ensure different initialization for each fold
            fold_seed = train_config['seed'] + (fold_idx if fold_idx is not None else 0)
            set_seed(fold_seed)
            # Ensure all processes have the same seed before proceeding
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self.logger.info(f"Set random seed to {fold_seed}")
        
        # Ensure all processes are synchronized before model creation
        self.accelerator.wait_for_everyone()
        
        # Initialize wandb only once per accelerator (not per fold)
        if (self.accelerator.is_main_process and 
            train_config.get('use_wandb', False) and 
            WANDB_AVAILABLE and 
            not self.shared_accelerator):
            # Only initialize wandb if this is not a shared accelerator
            self.accelerator.init_trackers(
                project_name=train_config.get('wandb_project', 'accelerated-lstm-training'),
                config={**model_config, **train_config},
                init_kwargs={"wandb": {"name": f"accelerated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}}
            )
            self.use_wandb = True
            self.logger.info("WandB logging initialized")
        else:
            self.use_wandb = False
            self.logger.info("WandB logging disabled")
        
        # Create model (fresh for each fold) - ensure consistent initialization across all processes
        if self.accelerator.is_main_process:
            self.logger.info(f"Creating model with config: {model_config}")
        
        self.model = LSTMClassifier(**model_config)
        
        # Log model structure for debugging
        if self.accelerator.is_main_process:
            self.logger.info(f"Model created successfully")
            self.logger.info(f"Model type: {type(self.model)}")
            param_count = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Total parameters before float64 conversion: {param_count}")
        
        # Ensure consistent model initialization across all processes
        self.accelerator.wait_for_everyone()
        
        # Convert to float64 after ensuring consistency
        self.model = self.model.to(torch.float64)
        
        model_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Log detailed parameter information
        if self.accelerator.is_main_process:
            self.logger.info(f"Model parameters after float64 conversion: {model_params}")
            self.logger.info(f"Trainable parameters: {trainable_params}")
            self.logger.info(f"Model device: {next(self.model.parameters()).device}")
            self.logger.info(f"Model dtype: {next(self.model.parameters()).dtype}")
        
        if self.accelerator.is_main_process:
            self.logger.info(f"Model created with {model_params:,} parameters ({trainable_params:,} trainable)")
            self.logger.info(f"Model config: {model_config}")
        
        # Setup optimizer and scheduler (fresh for each fold)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config.get('weight_decay', 1e-4)
        )
        
        if self.accelerator.is_main_process:
            self.logger.info(f"Optimizer: AdamW with lr={train_config['learning_rate']}, weight_decay={train_config.get('weight_decay', 1e-4)}")
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history (fresh for each fold)
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Performance metrics (fresh for each fold)
        self.performance_metrics = {
            'throughput': [],
            'memory_usage': [],
            'training_time': []
        }
        
        if self.accelerator.is_main_process:
            self.logger.info("Trainer initialization completed")
    
    def setup_fold_logger(self):
        """Setup logger for current fold"""
        fold_suffix = f"_fold_{self.fold_idx + 1}" if self.fold_idx is not None else ""
        results_dir = self.train_config.get('results_dir', './results')
        
        # Create logs directory
        logs_dir = os.path.join(results_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Setup logger
        log_file = os.path.join(logs_dir, f'training{fold_suffix}.log')
        logger_name = f"LSTM_Trainer{fold_suffix}"
        
        self.logger = setup_logger(
            name=logger_name,
            log_file=log_file,
            level=logging.INFO,
            console_output=True
        )
    
    def reset_for_new_fold(self, fold_idx):
        """
        Reset trainer state for a new fold while keeping the same accelerator
        """
        self.fold_idx = fold_idx
        
        # Setup new logger for this fold
        self.setup_fold_logger()
        
        if self.accelerator.is_main_process:
            self.logger.info(f"Resetting trainer for fold {fold_idx + 1}")
        
        # Reset random seed with fold-specific seed
        if self.train_config.get('seed'):
            fold_seed = self.train_config['seed'] + fold_idx
            set_seed(fold_seed)
            # Ensure all processes have the same seed before proceeding
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self.logger.info(f"Set random seed to {fold_seed}")
        
        # Ensure all processes are synchronized before model creation
        self.accelerator.wait_for_everyone()
        
        # Create new model for this fold - ensure consistent initialization
        if self.accelerator.is_main_process:
            self.logger.info(f"Creating new model for fold {fold_idx + 1} with config: {self.model_config}")
        
        self.model = LSTMClassifier(**self.model_config)
        
        # Log model structure for debugging
        if self.accelerator.is_main_process:
            self.logger.info(f"Model created successfully for fold {fold_idx + 1}")
            param_count = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Total parameters before float64 conversion: {param_count}")
        
        # Ensure consistent model initialization across all processes
        self.accelerator.wait_for_everyone()
        
        # Convert to float64 after ensuring consistency
        self.model = self.model.to(torch.float64)
        
        model_params = sum(p.numel() for p in self.model.parameters())
        
        # Log detailed parameter information
        if self.accelerator.is_main_process:
            self.logger.info(f"Model parameters after float64 conversion: {model_params}")
            self.logger.info(f"Model device: {next(self.model.parameters()).device}")
            self.logger.info(f"Model dtype: {next(self.model.parameters()).dtype}")
            self.logger.info(f"Created new model with {model_params:,} parameters")
        
        # Create new optimizer for this fold
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.train_config['learning_rate'],
            weight_decay=self.train_config.get('weight_decay', 1e-4)
        )
        
        # Reset training history
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Reset performance metrics
        self.performance_metrics = {
            'throughput': [],
            'memory_usage': [],
            'training_time': []
        }
        
        if self.accelerator.is_main_process:
            self.logger.info("Fold reset completed")
    
    def create_scheduler(self, optimizer):
        """Create learning rate scheduler (fresh for each fold)"""
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=10, 
            factor=0.5,
            verbose=self.accelerator.is_main_process
        )
    
    def prepare_data_loaders(self, train_dataset, val_dataset):
        """Prepare data loaders for distributed training"""
        if self.accelerator.is_main_process:
            self.logger.info(f"Preparing data loaders:")
            self.logger.info(f"  Train dataset size: {len(train_dataset)}")
            self.logger.info(f"  Validation dataset size: {len(val_dataset)}")
            self.logger.info(f"  Batch size: {self.train_config['batch_size']}")
            
            # Test dataset access
            try:
                sample = train_dataset[0]
                self.logger.info(f"  Train sample format: {type(sample)}")
                if isinstance(sample, (tuple, list)):
                    self.logger.info(f"  Train sample length: {len(sample)}")
                    for i, item in enumerate(sample):
                        if hasattr(item, 'shape'):
                            self.logger.info(f"    Item {i} shape: {item.shape}")
                        else:
                            self.logger.info(f"    Item {i} type: {type(item)}")
            except Exception as e:
                self.logger.error(f"  Error accessing train dataset sample: {e}")
        
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
        
        if self.accelerator.is_main_process:
            self.logger.info(f"  Train batches: {len(train_loader)}")
            self.logger.info(f"  Validation batches: {len(val_loader)}")
            
            # Test data loader access
            try:
                batch = next(iter(train_loader))
                self.logger.info(f"  First batch type: {type(batch)}")
                if isinstance(batch, (tuple, list)):
                    self.logger.info(f"  First batch length: {len(batch)}")
                    for i, item in enumerate(batch):
                        if hasattr(item, 'shape'):
                            self.logger.info(f"    Batch item {i} shape: {item.shape}")
                        else:
                            self.logger.info(f"    Batch item {i} type: {type(item)}")
            except Exception as e:
                self.logger.error(f"  Error accessing train loader batch: {e}")
        
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
            # Debug batch content
            if batch is None:
                if self.accelerator.is_main_process:
                    self.logger.error(f"Received None batch at index {batch_idx}")
                continue
            
            # Check if batch is a tuple/list with expected length
            if not isinstance(batch, (tuple, list)) or len(batch) != 3:
                if self.accelerator.is_main_process:
                    self.logger.error(f"Unexpected batch format at index {batch_idx}: {type(batch)}, length: {len(batch) if hasattr(batch, '__len__') else 'N/A'}")
                continue
                
            try:
                x_batch, u_batch, p_batch = batch
            except Exception as e:
                if self.accelerator.is_main_process:
                    self.logger.error(f"Failed to unpack batch at index {batch_idx}: {e}")
                continue
            
            x_batch = x_batch.to(torch.float64).to(self.accelerator.device)
            u_batch = u_batch.to(torch.float64).to(self.accelerator.device)
            
            # Convert one-hot to class indices if needed
            if p_batch.dim() > 1 and p_batch.size(1) > 1:
                p_indices = torch.argmax(p_batch, dim=1)
            else:
                p_indices = p_batch.long()
            
            # Move targets to the same device as model
            p_indices = p_indices.to(self.accelerator.device)
            
            with self.accelerator.accumulate(self.model):
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, p_indices)
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping (only if not using gradient accumulation or on the last accumulation step)
                if self.train_config.get('grad_clip') and self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), 
                        self.train_config['grad_clip']
                    )
                
                # Optimizer step
                self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += p_indices.size(0)
            total_correct += (predicted == p_indices).sum().item()
            
            # Update progress bar
            if self.accelerator.is_main_process and hasattr(progress_bar, 'set_postfix'):
                current_acc = total_correct / total_samples
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'Loss': f'{total_loss/(batch_idx+1):.4f}',
                    'Acc': f'{current_acc:.4f}',
                    'LR': f'{current_lr:.6f}'
                })
        
        # Calculate metrics
        if len(train_loader) > 0:
            avg_loss = total_loss / len(train_loader)
        else:
            avg_loss = float('inf')
            
        if total_samples > 0:
            accuracy = total_correct / total_samples
        else:
            accuracy = 0.0
        
        # Calculate throughput
        epoch_time = time.time() - start_time
        if epoch_time > 0 and total_samples > 0:
            samples_per_sec = total_samples / epoch_time
            self.performance_metrics['throughput'].append(samples_per_sec)
        else:
            self.performance_metrics['throughput'].append(0.0)
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Debug batch content
                if batch is None:
                    continue
                
                # Check if batch is a tuple/list with expected length
                if not isinstance(batch, (tuple, list)) or len(batch) != 3:
                    continue
                    
                try:
                    x_batch, u_batch, p_batch = batch
                except Exception as e:
                    continue
                
                x_batch = x_batch.to(torch.float64).to(self.accelerator.device)
                u_batch = u_batch.to(torch.float64).to(self.accelerator.device)
                
                # Convert one-hot to class indices if needed
                if p_batch.dim() > 1 and p_batch.size(1) > 1:
                    p_indices = torch.argmax(p_batch, dim=1)
                else:
                    p_indices = p_batch.long()
                
                # Move targets to the same device as model
                p_indices = p_indices.to(self.accelerator.device)
                
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, p_indices)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += p_indices.size(0)
                total_correct += (predicted == p_indices).sum().item()
        
        if len(val_loader) > 0:
            avg_loss = total_loss / len(val_loader)
        else:
            avg_loss = float('inf')
            
        if total_samples > 0:
            accuracy = total_correct / total_samples
        else:
            accuracy = 0.0
        
        return avg_loss, accuracy
    
    def evaluate_model_on_datasets(self, train_dataset, val_dataset, test_dataset):
        """Evaluate model on train, validation, and test datasets"""
        self.model.eval()
        results = {}
        
        # Evaluate on each dataset
        datasets = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
        
        with torch.no_grad():
            for dataset_name, dataset in datasets.items():
                if dataset is None:
                    continue
                    
                # Create temporary dataloader for evaluation
                eval_loader = DataLoader(
                    dataset,
                    batch_size=self.train_config['batch_size'],
                    shuffle=False,
                    num_workers=0
                )
                
                total_loss = 0
                total_correct = 0
                total_samples = 0
                
                for batch in eval_loader:
                    # Debug batch content
                    if batch is None:
                        continue
                    
                    # Check if batch is a tuple/list with expected length
                    if not isinstance(batch, (tuple, list)) or len(batch) != 3:
                        continue
                        
                    try:
                        x_batch, u_batch, p_batch = batch
                    except Exception as e:
                        continue
                    x_batch = x_batch.to(torch.float64).to(self.accelerator.device)
                    u_batch = u_batch.to(torch.float64).to(self.accelerator.device)
                    
                    # Convert one-hot to class indices if needed
                    if p_batch.dim() > 1 and p_batch.size(1) > 1:
                        p_indices = torch.argmax(p_batch, dim=1)
                    else:
                        p_indices = p_batch.long()
                    
                    # Move targets to the same device as model
                    p_indices = p_indices.to(self.accelerator.device)
                    
                    outputs = self.model(x_batch)
                    loss = self.criterion(outputs, p_indices)
                    
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_samples += p_indices.size(0)
                    total_correct += (predicted == p_indices).sum().item()
                
                if len(eval_loader) > 0:
                    avg_loss = total_loss / len(eval_loader)
                else:
                    avg_loss = float('inf')
                    
                if total_samples > 0:
                    accuracy = total_correct / total_samples
                else:
                    accuracy = 0.0
                
                results[dataset_name] = {
                    'loss': avg_loss,
                    'accuracy': accuracy
                }
                
                if self.accelerator.is_main_process:
                    self.logger.info(f"  {dataset_name.capitalize()} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return results
    
    def save_evaluation_to_csv(self, epoch, results):
        """Save evaluation results to CSV file"""
        if not self.accelerator.is_main_process:
            return
        
        # Create CSV file path
        fold_suffix = f"_fold_{self.fold_idx + 1}" if self.fold_idx is not None else ""
        csv_path = os.path.join(
            self.train_config['results_dir'], 
            f'evaluation_results{fold_suffix}.csv'
        )
        
        # Prepare data for CSV
        csv_data = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().isoformat(),
            'fold': self.fold_idx + 1 if self.fold_idx is not None else 0
        }
        
        # Add train, val, test metrics
        for dataset_name in ['train', 'val', 'test']:
            if dataset_name in results:
                csv_data[f'{dataset_name}_loss'] = results[dataset_name]['loss']
                csv_data[f'{dataset_name}_accuracy'] = results[dataset_name]['accuracy']
            else:
                csv_data[f'{dataset_name}_loss'] = None
                csv_data[f'{dataset_name}_accuracy'] = None
        
        # Create DataFrame
        df = pd.DataFrame([csv_data])
        
        # Write to CSV (append if file exists)
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, mode='w', header=True, index=False)
        
        self.logger.info(f"Evaluation results saved to {csv_path}")
    
    def save_checkpoint(self, epoch, val_acc, train_dataset=None, val_dataset=None, test_dataset=None, is_best=False):
        """Save checkpoint and evaluate model on all datasets if it's the best model"""
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
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model and evaluate on all datasets
        if is_best:
            best_path = os.path.join(
                self.train_config['results_dir'], 
                f'best_model{fold_suffix}.pth'
            )
            torch.save(checkpoint, best_path)
            fold_info = f"Fold {self.fold_idx + 1} - " if self.fold_idx is not None else ""
            self.logger.info(f"{fold_info}New best model saved with validation accuracy: {val_acc:.4f}")
            self.accelerator.print(f"{fold_info}New best model saved with validation accuracy: {val_acc:.4f}")
            
            # Evaluate on all datasets and save to CSV
            if train_dataset is not None or val_dataset is not None or test_dataset is not None:
                self.logger.info(f"{fold_info}Evaluating best model on all datasets...")
                results = self.evaluate_model_on_datasets(train_dataset, val_dataset, test_dataset)
                self.save_evaluation_to_csv(epoch, results)
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint for resuming training"""
        if os.path.exists(checkpoint_path):
            self.logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Load model and optimizer states
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.train_history = checkpoint.get('train_history', self.train_history)
            
            if self.accelerator.is_main_process:
                fold_info = f"Fold {checkpoint.get('fold_idx', 0) + 1} - " if checkpoint.get('fold_idx') is not None else ""
                self.logger.info(f"{fold_info}Checkpoint loaded successfully")
                self.logger.info(f"{fold_info}Resuming from epoch {checkpoint['epoch']} with val_acc {checkpoint['val_accuracy']:.4f}")
                self.accelerator.print(f"{fold_info}Checkpoint loaded from {checkpoint_path}")
                self.accelerator.print(f"{fold_info}Resuming from epoch {checkpoint['epoch']} with val_acc {checkpoint['val_accuracy']:.4f}")
            
            return checkpoint['epoch'], checkpoint['val_accuracy']
        else:
            self.logger.info(f"No checkpoint found at {checkpoint_path}, starting from scratch")
        
        return 0, 0.0
    
    def train(self, train_dataset, val_dataset, test_dataset=None):
        """Main training loop"""
        if self.accelerator.is_main_process:
            self.logger.info("="*60)
            self.logger.info("STARTING TRAINING")
            self.logger.info("="*60)
        
        # Prepare data loaders
        train_loader, val_loader = self.prepare_data_loaders(train_dataset, val_dataset)
        
        # Calculate total training steps
        num_epochs = self.train_config['num_epochs']
        gradient_accumulation_steps = self.train_config.get('gradient_accumulation_steps', 1)
        num_training_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        
        if self.accelerator.is_main_process:
            self.logger.info(f"Training configuration:")
            self.logger.info(f"  Number of epochs: {num_epochs}")
            self.logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
            self.logger.info(f"  Total training steps: {num_training_steps}")
            self.logger.info(f"  Patience: {self.train_config.get('patience', 20)}")
        
        # Create scheduler (fresh for each fold)
        self.scheduler = self.create_scheduler(self.optimizer)
        
        # Ensure all processes are synchronized before preparing with accelerator
        self.accelerator.wait_for_everyone()
        
        # Log model info before preparation on all processes
        model_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Process {self.accelerator.process_index}: Model has {model_params} parameters before preparation")
        
        # Another synchronization point
        self.accelerator.wait_for_everyone()
        
        # Prepare everything with accelerator
        # Note: This creates new distributed wrappers for each fold
        self.model, self.optimizer, train_loader, val_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, train_loader, val_loader, self.scheduler
        )
        
        # Verify model consistency after preparation on all processes
        prepared_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Process {self.accelerator.process_index}: Model has {prepared_params} parameters after preparation")
        
        if self.accelerator.is_main_process:
            self.logger.info("Model and optimizers prepared with accelerator")
        
        # Try to resume from checkpoint (fold-specific)
        fold_suffix = f"_fold_{self.fold_idx + 1}" if self.fold_idx is not None else ""
        checkpoint_path = os.path.join(self.train_config['results_dir'], f'latest_checkpoint{fold_suffix}.pth')
        start_epoch, best_val_acc = self.load_checkpoint(checkpoint_path)
        
        # Training parameters
        patience_counter = 0
        patience = self.train_config.get('patience', 20)
        
        if self.accelerator.is_main_process:
            fold_info = f"Fold {self.fold_idx + 1}" if self.fold_idx is not None else "Training"
            self.logger.info(f"Starting {fold_info} on {self.accelerator.num_processes} processes")
            self.logger.info(f"Device: {self.accelerator.device}")
            self.logger.info(f"Mixed precision: {self.accelerator.mixed_precision}")
            self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            self.accelerator.print(f"Starting {fold_info} on {self.accelerator.num_processes} processes")
            self.accelerator.print(f"Device: {self.accelerator.device}")
            self.accelerator.print(f"Mixed precision: {self.accelerator.mixed_precision}")
            self.accelerator.print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
            # Create results directory
            os.makedirs(self.train_config['results_dir'], exist_ok=True)
        
        # Wait for all processes
        self.accelerator.wait_for_everyone()
        
        # Training loop
        training_start_time = time.time()
        if self.accelerator.is_main_process:
            self.logger.info("Starting training loop...")
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Learning rate scheduling (using ReduceLROnPlateau)
            self.scheduler.step(val_loss)
            
            # Record metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start_time
            
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
            self.save_checkpoint(epoch, val_acc, train_dataset, val_dataset, test_dataset, is_best)
            
            # Logging (main process only)
            if self.accelerator.is_main_process:
                fold_info = f"Fold {self.fold_idx + 1} - " if self.fold_idx is not None else ""
                
                # Log to file and console
                self.logger.info(f"{fold_info}Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f}s")
                self.logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                self.logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                self.logger.info(f"  Learning Rate: {current_lr:.6f}")
                self.logger.info(f"  Best Val Acc: {best_val_acc:.4f}, Patience: {patience_counter}/{patience}")
                if self.performance_metrics['throughput']:
                    self.logger.info(f"  Throughput: {self.performance_metrics['throughput'][-1]:.2f} samples/sec")
                
                # Also print to console for immediate feedback
                self.accelerator.print(f"{fold_info}Epoch [{epoch+1}/{num_epochs}]:")
                self.accelerator.print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                self.accelerator.print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                self.accelerator.print(f"  Learning Rate: {current_lr:.6f}")
                self.accelerator.print(f"  Epoch Time: {epoch_time:.2f}s")
                if self.performance_metrics['throughput']:
                    self.accelerator.print(f"  Throughput: {self.performance_metrics['throughput'][-1]:.2f} samples/sec")
                self.accelerator.print("-" * 60)
                
                # Log to wandb with fold information
                if self.use_wandb:
                    log_dict = {
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'train_accuracy': train_acc,
                        'val_loss': val_loss,
                        'val_accuracy': val_acc,
                        'learning_rate': current_lr,
                        'epoch_time': epoch_time,
                        'best_val_accuracy': best_val_acc,
                        'patience_counter': patience_counter
                    }
                    if self.fold_idx is not None:
                        log_dict['fold'] = self.fold_idx + 1
                    if self.performance_metrics['throughput']:
                        log_dict['throughput'] = self.performance_metrics['throughput'][-1]
                    
                    self.accelerator.log(log_dict)
            
            # Early stopping
            if patience_counter >= patience:
                if self.accelerator.is_main_process:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    self.accelerator.print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Training completion
        total_training_time = time.time() - training_start_time
        if self.accelerator.is_main_process:
            fold_info = f"Fold {self.fold_idx + 1} " if self.fold_idx is not None else ""
            self.logger.info("="*60)
            self.logger.info(f"{fold_info}TRAINING COMPLETED")
            self.logger.info("="*60)
            self.logger.info(f"Total training time: {total_training_time:.2f}s")
            self.logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
            self.logger.info(f"Final learning rate: {current_lr:.6f}")
            
            self.accelerator.print(f"{fold_info}training completed. Best validation accuracy: {best_val_acc:.4f}")
            
            # Save final training history with fold info
            history_filename = f'training_history_fold_{self.fold_idx + 1}.json' if self.fold_idx is not None else 'training_history.json'
            history_path = os.path.join(self.train_config['results_dir'], history_filename)
            with open(history_path, 'w') as f:
                json.dump(self.train_history, f, indent=2)
            self.logger.info(f"Training history saved to {history_path}")
            
            # Save performance metrics with fold info
            metrics_filename = f'performance_metrics_fold_{self.fold_idx + 1}.json' if self.fold_idx is not None else 'performance_metrics.json'
            metrics_path = os.path.join(self.train_config['results_dir'], metrics_filename)
            # Add summary statistics to performance metrics
            self.performance_metrics['total_training_time'] = total_training_time
            self.performance_metrics['best_val_accuracy'] = best_val_acc
            self.performance_metrics['final_learning_rate'] = current_lr
            with open(metrics_path, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
            self.logger.info(f"Performance metrics saved to {metrics_path}")
        
        return best_val_acc
def main():
    parser = argparse.ArgumentParser(description='K-Fold Accelerated LSTM Training')

    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--window_size', type=int, default=30, help='Window size')
    parser.add_argument('--sample_step', type=int, default=1, help='Sampling step')
    parser.add_argument('--pca_dim', type=int, default=2, help='PCA dimension')

    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--bidirectional', action='store_true', default=True, help='Use bidirectional LSTM')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Logging
    parser.add_argument('--results_dir', type=str, default='./results_kfold', help='Root directory to store fold results')
    parser.add_argument('--use_wandb', action='store_true', help='Use WandB logging')
    parser.add_argument('--wandb_project', type=str, default='kfold-lstm-training', help='WandB project name')

    # K-Fold
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds')

    args = parser.parse_args()

    # Setup main logger
    experiment_name = f"kfold_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    main_log_file = os.path.join(args.results_dir, 'logs', 'main_experiment.log')
    main_logger = setup_logger(
        name="MAIN_EXPERIMENT",
        log_file=main_log_file,
        level=logging.INFO,
        console_output=True
    )
    
    # Setup exception handling early to catch all errors
    setup_exception_handling(main_logger)

    # Log system info
    log_system_info(main_logger)
    
    # Log experiment configuration
    main_logger.info("="*60)
    main_logger.info("EXPERIMENT CONFIGURATION")
    main_logger.info("="*60)
    for arg, value in vars(args).items():
        main_logger.info(f"{arg}: {value}")
    main_logger.info("="*60)

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
            init_kwargs={"wandb": {"name": experiment_name}}
        )
        main_logger.info(f"WandB tracking initialized with experiment name: {experiment_name}")
    
    # Step 1: Load full dataset
    data_config = {
        'window_size': args.window_size,
        'sample_step': args.sample_step,
        'batch_size': args.batch_size
    }

    main_logger.info("\n===== Loading Dataset =====")
    print("\n===== Loading Dataset =====")
    dataset, file_labels = load_full_dataset(args.data_dir, data_config)
    main_logger.info(f"Loaded dataset with {len(dataset)} samples from {len(file_labels)} files")

    # Step 2: Create K-Fold Dataloaders with shared preprocessing
    main_logger.info("\n===== Creating K-Fold Splits with Shared Preprocessing =====")
    print("\n===== Creating K-Fold Splits with Shared Preprocessing =====")
    folds, test_loader, preprocessing_params = create_preprocessed_kfold_dataloaders(
        dataset,
        file_labels,
        config=data_config,
        n_splits=args.k_folds,
        random_state=args.seed,
        pca_dim=args.pca_dim
    )
    main_logger.info(f"Created {args.k_folds} folds with PCA dimension {args.pca_dim}")
    main_logger.info(f"Test set size: {len(test_loader.dataset)} samples")

    # Step 3: Train across folds with shared accelerator
    fold_accuracies = []
    trainer = None  # Initialize trainer once
    
    for fold_idx, (train_loader, val_loader) in enumerate(folds):
        main_logger.info(f"\n========================\n Fold {fold_idx + 1}/{args.k_folds}\n========================")
        print(f"\n========================\n Fold {fold_idx + 1}/{args.k_folds}\n========================")

        fold_result_dir = os.path.join(args.results_dir, f"fold_{fold_idx + 1}")
        os.makedirs(fold_result_dir, exist_ok=True)

        model_config = {
            'input_size': train_loader.dataset[0][0].shape[1],
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'num_classes': 6,  # ErrorType: 2~7
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
            'seed': args.seed,
            'results_dir': fold_result_dir,
            'use_wandb': args.use_wandb,
            'wandb_project': args.wandb_project
        }

        main_logger.info(f"Model config: {model_config}")
        main_logger.info(f"Train config subset: batch_size={train_config['batch_size']}, lr={train_config['learning_rate']}")

        if trainer is None:
            # Create trainer for first fold
            trainer = AcceleratedLSTMTrainer(model_config, train_config, accelerator=accelerator, fold_idx=fold_idx)
        else:
            # Reset trainer for subsequent folds
            trainer.reset_for_new_fold(fold_idx)
            trainer.train_config['results_dir'] = fold_result_dir  # Update results directory

        print(model_config)
        
        fold_start_time = time.time()
        best_val_acc = trainer.train(train_loader.dataset, val_loader.dataset, test_loader.dataset)
        fold_time = time.time() - fold_start_time
        
        fold_accuracies.append(best_val_acc)
        main_logger.info(f"Fold {fold_idx + 1} completed in {fold_time:.2f}s with accuracy {best_val_acc:.4f}")
        
        # Log fold completion to wandb
        if accelerator.is_main_process and args.use_wandb and WANDB_AVAILABLE:
            accelerator.log({
                f'fold_{fold_idx + 1}_final_accuracy': best_val_acc,
                f'fold_{fold_idx + 1}_training_time': fold_time,
                'completed_folds': fold_idx + 1
            })

    # Step 4: Summary
    main_logger.info("\n===== K-Fold Summary =====")
    print("\n===== K-Fold Summary =====")
    for i, acc in enumerate(fold_accuracies):
        main_logger.info(f"Fold {i + 1}: {acc:.4f}")
        print(f"Fold {i + 1}: {acc:.4f}")
    
    avg_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    main_logger.info(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    
    # Save summary results
    summary_results = {
        'experiment_name': experiment_name,
        'fold_accuracies': fold_accuracies,
        'average_accuracy': avg_accuracy,
        'std_accuracy': std_accuracy,
        'config': vars(args),
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = os.path.join(args.results_dir, 'kfold_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=2)
    main_logger.info(f"Summary results saved to {summary_path}")
    
    # Create combined evaluation results CSV
    if accelerator.is_main_process:
        main_logger.info("Creating combined evaluation results CSV...")
        combined_csv_path = os.path.join(args.results_dir, 'combined_evaluation_results.csv')
        
        # Collect all fold evaluation results
        all_fold_results = []
        for fold_idx in range(args.k_folds):
            fold_csv_path = os.path.join(args.results_dir, f"fold_{fold_idx + 1}", f'evaluation_results_fold_{fold_idx + 1}.csv')
            if os.path.exists(fold_csv_path):
                try:
                    fold_df = pd.read_csv(fold_csv_path)
                    all_fold_results.append(fold_df)
                    main_logger.info(f"  Added fold {fold_idx + 1} evaluation results")
                except Exception as e:
                    main_logger.warning(f"  Could not read fold {fold_idx + 1} results: {e}")
        
        # Combine all results
        if all_fold_results:
            combined_df = pd.concat(all_fold_results, ignore_index=True)
            combined_df.to_csv(combined_csv_path, index=False)
            main_logger.info(f"Combined evaluation results saved to {combined_csv_path}")
            
            # Show summary statistics
            main_logger.info("Evaluation Summary:")
            for dataset in ['train', 'val', 'test']:
                acc_col = f'{dataset}_accuracy'
                loss_col = f'{dataset}_loss'
                if acc_col in combined_df.columns:
                    mean_acc = combined_df[acc_col].mean()
                    std_acc = combined_df[acc_col].std()
                    mean_loss = combined_df[loss_col].mean()
                    std_loss = combined_df[loss_col].std()
                    main_logger.info(f"  {dataset.upper()} - Accuracy: {mean_acc:.4f} ± {std_acc:.4f}, Loss: {mean_loss:.4f} ± {std_loss:.4f}")
        else:
            main_logger.warning("No fold evaluation results found to combine")
    
    # Final logging to wandb
    if accelerator.is_main_process and args.use_wandb and WANDB_AVAILABLE:
        accelerator.log({
            'final_average_accuracy': avg_accuracy,
            'final_std_accuracy': std_accuracy,
            'all_fold_accuracies': fold_accuracies
        })
        accelerator.end_training()
        main_logger.info("WandB tracking ended")

    main_logger.info(f"\n===== Training Complete =====")
    main_logger.info(f"Results saved in: {args.results_dir}")
    main_logger.info(f"Logs saved in: {os.path.join(args.results_dir, 'logs')}")
    print(f"\n===== Training Complete =====")
    print(f"Results saved in: {args.results_dir}")
    print(f"Logs saved in: {os.path.join(args.results_dir, 'logs')}")

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main()
