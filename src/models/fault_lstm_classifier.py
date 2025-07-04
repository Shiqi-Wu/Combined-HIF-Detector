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
from utils.dataloader import load_full_dataset, create_kfold_dataloaders

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
    
    def __init__(self, input_size=68, hidden_size=128, num_layers=2, 
                 num_classes=6, dropout=0.2, bidirectional=True):
        super(LSTMClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate the size of LSTM output
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
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
            self.hidden_size
        ).to(x.device)
        
        c0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            batch_size,
            self.hidden_size
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

class LSTMTrainer:
    """Trainer class for LSTM classifier with wandb logging and CSV recording"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', use_wandb=True, results_dir='./results', fold_idx=None):
        self.model = model.to(device)
        self.device = device
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.results_dir = results_dir
        self.fold_idx = fold_idx
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.epoch_data = []  # For CSV recording
        
        # Create CSV file for this fold if not using wandb
        if not self.use_wandb:
            os.makedirs(self.results_dir, exist_ok=True)
            if fold_idx is not None:
                self.csv_filename = os.path.join(self.results_dir, f'training_log_fold_{fold_idx}.csv')
            else:
                self.csv_filename = os.path.join(self.results_dir, 'training_log.csv')
            
            # Initialize CSV with headers
            with open(self.csv_filename, 'w') as f:
                f.write('epoch,train_loss,train_accuracy,val_loss,val_accuracy,learning_rate\n')
        
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (x_batch, u_batch, p_batch) in enumerate(tqdm(train_loader, desc="Training")):
            # Move data to device
            x_batch = x_batch.to(self.device)  # Shape: (batch_size, seq_len, input_size)
            p_batch = p_batch.to(self.device)  # Shape: (batch_size, num_classes)
            
            # Convert one-hot to class indices
            p_indices = torch.argmax(p_batch, dim=1)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(x_batch)
            loss = criterion(outputs, p_indices)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += p_indices.size(0)
            total_correct += (predicted == p_indices).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for x_batch, u_batch, p_batch in tqdm(val_loader, desc="Validation"):
                # Move data to device
                x_batch = x_batch.to(self.device)
                p_batch = p_batch.to(self.device)
                
                # Convert one-hot to class indices
                p_indices = torch.argmax(p_batch, dim=1)
                
                # Forward pass
                outputs = self.model(x_batch)
                loss = criterion(outputs, p_indices)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += p_indices.size(0)
                total_correct += (predicted == p_indices).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
        """Train the model"""
        # Setup optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        
        best_val_accuracy = 0
        patience_counter = 0
        patience = 20
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Print progress
            print(f"Epoch [{epoch+1}/{num_epochs}]:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 50)
            
            # Log to wandb or CSV
            current_lr = optimizer.param_groups[0]['lr']
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'learning_rate': current_lr
                })
            else:
                # Log to CSV file
                with open(self.csv_filename, 'a') as f:
                    f.write(f'{epoch + 1},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f},{current_lr:.8f}\n')
                
                # Store epoch data for later analysis
                self.epoch_data.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'learning_rate': current_lr
                })
            
            # Early stopping
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter = 0
                # Save best model
                if self.fold_idx is not None:
                    model_path = os.path.join(self.results_dir, f'best_model_fold_{self.fold_idx}.pth')
                else:
                    model_path = os.path.join(self.results_dir, 'best_lstm_classifier.pth')
                torch.save(self.model.state_dict(), model_path)
                print(f"New best validation accuracy: {best_val_accuracy:.4f}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        if self.fold_idx is not None:
            model_path = os.path.join(self.results_dir, f'best_model_fold_{self.fold_idx}.pth')
        else:
            model_path = os.path.join(self.results_dir, 'best_lstm_classifier.pth')
        self.model.load_state_dict(torch.load(model_path))
        print(f"Training completed. Best validation accuracy: {best_val_accuracy:.4f}")
        
        # Save final training log summary if not using wandb
        if not self.use_wandb:
            self._save_training_summary()
        
        return best_val_accuracy
    
    def _save_training_summary(self):
        """Save training summary to JSON file"""
        summary = {
            'fold': self.fold_idx,
            'total_epochs': len(self.epoch_data),
            'best_val_accuracy': max([epoch['val_accuracy'] for epoch in self.epoch_data]),
            'final_train_loss': self.epoch_data[-1]['train_loss'] if self.epoch_data else None,
            'final_val_loss': self.epoch_data[-1]['val_loss'] if self.epoch_data else None,
            'training_data': self.epoch_data
        }
        
        if self.fold_idx is not None:
            summary_path = os.path.join(self.results_dir, f'training_summary_fold_{self.fold_idx}.json')
        else:
            summary_path = os.path.join(self.results_dir, 'training_summary.json')
            
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Training summary saved to: {summary_path}")
    
    def evaluate(self, test_loader, fold_idx=None):
        """Evaluate the model on test set"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for x_batch, u_batch, p_batch in tqdm(test_loader, desc="Testing"):
                x_batch = x_batch.to(self.device)
                p_batch = p_batch.to(self.device)
                
                # Convert one-hot to class indices
                p_indices = torch.argmax(p_batch, dim=1)
                
                # Forward pass
                outputs = self.model(x_batch)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(p_indices.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1-Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, 
                                  target_names=[f"Class {i+2}" for i in range(6)]))
        
        # Log to wandb
        if self.use_wandb:
            test_metrics = {
                'test_accuracy': accuracy,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1
            }
            if fold_idx is not None:
                test_metrics['fold'] = fold_idx
            wandb.log(test_metrics)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[f"Class {i+2}" for i in range(6)],
                   yticklabels=[f"Class {i+2}" for i in range(6)])
        plt.title(f'Confusion Matrix - Fold {fold_idx}' if fold_idx is not None else 'Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save confusion matrix
        cm_filename = f'confusion_matrix_fold_{fold_idx}.png' if fold_idx is not None else 'confusion_matrix.png'
        plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
        
        if self.use_wandb:
            wandb.log({f"confusion_matrix_fold_{fold_idx}": wandb.Image(cm_filename)})
        
        plt.show()
        
        return accuracy, precision, recall, f1
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='LSTM Classifier for Power Grid Fault Detection')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, 
                       default='/Users/shiqi/Documents/PhD/Code/Project3-power-grid/Combined-HIF-Detector/data',
                       help='Path to data directory')
    parser.add_argument('--sample_step', type=int, default=1, help='Sampling step for data')
    parser.add_argument('--window_size', type=int, default=30, help='Window size for sequences')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--bidirectional', action='store_true', default=True, help='Use bidirectional LSTM')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    
    # K-fold parameters
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross validation')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    
    # Logging parameters
    parser.add_argument('--use_wandb', action='store_true', default=True, help='Use wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='power-grid-fault-detection', help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity name')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')
    
    # Output parameters
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--save_models', action='store_true', default=True, help='Save trained models')
    
    return parser.parse_args()

def main():
    """Main training function with k-fold cross validation"""
    args = parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Configuration from args
    config = {
        'sample_step': args.sample_step,
        'window_size': args.window_size,
        'batch_size': args.batch_size
    }
    
    # Initialize wandb if requested and available
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.experiment_name,
            config=vars(args)
        )
    elif args.use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not available. Using CSV logging only.")
    
    print("Loading complete dataset...")
    dataset, file_labels = load_full_dataset(args.data_dir, config)
    
    print("Creating k-fold cross validation splits...")
    fold_dataloaders = create_kfold_dataloaders(
        dataset, file_labels, config, 
        n_splits=args.n_folds, 
        random_state=args.random_state
    )
    
    # Get input size from data
    sample_batch = next(iter(fold_dataloaders[0][0]))  # First fold, train loader
    x_sample, u_sample, p_sample = sample_batch
    input_size = x_sample.shape[2]  # Number of features
    
    print(f"Input size (number of features): {input_size}")
    print(f"Sequence length: {x_sample.shape[1]}")
    print(f"Number of classes: {p_sample.shape[1]}")
    
    # Results storage
    fold_results = []
    
    # Train and evaluate for each fold
    for fold_idx, (train_loader, val_loader, test_loader) in enumerate(fold_dataloaders):
        print(f"\n{'='*50}")
        print(f"Training Fold {fold_idx + 1}/{args.n_folds}")
        print(f"{'='*50}")
        
        # Model configuration
        model_config = {
            'input_size': input_size,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'num_classes': 6,  # Classes 2-7 (6 classes total)
            'dropout': args.dropout,
            'bidirectional': args.bidirectional
        }
        
        # Create model
        model = LSTMClassifier(**model_config)
        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create trainer with fold information
        trainer = LSTMTrainer(
            model, 
            use_wandb=args.use_wandb, 
            results_dir=args.results_dir, 
            fold_idx=fold_idx + 1
        )
        
        # Train model
        print(f"\nStarting training for fold {fold_idx + 1}...")
        best_val_accuracy = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate
        )
        
        # Evaluate on test set
        print(f"\nEvaluating fold {fold_idx + 1} on test set:")
        test_accuracy, test_precision, test_recall, test_f1 = trainer.evaluate(test_loader, fold_idx + 1)
        
        # Store results
        fold_result = {
            'fold': fold_idx + 1,
            'best_val_accuracy': best_val_accuracy,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'test_samples': len(test_loader.dataset)
        }
        fold_results.append(fold_result)
        
        # Save model if requested
        if args.save_models:
            model_path = os.path.join(args.results_dir, f'best_model_fold_{fold_idx + 1}.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        
        # Plot training history for this fold
        trainer.plot_training_history()
        history_path = os.path.join(args.results_dir, f'training_history_fold_{fold_idx + 1}.png')
        plt.savefig(history_path, dpi=300, bbox_inches='tight')
        
        if args.use_wandb:
            wandb.log({f"training_history_fold_{fold_idx + 1}": wandb.Image(history_path)})
    
    # Calculate overall statistics
    test_accuracies = [result['test_accuracy'] for result in fold_results]
    test_precisions = [result['test_precision'] for result in fold_results]
    test_recalls = [result['test_recall'] for result in fold_results]
    test_f1s = [result['test_f1'] for result in fold_results]
    
    overall_stats = {
        'mean_test_accuracy': np.mean(test_accuracies),
        'std_test_accuracy': np.std(test_accuracies),
        'mean_test_precision': np.mean(test_precisions),
        'std_test_precision': np.std(test_precisions),
        'mean_test_recall': np.mean(test_recalls),
        'std_test_recall': np.std(test_recalls),
        'mean_test_f1': np.mean(test_f1s),
        'std_test_f1': np.std(test_f1s)
    }
    
    # Print overall results
    print(f"\n{'='*60}")
    print("OVERALL K-FOLD CROSS VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {overall_stats['mean_test_accuracy']:.4f} ± {overall_stats['std_test_accuracy']:.4f}")
    print(f"Test Precision: {overall_stats['mean_test_precision']:.4f} ± {overall_stats['std_test_precision']:.4f}")
    print(f"Test Recall: {overall_stats['mean_test_recall']:.4f} ± {overall_stats['std_test_recall']:.4f}")
    print(f"Test F1-Score: {overall_stats['mean_test_f1']:.4f} ± {overall_stats['std_test_f1']:.4f}")
    
    # Log overall results to wandb
    if args.use_wandb:
        wandb.log(overall_stats)
    
    # Save results to CSV
    results_df = pd.DataFrame(fold_results)
    csv_path = os.path.join(args.results_dir, f'kfold_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Save overall statistics
    overall_stats['timestamp'] = datetime.now().isoformat()
    overall_stats['args'] = vars(args)
    
    stats_path = os.path.join(args.results_dir, f'overall_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(stats_path, 'w') as f:
        json.dump(overall_stats, f, indent=2)
    print(f"Overall statistics saved to {stats_path}")
    
    # Combine training logs if not using wandb
    if not (args.use_wandb and WANDB_AVAILABLE):
        _create_combined_training_logs(args.results_dir, args.n_folds)
    
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    return fold_results, overall_stats

def _create_combined_training_logs(results_dir, n_folds):
    """Combine all fold training logs into a single CSV file"""
    combined_data = []
    
    for fold_idx in range(1, n_folds + 1):
        csv_path = os.path.join(results_dir, f'training_log_fold_{fold_idx}.csv')
        if os.path.exists(csv_path):
            fold_df = pd.read_csv(csv_path)
            fold_df['fold'] = fold_idx
            combined_data.append(fold_df)
    
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        # Reorder columns to put fold first
        cols = ['fold'] + [col for col in combined_df.columns if col != 'fold']
        combined_df = combined_df[cols]
        
        combined_path = os.path.join(results_dir, f'combined_training_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        combined_df.to_csv(combined_path, index=False)
        print(f"Combined training logs saved to: {combined_path}")
        
        # Create summary statistics
        summary_stats = combined_df.groupby('fold').agg({
            'train_loss': ['min', 'max', 'mean'],
            'train_accuracy': ['min', 'max', 'mean'], 
            'val_loss': ['min', 'max', 'mean'],
            'val_accuracy': ['min', 'max', 'mean']
        }).round(6)
        
        summary_path = os.path.join(results_dir, f'training_summary_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        summary_stats.to_csv(summary_path)
        print(f"Training summary statistics saved to: {summary_path}")
    else:
        print("Warning: No training log files found to combine")

if __name__ == "__main__":
    main()
