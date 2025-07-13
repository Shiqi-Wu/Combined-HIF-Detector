#!/usr/bin/env python3
"""
LSTM Evaluation Script

This script provides comprehensive evaluation of trained LSTM models including:
1. Data loading with same preprocessing as training
2. Training history visualization (loss and accuracy curves)
3. Model evaluation on train/val/test sets with boxplot visualization
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
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fault_lstm_classifier import LSTMClassifier
from utils.dataloader import load_full_dataset, create_kfold_dataloaders, ScaledDataset

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LSTMEvaluator:
    """Comprehensive LSTM model evaluator"""
    
    def __init__(self, results_dir: str, data_dir: str, preprocessing_params_path: str):
        """
        Initialize the evaluator
        
        Args:
            results_dir: Directory containing fold results
            data_dir: Directory containing original data
            preprocessing_params_path: Path to preprocessing parameters pickle file
        """
        self.results_dir = Path(results_dir)
        self.data_dir = data_dir
        self.preprocessing_params_path = preprocessing_params_path
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Data containers
        self.datasets = {}
        self.training_histories = {}
        self.evaluation_results = {}
        
    def load_preprocessing_params(self) -> Dict:
        """Load preprocessing parameters from pickle file"""
        print(f"Loading preprocessing parameters from {self.preprocessing_params_path}")
        
        if not os.path.exists(self.preprocessing_params_path):
            raise FileNotFoundError(f"Preprocessing parameters file not found: {self.preprocessing_params_path}")
        
        with open(self.preprocessing_params_path, 'rb') as f:
            params = pickle.load(f)
        
        print("Preprocessing parameters loaded successfully")
        return params
    
    def load_data(self, data_config: Dict, k_folds: int = 5, random_state: int = 42) -> None:
        """
        Load and preprocess data same as training
        
        Args:
            data_config: Data configuration dictionary
            k_folds: Number of folds
            random_state: Random seed
        """
        print("\n===== Loading Dataset =====")
        
        # Load full dataset
        dataset, file_labels = load_full_dataset(self.data_dir, data_config)
        print(f"Loaded dataset with {len(dataset)} samples from {len(file_labels)} files")
        
        # Create K-fold splits
        fold_dataloaders, test_loader = create_kfold_dataloaders(
            dataset, file_labels, data_config, k_folds, random_state
        )
        
        # Load preprocessing parameters
        preprocessing_params = self.load_preprocessing_params()
        
        # Apply preprocessing to test set (shared across all folds)
        test_scaled = ScaledDataset(test_loader.dataset, pca_dim=2, fit_scalers=False)
        test_scaled.set_preprocessing_params(preprocessing_params)
        test_loader_scaled = DataLoader(test_scaled, batch_size=data_config.get('batch_size', 32), shuffle=False, num_workers=0)
        
        # Store test dataset
        self.datasets['test'] = test_loader_scaled.dataset
        
        # Apply preprocessing to each fold
        self.datasets['folds'] = []
        for fold_idx, (train_loader, val_loader) in enumerate(fold_dataloaders):
            print(f"Applying preprocessing to fold {fold_idx + 1}/{k_folds}")
            
            # Apply preprocessing
            train_scaled = ScaledDataset(train_loader.dataset, pca_dim=2, fit_scalers=False)
            train_scaled.set_preprocessing_params(preprocessing_params)
            val_scaled = ScaledDataset(val_loader.dataset, pca_dim=2, fit_scalers=False)
            val_scaled.set_preprocessing_params(preprocessing_params)
            
            train_loader_scaled = DataLoader(train_scaled, batch_size=data_config.get('batch_size', 32), shuffle=False, num_workers=0)
            val_loader_scaled = DataLoader(val_scaled, batch_size=data_config.get('batch_size', 32), shuffle=False, num_workers=0)
            
            self.datasets['folds'].append({
                'train': train_loader_scaled.dataset,
                'val': val_loader_scaled.dataset,
                'fold_idx': fold_idx,
            })
        
        print(f"Data loading completed for {len(self.datasets['folds'])} folds")
    
    def load_training_histories(self) -> None:
        """Load training history JSON files from each fold"""
        print("\n===== Loading Training Histories =====")
        
        for fold_dir in sorted(self.results_dir.glob("fold_*")):
            fold_num = int(fold_dir.name.split("_")[1])
            history_file = fold_dir / f"training_history_fold_{fold_num}.json"
            
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
                self.training_histories[fold_num] = history
                print(f"Loaded training history for fold {fold_num}")
            else:
                print(f"Warning: Training history not found for fold {fold_num}")
        
        print(f"Loaded training histories for {len(self.training_histories)} folds")
    
    def get_available_model_paths(self) -> List[Tuple[int, Path]]:
        """
        Get list of available model paths for each fold
        
        Returns:
            List of tuples (fold_num, model_path)
        """
        model_paths = []
        
        for fold_dir in sorted(self.results_dir.glob("fold_*")):
            fold_num = int(fold_dir.name.split("_")[1])
            model_file = fold_dir / f"best_model_fold_{fold_num}.pth"
            
            if model_file.exists():
                model_paths.append((fold_num, model_file))
            else:
                print(f"Warning: Model file not found for fold {fold_num}")
        
        return model_paths
    
    def load_single_model(self, model_path: Path, model_config: Dict) -> Tuple[nn.Module, Dict]:
        """
        Load a single model from checkpoint
        
        Args:
            model_path: Path to model checkpoint
            model_config: Model configuration dictionary
            
        Returns:
            Tuple of (model, checkpoint_info)
        """
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create model
        model = LSTMClassifier(**model_config).to(torch.float64)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        checkpoint_info = {
            'epoch': checkpoint['epoch'],
            'val_accuracy': checkpoint['val_accuracy']
        }
        
        return model, checkpoint_info
    
    def evaluate_model_on_dataset(self, model: nn.Module, dataset, batch_size: int = 32) -> Dict[str, float]:
        """
        Evaluate a single model on a dataset
        
        Args:
            model: Trained model
            dataset: Dataset to evaluate on
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with loss and accuracy
        """
        model.eval()
        criterion = nn.CrossEntropyLoss()
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                x_batch, u_batch, p_batch = batch
                x_batch = x_batch.to(self.device).to(torch.float64)
                u_batch = u_batch.to(self.device).to(torch.float64)
                
                # Convert one-hot to class indices if needed
                if p_batch.dim() > 1 and p_batch.size(1) > 1:
                    p_indices = torch.argmax(p_batch, dim=1)
                else:
                    p_indices = p_batch.long()
                p_indices = p_indices.to(self.device)
                
                outputs = model(x_batch)
                loss = criterion(outputs, p_indices)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += p_indices.size(0)
                total_correct += (predicted == p_indices).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'total_samples': total_samples
        }
    
    
    def evaluate_all_models(self, model_config: Dict, batch_size: int = 32, save_csv: Optional[str] = None) -> None:
        """
        Evaluate all models sequentially (one at a time to save memory)
        
        Args:
            model_config: Model configuration dictionary
            batch_size: Batch size for evaluation
            save_csv: Path to save results CSV file (optional)
        """
        print("\n===== Evaluating All Models (Sequential Loading) =====")
        
        # Get available model paths
        model_paths = self.get_available_model_paths()
        
        if not model_paths:
            print("No model files found!")
            return
        
        self.evaluation_results = {}
        
        for fold_num, model_path in model_paths:
            print(f"Loading and evaluating fold {fold_num}...")
            
            # Load model for this fold only
            model, checkpoint_info = self.load_single_model(model_path, model_config)
            print(f"  Loaded model (epoch {checkpoint_info['epoch']}, val_acc: {checkpoint_info['val_accuracy']:.4f})")
            
            # Get fold data
            fold_data = self.datasets['folds'][fold_num - 1]  # fold_num is 1-indexed
            
            # Evaluate on train, val, test
            results = {}
            
            # Train set
            results['train'] = self.evaluate_model_on_dataset(model, fold_data['train'], batch_size)
            print(f"  Train - Loss: {results['train']['loss']:.4f}, Acc: {results['train']['accuracy']:.4f}")
            
            # Validation set
            results['val'] = self.evaluate_model_on_dataset(model, fold_data['val'], batch_size)
            print(f"  Val   - Loss: {results['val']['loss']:.4f}, Acc: {results['val']['accuracy']:.4f}")
            
            # Test set
            results['test'] = self.evaluate_model_on_dataset(model, self.datasets['test'], batch_size)
            print(f"  Test  - Loss: {results['test']['loss']:.4f}, Acc: {results['test']['accuracy']:.4f}")
            
            # Store results
            self.evaluation_results[fold_num] = results
            
            # Clear model from memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print(f"  Model for fold {fold_num} evaluated and cleared from memory")

        # Save results to CSV if requested
        if save_csv:
            self._save_evaluation_to_csv(save_csv)

        print("Model evaluation completed")
    
    def plot_training_histories(self, save_dir: Optional[str] = None) -> None:
        """
        Plot training loss and accuracy curves for all folds
        
        Args:
            save_dir: Directory to save plots (optional)
        """
        print("\n===== Plotting Training Histories =====")
        
        if not self.training_histories:
            print("No training histories loaded")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training History Across All Folds', fontsize=16)
        
        # Prepare data for plotting
        all_train_loss = []
        all_val_loss = []
        all_train_acc = []
        all_val_acc = []
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.training_histories)))
        
        for i, (fold_num, history) in enumerate(self.training_histories.items()):
            epochs = range(1, len(history['train_loss']) + 1)
            color = colors[i]
            
            # Training and validation loss
            axes[0, 0].plot(epochs, history['train_loss'], 
                           label=f'Fold {fold_num}', color=color, alpha=0.7)
            axes[0, 1].plot(epochs, history['val_loss'], 
                           label=f'Fold {fold_num}', color=color, alpha=0.7)
            
            # Training and validation accuracy
            axes[1, 0].plot(epochs, history['train_acc'], 
                           label=f'Fold {fold_num}', color=color, alpha=0.7)
            axes[1, 1].plot(epochs, history['val_acc'], 
                           label=f'Fold {fold_num}', color=color, alpha=0.7)
            
            # Collect data for summary statistics
            all_train_loss.extend(history['train_loss'])
            all_val_loss.extend(history['val_loss'])
            all_train_acc.extend(history['train_acc'])
            all_val_acc.extend(history['val_acc'])
        
        # Configure subplots
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Training Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Validation Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_dir:
            save_path = Path(save_dir) / 'training_history.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
        
        # Print summary statistics
        print("\nTraining History Summary:")
        print(f"  Training Loss   - Mean: {np.mean(all_train_loss):.4f}, Std: {np.std(all_train_loss):.4f}")
        print(f"  Validation Loss - Mean: {np.mean(all_val_loss):.4f}, Std: {np.std(all_val_loss):.4f}")
        print(f"  Training Acc    - Mean: {np.mean(all_train_acc):.4f}, Std: {np.std(all_train_acc):.4f}")
        print(f"  Validation Acc  - Mean: {np.mean(all_val_acc):.4f}, Std: {np.std(all_val_acc):.4f}")
    
    def plot_evaluation_boxplots(self, save_dir: Optional[str] = None) -> None:
        """
        Plot boxplots of model evaluation results across folds
        
        Args:
            save_dir: Directory to save plots (optional)
        """
        print("\n===== Plotting Evaluation Boxplots =====")
        
        if not self.evaluation_results:
            print("No evaluation results available")
            return
        
        # Prepare data for boxplots
        data_for_plotting = []
        
        for fold_num, results in self.evaluation_results.items():
            for dataset_name in ['train', 'val', 'test']:
                data_for_plotting.append({
                    'Fold': fold_num,
                    'Dataset': dataset_name.capitalize(),
                    'Loss': results[dataset_name]['loss'],
                    'Accuracy': results[dataset_name]['accuracy']
                })
        
        df = pd.DataFrame(data_for_plotting)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Model Evaluation Results Across All Folds', fontsize=16)
        
        # Loss boxplot
        sns.boxplot(data=df, x='Dataset', y='Loss', ax=axes[0])
        axes[0].set_title('Loss Distribution')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        
        # Add individual points
        sns.stripplot(data=df, x='Dataset', y='Loss', ax=axes[0], 
                     color='red', alpha=0.7, size=8)
        
        # Accuracy boxplot
        sns.boxplot(data=df, x='Dataset', y='Accuracy', ax=axes[1])
        axes[1].set_title('Accuracy Distribution')
        axes[1].set_ylabel('Accuracy')
        axes[1].grid(True, alpha=0.3)
        
        # Add individual points
        sns.stripplot(data=df, x='Dataset', y='Accuracy', ax=axes[1], 
                     color='red', alpha=0.7, size=8)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_dir:
            save_path = Path(save_dir) / 'evaluation_boxplots.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Evaluation boxplots saved to {save_path}")
        
        plt.show()
        
        # Print summary statistics
        print("\nEvaluation Results Summary:")
        summary_stats = df.groupby('Dataset').agg({
            'Loss': ['mean', 'std', 'min', 'max'],
            'Accuracy': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        print(summary_stats)
        
        # Save summary to CSV
        if save_dir:
            summary_path = Path(save_dir) / 'evaluation_summary.csv'
            summary_stats.to_csv(summary_path)
            print(f"Evaluation summary saved to {summary_path}")
        
        # Create detailed results table
        detailed_results = df.pivot(index='Fold', columns='Dataset', values=['Loss', 'Accuracy'])
        print("\nDetailed Results by Fold:")
        print(detailed_results.round(4))
        
        if save_dir:
            detailed_path = Path(save_dir) / 'detailed_results.csv'
            detailed_results.to_csv(detailed_path)
            print(f"Detailed results saved to {detailed_path}")
    
    def save_evaluation_results(self, save_dir: str) -> None:
        """
        Save evaluation results to files
        
        Args:
            save_dir: Directory to save results
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save evaluation results as JSON
        results_file = save_path / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        print(f"Evaluation results saved to {results_file}")
    
    def _save_evaluation_to_csv(self, csv_path: str) -> None:
        """
        Save evaluation results to CSV file
        
        Args:
            csv_path: Path to save CSV file
        """
        print(f"\n===== Saving Evaluation Results to CSV =====")
        
        if not self.evaluation_results:
            print("No evaluation results to save")
            return
        
        # Prepare data for CSV
        csv_data = []
        
        for fold_num, results in self.evaluation_results.items():
            for dataset_name in ['train', 'val', 'test']:
                csv_data.append({
                    'fold': fold_num,
                    'dataset': dataset_name,
                    'loss': results[dataset_name]['loss'],
                    'accuracy': results[dataset_name]['accuracy'],
                })
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(csv_data)
        
        # Create directory if it doesn't exist
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"Evaluation results saved to {csv_path}")
        
        # Print summary
        print("\nCSV Summary:")
        print(f"  Total rows: {len(df)}")
        print(f"  Folds: {sorted(df['fold'].unique())}")
        print(f"  Datasets: {sorted(df['dataset'].unique())}")
        
        # Show first few rows
        print("\nFirst few rows:")
        print(df.head().to_string(index=False))
    
    def run_full_evaluation(self, model_config: Dict, data_config: Dict, 
                           k_folds: int = 5, random_state: int = 42, 
                           save_dir: Optional[str] = None) -> None:
        """
        Run complete evaluation pipeline
        
        Args:
            model_config: Model configuration
            data_config: Data configuration
            k_folds: Number of folds
            random_state: Random seed
            save_dir: Directory to save results and plots
        """
        print("="*60)
        print("STARTING COMPREHENSIVE LSTM EVALUATION")
        print("="*60)
        
        # Create save directory
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Step 1: Load data
        self.load_data(data_config, k_folds, random_state)
        
        # Step 2: Load training histories
        self.load_training_histories()
        
        # Step 3: Evaluate models (sequential loading)
        csv_path = None
        if save_dir:
            csv_path = str(Path(save_dir) / 'fold_evaluation_results.csv')
        self.evaluate_all_models(model_config, data_config.get('batch_size', 32), csv_path)
        
        # Step 4: Plot training histories
        self.plot_training_histories(save_dir)
        
        # Step 5: Plot evaluation boxplots
        self.plot_evaluation_boxplots(save_dir)
        
        # Step 6: Save results
        if save_dir:
            self.save_evaluation_results(save_dir)
        
        print("="*60)
        print("EVALUATION COMPLETED")
        print("="*60)


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='LSTM Model Evaluation')
    
    # Required arguments
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing fold results')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing original data')
    
    # Optional arguments
    parser.add_argument('--preprocessing_params', type=str, 
                       default='/home/shiqi_w/code/Combined-HIF-detector/preprocessing_params_fold.pkl',
                       help='Path to preprocessing parameters pickle file')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save evaluation results and plots')
    parser.add_argument('--csv_file', type=str, default=None,
                       help='Path to save evaluation results CSV file')
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
                       help='Batch size for evaluation')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--bidirectional', action='store_true', default=True,
                       help='Use bidirectional LSTM')
    
    args = parser.parse_args()
    
    # Configuration
    data_config = {
        'window_size': args.window_size,
        'sample_step': args.sample_step,
        'batch_size': args.batch_size
    }
    
    model_config = {
        'input_size': 2,  # After PCA
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'num_classes': 6,  # ErrorType: 2~7
        'dropout': args.dropout,
        'bidirectional': args.bidirectional
    }
    
    # Create evaluator
    evaluator = LSTMEvaluator(
        results_dir=args.results_dir,
        data_dir=args.data_dir,
        preprocessing_params_path=args.preprocessing_params
    )
    
    # Run evaluation
    save_dir = args.save_dir
    if args.csv_file and not save_dir:
        # If CSV file is specified but no save_dir, use CSV file's directory as save_dir
        save_dir = str(Path(args.csv_file).parent)
    
    evaluator.run_full_evaluation(
        model_config=model_config,
        data_config=data_config,
        k_folds=args.k_folds,
        random_state=args.random_state,
        save_dir=save_dir
    )
    
    # Save to specific CSV file if requested
    if args.csv_file:
        evaluator._save_evaluation_to_csv(args.csv_file)


if __name__ == "__main__":
    main()
