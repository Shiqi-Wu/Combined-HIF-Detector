#!/usr/bin/env python3
"""
Test script to validate data loader integrity
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.dataloader import load_full_dataset, create_kfold_dataloaders, validate_data_split_integrity

def test_dataloader_integrity():
    """Test that the dataloader maintains file-level separation"""
    
    # Configuration
    config = {
        'window_size': 3317,
        'sample_step': 1,
        'batch_size': 32
    }
    
    data_dir = "data"
    
    print("Testing dataloader integrity...")
    
    try:
        # Load dataset
        dataset, file_labels = load_full_dataset(data_dir, config)
        print(f"Loaded dataset with {len(dataset)} samples from {len(file_labels)} files")
        
        # Create k-fold dataloaders
        fold_dataloaders, test_loader = create_kfold_dataloaders(
            dataset, file_labels, config, n_splits=3, random_state=42
        )
        
        # Validate integrity
        validate_data_split_integrity(dataset, fold_dataloaders, test_loader)
        
        print("✓ All tests passed! Data integrity is maintained.")
        
        # Print some statistics
        print("\nDataset Statistics:")
        print(f"Total files: {len(file_labels)}")
        print(f"Total samples: {len(dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
        for i, (train_loader, val_loader) in enumerate(fold_dataloaders):
            print(f"Fold {i+1}: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataloader_integrity()
