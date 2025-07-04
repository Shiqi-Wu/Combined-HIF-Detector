import numpy as np
import os
from sklearn.model_selection import train_test_split, KFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader, Subset
import torch


def integer_to_one_hot(integer, min_val, max_val):
    """Convert integer labels to one-hot encoding"""
    vector_length = max_val - min_val + 1
    one_hot_vector = [0] * vector_length
    one_hot_vector[integer - min_val] = 1
    return one_hot_vector

def non_overlapping_window_split(sequence, window_size=30):
    """Split time series data using non-overlapping windows"""
    slices = []
    for start in range(0, len(sequence), window_size):
        end = start + window_size
        if end <= len(sequence):  # Ensure complete windows only
            slices.append(sequence[start:end])
    return slices

def load_dataset_from_folder(data_dir, config, test_size=0.2, random_state=42):
    """
    Load dataset from folder and split into training and test sets
    
    Args:
        data_dir: Data folder path
        config: Configuration dictionary containing sample_step and other parameters
        test_size: Test set ratio
        random_state: Random seed
    
    Returns:
        train_loader, test_loader: Training and testing DataLoaders
    """
    x_trajectories = []
    u_trajectories = []
    p_labels = []
    min_val = 2
    max_val = 7
    
    print(f"Loading data from folder {data_dir}...")
    
    # Iterate through all npy files in the folder
    for item in os.listdir(data_dir):
        data_file_path = os.path.join(data_dir, item)
        
        if data_file_path.endswith('.npy') and os.path.exists(data_file_path):
            try:
                data_dict = np.load(data_file_path, allow_pickle=True).item()
                
                # Extract signal data
                x_data = data_dict['signals'][:, :-6]  # State signals
                u_data = data_dict['signals'][:, -6:-4]  # Control signals
                
                # Data sampling
                x_data = x_data[::config['sample_step'], :]
                u_data = u_data[::config['sample_step'], :]
                
                # Extract error type and convert to one-hot encoding
                error_type = data_dict['ErrorType']
                p_data = np.array(integer_to_one_hot(error_type, min_val, max_val))
                
                x_trajectories.append(x_data)
                u_trajectories.append(u_data)
                p_labels.append(p_data)
                
                print(f"Loaded: {item}, data shape: {x_data.shape}, error type: {error_type}")
                
            except Exception as e:
                print(f"Error loading file {item}: {e}")
    
    print(f"Total loaded {len(x_trajectories)} data files")
    
    # Split training and test sets
    if len(x_trajectories) == 0:
        raise ValueError("No valid data files found")
    
    # Split data into training and test sets
    x_train, x_test, u_train, u_test, p_train, p_test = train_test_split(
        x_trajectories, u_trajectories, p_labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=[np.argmax(p) for p in p_labels]  # Stratified sampling by error type
    )
    
    print(f"Training set: {len(x_train)} files, Test set: {len(x_test)} files")
    
    # Create datasets
    window_size = config.get('window_size', 30)
    batch_size = config.get('batch_size', 32)
    
    train_dataset = NonOverlappingTrajectoryDataset(x_train, u_train, p_train, window_size)
    test_dataset = NonOverlappingTrajectoryDataset(x_test, u_test, p_test, window_size)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader

class NonOverlappingTrajectoryDataset(Dataset):
    """Trajectory dataset using non-overlapping windows"""
    
    def __init__(self, x_trajectories, u_trajectories, p_labels, window_size):
        self.x_samples = []
        self.u_samples = []
        self.p_labels = []
        
        for x_traj, u_traj, p_label in zip(x_trajectories, u_trajectories, p_labels):
            # Use non-overlapping window splitting
            x_slices = non_overlapping_window_split(x_traj, window_size)
            u_slices = non_overlapping_window_split(u_traj, window_size)
            
            # Ensure x and u have the same number of slices
            min_slices = min(len(x_slices), len(u_slices))
            
            self.x_samples.extend(x_slices[:min_slices])
            self.u_samples.extend(u_slices[:min_slices])
            self.p_labels.extend([p_label] * min_slices)
    
    def __len__(self):
        return len(self.x_samples)
    
    def __getitem__(self, idx):
        x_sample = torch.FloatTensor(self.x_samples[idx])
        u_sample = torch.FloatTensor(self.u_samples[idx])
        p_label = torch.FloatTensor(self.p_labels[idx])
        
        return x_sample, u_sample, p_label

def load_full_dataset(data_dir, config):
    """
    Load complete dataset for k-fold cross validation
    
    Args:
        data_dir: Data folder path
        config: Configuration dictionary
    
    Returns:
        dataset: Complete dataset for k-fold splitting
        file_labels: Labels for stratified splitting
    """
    x_trajectories = []
    u_trajectories = []
    p_labels = []
    file_labels = []  # For stratified k-fold
    min_val = 2
    max_val = 7
    
    print(f"Loading complete dataset from folder {data_dir}...")
    
    # Iterate through all npy files in the folder
    for item in os.listdir(data_dir):
        data_file_path = os.path.join(data_dir, item)
        
        if data_file_path.endswith('.npy') and os.path.exists(data_file_path):
            try:
                data_dict = np.load(data_file_path, allow_pickle=True).item()
                
                # Extract signal data
                x_data = data_dict['signals'][:, :-6]  # State signals
                u_data = data_dict['signals'][:, -6:-4]  # Control signals
                # print(x_data.shape, u_data.shape)
                x_data = x_data[100:]  # Skip first 1000 samples
                u_data = u_data[100:]  # Skip first 1000 samples
                
                # Data sampling
                x_data = x_data[::config['sample_step'], :]
                u_data = u_data[::config['sample_step'], :]
                
                # Extract error type and convert to one-hot encoding
                error_type = data_dict['ErrorType']
                p_data = np.array(integer_to_one_hot(error_type, min_val, max_val))
                
                x_trajectories.append(x_data)
                u_trajectories.append(u_data)
                p_labels.append(p_data)
                file_labels.append(error_type)  # For stratified splitting
                
                print(f"Loaded: {item}, data shape: {x_data.shape}, error type: {error_type}")
                
            except Exception as e:
                print(f"Error loading file {item}: {e}")
    
    print(f"Total loaded {len(x_trajectories)} data files")
    
    if len(x_trajectories) == 0:
        raise ValueError("No valid data files found")
    
    # Create complete dataset
    window_size = config.get('window_size', 30)
    complete_dataset = NonOverlappingTrajectoryDataset(x_trajectories, u_trajectories, p_labels, window_size)
    
    return complete_dataset, file_labels

def create_kfold_dataloaders(dataset, file_labels, config, n_splits=5, random_state=42):
    """
    Create k-fold cross validation dataloaders
    
    Args:
        dataset: Complete dataset
        file_labels: Labels for stratified splitting
        config: Configuration dictionary
        n_splits: Number of folds
        random_state: Random seed
    
    Returns:
        fold_dataloaders: List of (train_loader, val_loader, test_loader) for each fold
    """
    
    # Create file-level indices
    file_indices = list(range(len(file_labels)))
    
    # Split into train+val (80%) and test (20%) at file level
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_val_files, test_files = next(sss_test.split(file_indices, file_labels))
    
    # Create mapping from file labels to sample indices
    file_to_sample_indices = {}
    for file_idx, file_label in enumerate(file_labels):
        file_to_sample_indices[file_idx] = []
        target_one_hot = np.array(integer_to_one_hot(file_label, 2, 7))
        
        for sample_idx in range(len(dataset)):
            if np.array_equal(dataset.p_labels[sample_idx], target_one_hot):
                file_to_sample_indices[file_idx].append(sample_idx)
    
    # Get test indices
    test_indices = []
    for file_idx in test_files:
        test_indices.extend(file_to_sample_indices[file_idx])
    
    # Create test dataset
    test_dataset = Subset(dataset, test_indices)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        num_workers=0
    )
    
    # Get train+val indices
    train_val_indices = []
    for file_idx in train_val_files:
        train_val_indices.extend(file_to_sample_indices[file_idx])
    
    # Create k-fold splits on train+val data
    train_val_labels = [file_labels[i] for i in train_val_files]
    kfold = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=random_state)  # 25% of 80% = 20% val
    
    fold_dataloaders = []
    
    for fold_idx, (train_file_idx, val_file_idx) in enumerate(kfold.split(train_val_files, train_val_labels)):
        print(f"Creating fold {fold_idx + 1}/{n_splits}")
        
        # Get actual file indices
        train_files_fold = [train_val_files[i] for i in train_file_idx]
        val_files_fold = [train_val_files[i] for i in val_file_idx]
        
        # Get sample indices for this fold
        train_indices_fold = []
        val_indices_fold = []
        
        # Get indices for train and val sets
        for file_idx in train_files_fold:
            train_indices_fold.extend(file_to_sample_indices[file_idx])
        
        for file_idx in val_files_fold:
            val_indices_fold.extend(file_to_sample_indices[file_idx])
        
        # Create datasets for this fold
        train_dataset_fold = Subset(dataset, train_indices_fold)
        val_dataset_fold = Subset(dataset, val_indices_fold)
        
        # Create dataloaders
        train_loader_fold = DataLoader(
            train_dataset_fold,
            batch_size=config.get('batch_size', 32),
            shuffle=True,
            num_workers=0
        )
        
        val_loader_fold = DataLoader(
            val_dataset_fold,
            batch_size=config.get('batch_size', 32),
            shuffle=False,
            num_workers=0
        )
        
        fold_dataloaders.append((train_loader_fold, val_loader_fold, test_loader))
        
        print(f"  Fold {fold_idx + 1}: Train samples: {len(train_indices_fold)}, "
              f"Val samples: {len(val_indices_fold)}, Test samples: {len(test_indices)}")
    
    return fold_dataloaders


class ScaledDataset(Dataset):
    """
    Dataset wrapper that applies scaling to x (state) and u (control) data
    """
    
    def __init__(self, base_dataset, scaler_x=None, scaler_u=None, fit_scalers=False):
        """
        Initialize scaled dataset
        
        Args:
            base_dataset: Original dataset
            scaler_x: Scaler for x data (state signals)
            scaler_u: Scaler for u data (control signals)  
            fit_scalers: Whether to fit the scalers on this dataset
        """
        self.base_dataset = base_dataset
        # self.p_labels = base_dataset.p_labels
        self.scaler_x = scaler_x or StandardScaler()
        self.scaler_u = scaler_u or StandardScaler()
        
        if fit_scalers:
            self._fit_scalers()
    
    def _fit_scalers(self):
        """Fit scalers on the entire dataset"""
        print("Fitting scalers on dataset...")
        
        # Collect all x and u data for fitting
        all_x_data = []
        all_u_data = []
        
        for i in range(len(self.base_dataset)):
            x_batch, u_batch, p_batch = self.base_dataset[i]
            
            # Reshape for scaler (samples, features)
            x_reshaped = x_batch.view(-1, x_batch.shape[-1])  # (seq_len, features)
            u_reshaped = u_batch.view(-1, u_batch.shape[-1])  # (seq_len, features)
            
            all_x_data.append(x_reshaped.numpy())
            all_u_data.append(u_reshaped.numpy())
        
        # Concatenate all data
        all_x = np.concatenate(all_x_data, axis=0)
        all_u = np.concatenate(all_u_data, axis=0)
        
        # Fit scalers
        self.scaler_x.fit(all_x)
        self.scaler_u.fit(all_u)
        
        print(f"X data - Mean: {self.scaler_x.mean_[:5]}, Std: {self.scaler_x.scale_[:5]}")
        print(f"U data - Mean: {self.scaler_u.mean_[:5]}, Std: {self.scaler_u.scale_[:5]}")
        print("Scalers fitted successfully!")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        x_batch, u_batch, p_batch = self.base_dataset[idx]
        
        # Scale x data (state signals)
        x_shape = x_batch.shape
        x_reshaped = x_batch.view(-1, x_shape[-1])  # (seq_len, features)
        x_scaled = self.scaler_x.transform(x_reshaped.numpy())
        x_scaled = torch.from_numpy(x_scaled).view(x_shape).float()
        
        # Scale u data (control signals)
        u_shape = u_batch.shape
        u_reshaped = u_batch.view(-1, u_shape[-1])  # (seq_len, features)
        u_scaled = self.scaler_u.transform(u_reshaped.numpy())
        u_scaled = torch.from_numpy(u_scaled).view(u_shape).float()
        
        return x_scaled, u_scaled, p_batch
    
    def get_scalers(self):
        """Return the fitted scalers"""
        return self.scaler_x, self.scaler_u
    
    @property
    def p_labels(self):
        # Delegates to base dataset (assuming it has p_labels)
        return self.base_dataset.p_labels if hasattr(self.base_dataset, "p_labels") else None


# # Usage example
# if __name__ == "__main__":
#     # Configuration parameters
#     config = {
#         'sample_step': 1,      # Sampling step
#         'window_size': 30,     # Window size
#         'batch_size': 32       # Batch size
#     }
    
#     # Data folder path
#     data_dir = "/Users/shiqi/Documents/PhD/Code/Project3-power-grid/Combined-HIF-Detector/data"
    
#     try:
#         # Create training and testing DataLoaders
#         train_loader, test_loader = load_dataset_from_folder(
#             data_dir=data_dir,
#             config=config,
#             test_size=0.2,  # 20% as test set
#             random_state=42
#         )
        
#         # Test DataLoaders
#         print("\n=== Testing DataLoaders ===")
#         for batch_idx, (x_batch, u_batch, p_batch) in enumerate(train_loader):
#             print(f"Batch {batch_idx + 1}:")
#             print(f"  x_batch shape: {x_batch.shape}")
#             print(f"  u_batch shape: {u_batch.shape}")
#             print(f"  p_batch shape: {p_batch.shape}")
            
#             if batch_idx >= 2:  # Show only first 3 batches
#                 break
                
#         print(f"\nTotal training batches: {len(train_loader)}")
#         print(f"Total test batches: {len(test_loader)}")
        
#         # Load full dataset for k-fold cross validation
#         complete_dataset, file_labels = load_full_dataset(data_dir=data_dir, config=config)
        
#         # Create k-fold dataloaders
#         n_splits = 5
#         fold_dataloaders = create_kfold_dataloaders(dataset=complete_dataset, file_labels=file_labels, config=config, n_splits=n_splits)
        
#         # Test k-fold dataloaders
#         print("\n=== Testing k-fold DataLoaders ===")
#         for fold_idx, (train_loader_fold, val_loader_fold, test_loader) in enumerate(fold_dataloaders):
#             print(f"Fold {fold_idx + 1}:")
#             print(f"  Train batches: {len(train_loader_fold)}")
#             print(f"  Val batches: {len(val_loader_fold)}")
#             print(f"  Test batches: {len(test_loader)}")
            
#             # Test one batch from train and val loaders
#             for batch_idx, (x_batch, u_batch, p_batch) in enumerate(train_loader_fold):
#                 print(f"  Train Batch {batch_idx + 1}: x_batch shape: {x_batch.shape}, u_batch shape: {u_batch.shape}, p_batch shape: {p_batch.shape}")
#                 if batch_idx >= 1:  # Show only first 2 batches
#                     break
            
#             for batch_idx, (x_batch, u_batch, p_batch) in enumerate(val_loader_fold):
#                 print(f"  Val Batch {batch_idx + 1}: x_batch shape: {x_batch.shape}, u_batch shape: {u_batch.shape}, p_batch shape: {p_batch.shape}")
#                 if batch_idx >= 1:  # Show only first 2 batches
#                     break
    
#     except Exception as e:
#         print(f"Error: {e}")
