import numpy as np
import os
from sklearn.model_selection import train_test_split, KFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import pickle


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

def load_dataset_from_folder(data_dir, config, test_size=0.2, random_state=42, delete_columns=[9, 21, 25, 39, 63]):
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
                
                # Apply column deletion
                x_data = np.delete(x_data, delete_columns, axis=1)
                
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
    """Trajectory dataset using non-overlapping windows (backward compatibility)"""
    
    def __init__(self, x_trajectories, u_trajectories, p_labels, window_size):
        self.x_samples = []
        self.u_samples = []
        self.p_labels = []
        self.file_indices = []  # Add file tracking for compatibility
        
        for file_idx, (x_traj, u_traj, p_label) in enumerate(zip(x_trajectories, u_trajectories, p_labels)):
            # Use non-overlapping window splitting
            x_slices = non_overlapping_window_split(x_traj, window_size)
            u_slices = non_overlapping_window_split(u_traj, window_size)
            
            # Ensure x and u have the same number of slices
            min_slices = min(len(x_slices), len(u_slices))
            
            self.x_samples.extend(x_slices[:min_slices])
            self.u_samples.extend(u_slices[:min_slices])
            self.p_labels.extend([p_label] * min_slices)
            self.file_indices.extend([file_idx] * min_slices)  # Track file origin
    
    def __len__(self):
        return len(self.x_samples)
    
    def __getitem__(self, idx):
        x_sample = torch.FloatTensor(self.x_samples[idx])
        u_sample = torch.FloatTensor(self.u_samples[idx])
        p_label = torch.FloatTensor(self.p_labels[idx])
        
        return x_sample, u_sample, p_label
    
    def get_samples_by_file(self, file_idx):
        """Get all sample indices that belong to a specific file"""
        return [i for i, f_idx in enumerate(self.file_indices) if f_idx == file_idx]
    
    def get_file_index_for_sample(self, sample_idx):
        """Get the file index for a specific sample"""
        return self.file_indices[sample_idx]


class FileTrackingTrajectoryDataset(Dataset):
    """Trajectory dataset that tracks which file each sample comes from"""
    
    def __init__(self, x_trajectories, u_trajectories, p_labels, file_names, window_size):
        self.x_samples = []
        self.u_samples = []
        self.p_labels = []
        self.file_indices = []  # Track which file each sample comes from
        
        for file_idx, (x_traj, u_traj, p_label, file_name) in enumerate(zip(x_trajectories, u_trajectories, p_labels, file_names)):
            # Use non-overlapping window splitting
            x_slices = non_overlapping_window_split(x_traj, window_size)
            u_slices = non_overlapping_window_split(u_traj, window_size)
            
            # Ensure x and u have the same number of slices
            min_slices = min(len(x_slices), len(u_slices))
            
            self.x_samples.extend(x_slices[:min_slices])
            self.u_samples.extend(u_slices[:min_slices])
            self.p_labels.extend([p_label] * min_slices)
            self.file_indices.extend([file_idx] * min_slices)  # Track file origin
    
    def __len__(self):
        return len(self.x_samples)
    
    def __getitem__(self, idx):
        x_sample = torch.FloatTensor(self.x_samples[idx])
        u_sample = torch.FloatTensor(self.u_samples[idx])
        p_label = torch.FloatTensor(self.p_labels[idx])
        
        return x_sample, u_sample, p_label
    
    def get_samples_by_file(self, file_idx):
        """Get all sample indices that belong to a specific file"""
        return [i for i, f_idx in enumerate(self.file_indices) if f_idx == file_idx]
    
    def get_file_index_for_sample(self, sample_idx):
        """Get the file index for a specific sample"""
        return self.file_indices[sample_idx]

def load_full_dataset(data_dir, config, delete_columns=[9, 21, 25, 39, 63]):
    """
    Load complete dataset for k-fold cross validation
    
    Args:
        data_dir: Data folder path
        config: Configuration dictionary
    
    Returns:
        dataset: Complete dataset for k-fold splitting
        file_labels: Labels for stratified splitting
        file_to_samples_mapping: Mapping from file index to sample indices
    """
    x_trajectories = []
    u_trajectories = []
    p_labels = []
    file_labels = []  # For stratified k-fold
    file_names = []   # Track original file names
    min_val = 2
    max_val = 7
    
    print(f"Loading complete dataset from folder {data_dir}...")
    
    # Sort files for consistent ordering
    file_list = sorted([item for item in os.listdir(data_dir) if item.endswith('.npy')])
    
    # Iterate through all npy files in the folder
    for item in file_list:
        data_file_path = os.path.join(data_dir, item)
        
        if os.path.exists(data_file_path):
            try:
                data_dict = np.load(data_file_path, allow_pickle=True).item()
                
                # Extract signal data
                x_data = data_dict['signals'][:, :-6]  # State signals
                u_data = data_dict['signals'][:, -6:-4]  # Control signals
                
                # Apply column deletion
                x_data = np.delete(x_data, delete_columns, axis=1)
                
                # Skip first 100 samples for stability
                x_data = x_data[100:]  
                u_data = u_data[100:]  
                
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
                file_names.append(item)  # Track file name
                
                print(f"Loaded: {item}, data shape: {x_data.shape}, error type: {error_type}")
                
            except Exception as e:
                print(f"Error loading file {item}: {e}")
    
    print(f"Total loaded {len(x_trajectories)} data files")
    
    if len(x_trajectories) == 0:
        raise ValueError("No valid data files found")
    
    # Create complete dataset with file tracking
    window_size = config.get('window_size', 30)
    complete_dataset = FileTrackingTrajectoryDataset(x_trajectories, u_trajectories, p_labels, file_names, window_size)
    
    return complete_dataset, file_labels

def create_kfold_dataloaders(dataset, file_labels, config, n_splits=5, random_state=42):
    """
    Create k-fold cross validation dataloaders with train/val/test split
    Ensures that each trajectory (file) appears in only one subset (train/val/test)
    
    Args:
        dataset: Complete FileTrackingTrajectoryDataset
        file_labels: Labels for stratified splitting
        config: Configuration dictionary
        n_splits: Number of folds
        random_state: Random seed
    
    Returns:
        fold_dataloaders: List of (train_loader, val_loader) for each fold
        test_loader: Fixed test loader (same for all folds)
    """
    
    # Create file-level indices
    file_indices = list(range(len(file_labels)))
    
    # Split into train+val (80%) and test (20%) at file level
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_val_files, test_files = next(sss_test.split(file_indices, file_labels))
    
    print(f"File-level split: {len(train_val_files)} train+val files, {len(test_files)} test files")
    print(f"Test files: {test_files}")
    print(f"Train+Val files: {train_val_files}")
    
    # Get sample indices for test set (all samples from test files)
    test_indices = []
    for file_idx in test_files:
        sample_indices = dataset.get_samples_by_file(file_idx)
        test_indices.extend(sample_indices)
        print(f"File {file_idx} contributes {len(sample_indices)} samples to test set")
    
    # Create test dataset (shared across all folds)
    test_dataset = Subset(dataset, test_indices)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        num_workers=0
    )
    
    print(f"Test set: {len(test_indices)} samples from {len(test_files)} files")
    
    # Create k-fold splits on train+val files only
    train_val_labels = [file_labels[i] for i in train_val_files]
    kfold = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=random_state)  # 25% of 80% = 20% val
    
    fold_dataloaders = []
    
    for fold_idx, (train_file_idx, val_file_idx) in enumerate(kfold.split(train_val_files, train_val_labels)):
        print(f"Creating fold {fold_idx + 1}/{n_splits}")
        
        # Get actual file indices
        train_files_fold = [train_val_files[i] for i in train_file_idx]
        val_files_fold = [train_val_files[i] for i in val_file_idx]
        
        print(f"  Fold {fold_idx + 1} - Train files: {train_files_fold}")
        print(f"  Fold {fold_idx + 1} - Val files: {val_files_fold}")
        
        # Get sample indices for this fold
        train_indices_fold = []
        val_indices_fold = []
        
        # Get indices for train set (all samples from train files)
        for file_idx in train_files_fold:
            sample_indices = dataset.get_samples_by_file(file_idx)
            train_indices_fold.extend(sample_indices)
        
        # Get indices for val set (all samples from val files)
        for file_idx in val_files_fold:
            sample_indices = dataset.get_samples_by_file(file_idx)
            val_indices_fold.extend(sample_indices)
        
        # Verify no overlap between train, val, and test
        train_files_set = set(train_files_fold)
        val_files_set = set(val_files_fold)
        test_files_set = set(test_files)
        
        assert len(train_files_set & val_files_set) == 0, f"Overlap between train and val files in fold {fold_idx + 1}"
        assert len(train_files_set & test_files_set) == 0, f"Overlap between train and test files in fold {fold_idx + 1}"
        assert len(val_files_set & test_files_set) == 0, f"Overlap between val and test files in fold {fold_idx + 1}"
        
        print(f"  Fold {fold_idx + 1} - No file overlap verified ✓")
        
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
        
        fold_dataloaders.append((train_loader_fold, val_loader_fold))
        
        print(f"  Fold {fold_idx + 1}: Train samples: {len(train_indices_fold)} from {len(train_files_fold)} files, "
              f"Val samples: {len(val_indices_fold)} from {len(val_files_fold)} files")
    
    print(f"All folds created. Test samples: {len(test_indices)} from {len(test_files)} files (shared across all folds)")
    
    return fold_dataloaders, test_loader


def create_preprocessed_kfold_dataloaders(dataset, file_labels, config, n_splits=5, random_state=42, pca_dim=2):
    """
    Create k-fold cross validation dataloaders with shared preprocessing (scaling + PCA) parameters.
    Ensures that each trajectory (file) appears in only one subset (train/val/test).

    Args:
        dataset: Complete FileTrackingTrajectoryDataset
        file_labels: Labels for stratified splitting
        config: Configuration dictionary
        n_splits: Number of folds
        random_state: Random seed
        pca_dim: Number of PCA components

    Returns:
        fold_dataloaders: List of (train_loader, val_loader) for each fold
        test_loader: Fixed test loader (same for all folds)
        preprocessing_params: Shared preprocessing parameters used for all folds
    """
    from torch.utils.data import DataLoader

    # First get the basic fold dataloaders
    fold_dataloaders, test_loader = create_kfold_dataloaders(dataset, file_labels, config, n_splits, random_state)

    # Validate data split integrity
    validate_data_split_integrity(dataset, fold_dataloaders, test_loader)

    print("Fitting shared preprocessing parameters on full dataset...")
    scaler_dataset = ScaledDataset(dataset, pca_dim=pca_dim, fit_scalers=True)
    shared_params = scaler_dataset.get_preprocessing_params()

    preprocessed_fold_dataloaders = []

    for fold_idx, (train_loader, val_loader) in enumerate(fold_dataloaders):
        print(f"Applying shared preprocessing to fold {fold_idx + 1}/{n_splits}")

        train_scaled = ScaledDataset(train_loader.dataset, pca_dim=pca_dim, fit_scalers=False)
        train_scaled.set_preprocessing_params(shared_params)
        val_scaled = ScaledDataset(val_loader.dataset, pca_dim=pca_dim, fit_scalers=False)
        val_scaled.set_preprocessing_params(shared_params)

        train_loader_scaled = DataLoader(train_scaled, batch_size=config.get('batch_size', 32), shuffle=True, num_workers=0)
        val_loader_scaled = DataLoader(val_scaled, batch_size=config.get('batch_size', 32), shuffle=False, num_workers=0)

        preprocessed_fold_dataloaders.append((train_loader_scaled, val_loader_scaled))

        print(f"  Fold {fold_idx + 1}: Preprocessing applied successfully")
    
    # Apply preprocessing to test loader
    test_scaled = ScaledDataset(test_loader.dataset, pca_dim=pca_dim, fit_scalers=False)
    test_scaled.set_preprocessing_params(shared_params)
    test_loader_scaled = DataLoader(test_scaled, batch_size=config.get('batch_size', 32), shuffle=False, num_workers=0)

    print("Preprocessing applied to all datasets with file-level split integrity maintained ✓")

    return preprocessed_fold_dataloaders, test_loader_scaled, shared_params


class ScaledDataset(Dataset):
    """
    Dataset wrapper that applies scaling and PCA to x (state) and scaling to u (control) data
    Following the preprocessing pipeline: x -> scale -> PCA(2D) -> scale, u -> scale
    """
    
    def __init__(self, base_dataset, pca_dim=2, fit_scalers=False):
        """
        Initialize scaled dataset with PCA preprocessing
        
        Args:
            base_dataset: Original dataset
            pca_dim: Number of PCA components (default: 2)
            fit_scalers: Whether to fit the scalers and PCA on this dataset
        """
        self.base_dataset = base_dataset
        self.pca_dim = pca_dim
        
        # Initialize components
        self.scaler_x1 = StandardScaler()  # First x scaler (before PCA)
        self.scaler_x2 = StandardScaler()  # Second x scaler (after PCA)
        self.scaler_u = StandardScaler()   # U scaler
        self.pca = PCA(n_components=pca_dim)
        
        # Statistics storage
        self.mean_1 = None
        self.std_1 = None
        self.mean_2 = None
        self.std_2 = None
        self.mean_u = None
        self.std_u = None
        
        if fit_scalers:
            self._fit_scalers_and_pca()
    
    def _fit_scalers_and_pca(self):
        """Fit scalers and PCA on the entire dataset following the preprocessing pipeline"""
        print("Fitting scalers and PCA on dataset...")
        
        # Collect all x and u data for fitting
        all_x_data = []
        all_u_data = []
        
        for i in range(len(self.base_dataset)):
            x_batch, u_batch, p_batch = self.base_dataset[i]
            
            # Reshape for processing (samples, features)
            x_reshaped = x_batch.view(-1, x_batch.shape[-1])  # (seq_len, features)
            u_reshaped = u_batch.view(-1, u_batch.shape[-1])  # (seq_len, features)
            
            all_x_data.append(x_reshaped.numpy())
            all_u_data.append(u_reshaped.numpy())
        
        # Concatenate all data
        x_data = np.concatenate(all_x_data, axis=0)
        u_data = np.concatenate(all_u_data, axis=0)
        
        # Step 1: First scaling for x_data
        self.mean_1 = np.mean(x_data, axis=0)
        self.std_1 = np.std(x_data, axis=0)
        # Avoid division by zero
        self.std_1 = np.where(self.std_1 == 0, 1, self.std_1)
        x_data_scaled1 = (x_data - self.mean_1) / self.std_1
        
        # Step 2: Scaling for u_data
        self.mean_u = np.mean(u_data, axis=0)
        self.std_u = np.std(u_data, axis=0)
        # Avoid division by zero
        self.std_u = np.where(self.std_u == 0, 1, self.std_u)
        
        # Step 3: PCA on scaled x_data
        self.pca.fit(x_data_scaled1)
        x_data_pca = self.pca.transform(x_data_scaled1)
        
        # Step 4: Second scaling after PCA
        self.mean_2 = np.mean(x_data_pca, axis=0)
        self.std_2 = np.std(x_data_pca, axis=0)
        # Avoid division by zero
        self.std_2 = np.where(self.std_2 == 0, 1, self.std_2)
        
        print(f"X data - Original shape: {x_data.shape}")
        print(f"X data - After PCA shape: {x_data_pca.shape}")
        print(f"X data - Mean1: {self.mean_1[:5]}, Std1: {self.std_1[:5]}")
        print(f"X data - Mean2: {self.mean_2}, Std2: {self.std_2}")
        print(f"U data - Mean: {self.mean_u}, Std: {self.std_u}")
        print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_}")
        print("Scalers and PCA fitted successfully!")
        # Save preprocessing parameters to file
        self.save_preprocessing_params("./results/preprocessing_params_fold.pkl")

    def save_preprocessing_params(self, filepath):
        """Save preprocessing parameters to a file"""
        params = self.get_preprocessing_params()
        with open(filepath, "wb") as f:
            pickle.dump(params, f)

    def load_preprocessing_params(self, filepath):
        """Load preprocessing parameters from a file"""
        with open(filepath, "rb") as f:
            params = pickle.load(f)
        self.set_preprocessing_params(params)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        x_batch, u_batch, p_batch = self.base_dataset[idx]
        
        # Process x data: scale -> PCA -> scale
        x_shape = x_batch.shape
        x_reshaped = x_batch.view(-1, x_shape[-1])  # (seq_len, features)
        x_numpy = x_reshaped.numpy()
        
        # Step 1: First scaling
        x_scaled1 = (x_numpy - self.mean_1) / self.std_1
        
        # Step 2: PCA transformation
        x_pca = self.pca.transform(x_scaled1)
        
        # Step 3: Second scaling
        x_final = (x_pca - self.mean_2) / self.std_2
        
        # Convert back to tensor with new shape (seq_len, pca_dim)
        x_processed = torch.from_numpy(x_final).view(x_shape[0], self.pca_dim).float()
        
        # Process u data: scale only
        u_shape = u_batch.shape
        u_reshaped = u_batch.view(-1, u_shape[-1])  # (seq_len, features)
        u_numpy = u_reshaped.numpy()
        u_scaled = (u_numpy - self.mean_u) / self.std_u
        u_processed = torch.from_numpy(u_scaled).view(u_shape).float()
        
        return x_processed, u_processed, p_batch
    
    def get_preprocessing_params(self):
        """Return the preprocessing parameters"""
        return {
            'mean_1': self.mean_1,
            'std_1': self.std_1,
            'mean_2': self.mean_2,
            'std_2': self.std_2,
            'mean_u': self.mean_u,
            'std_u': self.std_u,
            'pca_components': self.pca.components_,
            'pca_explained_variance_ratio': self.pca.explained_variance_ratio_,
            'pca_explained_variance': self.pca.explained_variance_,
        }
    
    def set_preprocessing_params(self, params):
        """Set preprocessing parameters from external source"""
        self.mean_1 = params['mean_1']
        self.std_1 = params['std_1']
        self.mean_2 = params['mean_2']
        self.std_2 = params['std_2']
        self.mean_u = params['mean_u']
        self.std_u = params['std_u']

        # Reconstruct PCA model
        self.pca.components_ = params['pca_components']
        self.pca.n_components_ = params['pca_components'].shape[0]
        self.pca.n_features_ = params['pca_components'].shape[1]
        self.pca.mean_ = np.zeros(self.pca.n_features_)  # safe default

        if 'pca_explained_variance_ratio' in params:
            self.pca.explained_variance_ratio_ = params['pca_explained_variance_ratio']

        # Add this to fix AttributeError during transform
        self.pca.explained_variance_ = np.ones(self.pca.n_components_)  # dummy values, needed for transform

    
    @property
    def p_labels(self):
        # Delegates to base dataset
        return self.base_dataset.p_labels if hasattr(self.base_dataset, "p_labels") else None
    
    @property 
    def file_indices(self):
        # Delegates to base dataset if it has file tracking
        return self.base_dataset.file_indices if hasattr(self.base_dataset, "file_indices") else None
    
    def get_samples_by_file(self, file_idx):
        # Delegates to base dataset if it has file tracking
        if hasattr(self.base_dataset, "get_samples_by_file"):
            return self.base_dataset.get_samples_by_file(file_idx)
        else:
            return []
    
    def get_file_index_for_sample(self, sample_idx):
        # Delegates to base dataset if it has file tracking  
        if hasattr(self.base_dataset, "get_file_index_for_sample"):
            return self.base_dataset.get_file_index_for_sample(sample_idx)
        else:
            return None

def validate_data_split_integrity(dataset, fold_dataloaders, test_loader):
    """
    Validate that there's no data leakage between train/val/test splits
    
    Args:
        dataset: Original dataset with file tracking
        fold_dataloaders: List of (train_loader, val_loader) for each fold
        test_loader: Test data loader
    """
    print("Validating data split integrity...")
    
    # Get test file indices
    test_file_indices = set()
    for idx in test_loader.dataset.indices:
        file_idx = dataset.get_file_index_for_sample(idx)
        test_file_indices.add(file_idx)
    
    print(f"Test set uses files: {sorted(test_file_indices)}")
    
    # Check each fold
    for fold_idx, (train_loader, val_loader) in enumerate(fold_dataloaders):
        print(f"\nValidating fold {fold_idx + 1}:")
        
        # Get train file indices
        train_file_indices = set()
        for idx in train_loader.dataset.indices:
            file_idx = dataset.get_file_index_for_sample(idx)
            train_file_indices.add(file_idx)
        
        # Get val file indices
        val_file_indices = set()
        for idx in val_loader.dataset.indices:
            file_idx = dataset.get_file_index_for_sample(idx)
            val_file_indices.add(file_idx)
        
        print(f"  Train set uses files: {sorted(train_file_indices)}")
        print(f"  Val set uses files: {sorted(val_file_indices)}")
        
        # Check for overlaps
        train_val_overlap = train_file_indices & val_file_indices
        train_test_overlap = train_file_indices & test_file_indices
        val_test_overlap = val_file_indices & test_file_indices
        
        if train_val_overlap:
            raise ValueError(f"Fold {fold_idx + 1}: Train-Val file overlap detected: {train_val_overlap}")
        if train_test_overlap:
            raise ValueError(f"Fold {fold_idx + 1}: Train-Test file overlap detected: {train_test_overlap}")
        if val_test_overlap:
            raise ValueError(f"Fold {fold_idx + 1}: Val-Test file overlap detected: {val_test_overlap}")
        
        print(f"  Fold {fold_idx + 1}: No file overlap detected ✓")
    
    print("Data split integrity validation completed successfully! ✓")
