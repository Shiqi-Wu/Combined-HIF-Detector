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

def load_full_dataset(data_dir, config, delete_columns=[9, 21, 25, 39, 63]):
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
                
                # Apply column deletion
                x_data = np.delete(x_data, delete_columns, axis=1)
                
                # print(x_data.shape, u_data.shape)
                x_data = x_data[100:]  # Skip first 100 samples
                u_data = u_data[100:]  # Skip first 100 samples
                
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


def create_preprocessed_kfold_dataloaders(dataset, file_labels, config, n_splits=5, random_state=42, pca_dim=2):
    """
    Create k-fold cross validation dataloaders with shared preprocessing (scaling + PCA) parameters.

    Args:
        dataset: Complete dataset
        file_labels: Labels for stratified splitting
        config: Configuration dictionary
        n_splits: Number of folds
        random_state: Random seed
        pca_dim: Number of PCA components

    Returns:
        fold_dataloaders: List of (train_loader, val_loader, test_loader) for each fold
        preprocessing_params: Shared preprocessing parameters used for all folds
    """
    from torch.utils.data import DataLoader

    # First get the basic fold dataloaders
    fold_dataloaders = create_kfold_dataloaders(dataset, file_labels, config, n_splits, random_state)

    print("Fitting shared preprocessing parameters on full dataset...")
    scaler_dataset = ScaledDataset(dataset, pca_dim=pca_dim, fit_scalers=True)
    shared_params = scaler_dataset.get_preprocessing_params()

    preprocessed_fold_dataloaders = []

    for fold_idx, (train_loader, val_loader, test_loader) in enumerate(fold_dataloaders):
        print(f"Applying shared preprocessing to fold {fold_idx + 1}/{n_splits}")

        train_scaled = ScaledDataset(train_loader.dataset, pca_dim=pca_dim, fit_scalers=False)
        train_scaled.set_preprocessing_params(shared_params)
        val_scaled = ScaledDataset(val_loader.dataset, pca_dim=pca_dim, fit_scalers=False)
        val_scaled.set_preprocessing_params(shared_params)
        test_scaled = ScaledDataset(test_loader.dataset, pca_dim=pca_dim, fit_scalers=False)
        test_scaled.set_preprocessing_params(shared_params)

        train_loader_scaled = DataLoader(train_scaled, batch_size=config.get('batch_size', 32), shuffle=True, num_workers=0)
        val_loader_scaled = DataLoader(val_scaled, batch_size=config.get('batch_size', 32), shuffle=False, num_workers=0)
        test_loader_scaled = DataLoader(test_scaled, batch_size=config.get('batch_size', 32), shuffle=False, num_workers=0)

        preprocessed_fold_dataloaders.append((train_loader_scaled, val_loader_scaled, test_loader_scaled))

        print(f"  Fold {fold_idx + 1}: Preprocessing applied successfully")

    return preprocessed_fold_dataloaders, shared_params


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
        # Delegates to base dataset (assuming it has p_labels)
        return self.base_dataset.p_labels if hasattr(self.base_dataset, "p_labels") else None
