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
                x_data = data_dict['signals'][1000:, :-6]  # State signals
                u_data = data_dict['signals'][1000:, -6:-4]  # Control signals
                
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
    
    return train_dataset, test_dataset

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

if __name__ == "__main__":
    # Example config
    config = {
        'sample_step': 1,
        'window_size': 30,
        'batch_size': 2
    }
    data_dir = "./data"

    # Load datasets
    train_dataset, val_dataset = load_dataset_from_folder(data_dir, config)

    # Wrap with ScaledDataset and fit scalers/PCA on train set
    scaled_train = ScaledDataset(train_dataset, pca_dim=2, fit_scalers=True)
    scaled_val = ScaledDataset(val_dataset, pca_dim=2)
    # Load preprocessing params from train to val
    scaled_val.set_preprocessing_params(scaled_train.get_preprocessing_params())

    # Print first 3 samples from train and val
    print("Train samples:")
    for i in range(3):
        x, u, p = scaled_train[i]
        print(f"Sample {i}: x shape {x.shape}, u shape {u.shape}, p {p}")

    print("\nValidation samples:")
    for i in range(3):
        x, u, p = scaled_val[i]
        print(f"Sample {i}: x shape {x.shape}, u shape {u.shape}, p {p}")