import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

def integer_to_one_hot(integer, min_val, max_val):
    vector_length = max_val - min_val + 1
    one_hot_vector = [0] * vector_length
    one_hot_vector[integer - min_val] = 1
    return one_hot_vector

def sliding_window_split(sequence, window_size=30, stride=10):
    slices = []
    for start in range(0, len(sequence) - window_size + 1, stride):
        end = start + window_size
        slices.append(sequence[start:end])
    return slices

def load_dataset_from_files(config):
    data_dir = config['data_dir']
    x_trajectories = []
    u_trajetories = []
    p_lables = []
    min_val = 2
    max_val = 7
    for item in os.listdir(data_dir):
        data_file_path = os.path.join(data_dir, item)

        # Check if the file is a directory
        if data_file_path.endswith('.npy') and os.path.exists(data_file_path):
            data_dict = np.load(data_file_path, allow_pickle=True).item()
            x_data = data_dict['signals'][:, :-6]
            if x_data.shape[0]!= 3417 or x_data.shape[1]!= 68:
                print(x_data.shape)
                # print(data_file_path)
            u_data = data_dict['signals'][:, -6:-4]

            x_data = x_data[::config['sample_step'], :]
            u_data = u_data[::config['sample_step'], :]

            ErrorType = data_dict['ErrorType']
            p_data = np.array(integer_to_one_hot(ErrorType, min_val, max_val))
            
            x_trajectories.append(x_data)
            u_trajetories.append(u_data)
            p_lables.append(p_data)

    return x_trajectories, u_trajetories, p_lables

class TrajectoryDataset(Dataset):
    def __init__(self, x_trajectories, u_trajetories, p_lables, )


