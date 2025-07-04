from src.utils.data import load_dataset_from_files, TrajectoryDataset
from src.utils.args import parse_arguments, read_config_file
from src.utils.data import TrajectoryDataset

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load configuration file
    config = read_config_file(args.config)

    # Load datasets
    training_dir = config['training_data_dir']
    x_trajectories, u_trajetories, p_lables = load_dataset_from_files(training_dir, config)
    training_dataset = TrajectoryDataset(x_trajectories, u_trajetories, p_lables, config['window_size'], config['stride'])
    testing_dir = config['Testing_data_dir']
    x_trajectories, u_trajetories, p_lables = load_dataset_from_files(testing_dir, config)
    testing_dataset = TrajectoryDataset(x_trajectories, u_trajetories, p_lables, config['window_size'], config['stride'])

    # Load Model
    if config['model_type'] == 'lstm_classifier':
        model = 

    
    