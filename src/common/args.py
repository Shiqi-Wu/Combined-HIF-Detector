import argparse
import yaml

def parse_arguments():
    """
    Input:
        None
    Function:
        Parses command-line arguments to retrieve the path to the configuration file.
        If no path is provided, defaults to 'config.yaml'.
    Returns:
        args (Namespace): Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()
    return args

# Reads and loads the configuration from a YAML file.
def read_config_file(config_file):
    """
    Input:
        config_file (str): Path to the YAML configuration file.
    Function:
        Opens the specified YAML file and loads its contents as a dictionary.
        If there is an error during reading, it will print the error message.
    Returns:
        config (dict): Configuration parameters loaded from the YAML file.
    """
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config
