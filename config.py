import logging
import os
import yaml
from typing import Dict, List, Optional
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    def __init__(self, config_file: str = 'config.yaml'):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Dict:
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                return config
        except FileNotFoundError:
            logger.error(f'Config file {self.config_file} not found.')
            raise
        except yaml.YAMLError as e:
            logger.error(f'Error parsing config file {self.config_file}: {e}')
            raise

    def validate_config(self) -> bool:
        required_keys = ['model', 'data', 'training']
        for key in required_keys:
            if key not in self.config:
                logger.error(f'Missing required key {key} in config file.')
                return False
        return True

    def get_config(self) -> Dict:
        return self.config

class ModelConfig:
    def __init__(self, config: Dict):
        self.config = config
        self.model = self.config['model']
        self.data = self.config['data']
        self.training = self.config['training']

    def get_model_config(self) -> Dict:
        return self.model

    def get_data_config(self) -> Dict:
        return self.data

    def get_training_config(self) -> Dict:
        return self.training

class DataConfig:
    def __init__(self, config: Dict):
        self.config = config
        self.data_path = self.config['data_path']
        self.batch_size = self.config['batch_size']
        self.num_workers = self.config['num_workers']

    def get_data_path(self) -> str:
        return self.data_path

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_num_workers(self) -> int:
        return self.num_workers

class TrainingConfig:
    def __init__(self, config: Dict):
        self.config = config
        self.learning_rate = self.config['learning_rate']
        self.num_epochs = self.config['num_epochs']
        self.validation_split = self.config['validation_split']

    def get_learning_rate(self) -> float:
        return self.learning_rate

    def get_num_epochs(self) -> int:
        return self.num_epochs

    def get_validation_split(self) -> float:
        return self.validation_split

class ConfigManager:
    def __init__(self, config_file: str = 'config.yaml'):
        self.config_file = config_file
        self.config = Config(config_file)

    def get_config(self) -> Dict:
        return self.config.get_config()

    def get_model_config(self) -> ModelConfig:
        return ModelConfig(self.config.get_config())

    def get_data_config(self) -> DataConfig:
        return DataConfig(self.config.get_config()['data'])

    def get_training_config(self) -> TrainingConfig:
        return TrainingConfig(self.config.get_config()['training'])

def load_config() -> ConfigManager:
    config_manager = ConfigManager()
    return config_manager

def get_model_config() -> ModelConfig:
    config_manager = load_config()
    return config_manager.get_model_config()

def get_data_config() -> DataConfig:
    config_manager = load_config()
    return config_manager.get_data_config()

def get_training_config() -> TrainingConfig:
    config_manager = load_config()
    return config_manager.get_training_config()

if __name__ == '__main__':
    config_manager = load_config()
    model_config = get_model_config()
    data_config = get_data_config()
    training_config = get_training_config()

    logger.info(f'Model config: {model_config.get_model_config()}')
    logger.info(f'Data config: {data_config.get_data_config()}')
    logger.info(f'Training config: {training_config.get_training_config()}')