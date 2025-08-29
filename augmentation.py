import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict

# Define constants and configuration
class Config:
    def __init__(self, 
                 batch_size: int = 32, 
                 num_workers: int = 4, 
                 augmentation_prob: float = 0.5, 
                 rotation_angle: int = 30, 
                 translation_x: int = 10, 
                 translation_y: int = 10):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation_prob = augmentation_prob
        self.rotation_angle = rotation_angle
        self.translation_x = translation_x
        self.translation_y = translation_y

class AugmentationException(Exception):
    """Base class for augmentation exceptions."""
    pass

class InvalidAugmentationConfig(AugmentationException):
    """Raised when the augmentation configuration is invalid."""
    pass

class Augmentation:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def _validate_config(self) -> None:
        """Validate the augmentation configuration."""
        if self.config.batch_size <= 0:
            raise InvalidAugmentationConfig("Batch size must be greater than 0")
        if self.config.num_workers <= 0:
            raise InvalidAugmentationConfig("Number of workers must be greater than 0")
        if self.config.augmentation_prob < 0 or self.config.augmentation_prob > 1:
            raise InvalidAugmentationConfig("Augmentation probability must be between 0 and 1")

    def _apply_rotation(self, image: np.ndarray) -> np.ndarray:
        """Apply rotation to the image."""
        import cv2
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, self.config.rotation_angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))

    def _apply_translation(self, image: np.ndarray) -> np.ndarray:
        """Apply translation to the image."""
        import cv2
        (h, w) = image.shape[:2]
        M = np.float32([[1, 0, self.config.translation_x], [0, 1, self.config.translation_y]])
        return cv2.warpAffine(image, M, (w, h))

    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation to the image."""
        import random
        if random.random() < self.config.augmentation_prob:
            image = self._apply_rotation(image)
            image = self._apply_translation(image)
        return image

    def augment(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Apply augmentation to a list of images."""
        self._validate_config()
        augmented_images = []
        for image in images:
            augmented_image = self._apply_augmentation(image)
            augmented_images.append(augmented_image)
        return augmented_images

class HandwritingDataset(Dataset):
    def __init__(self, images: List[np.ndarray], labels: List[int], config: Config):
        self.images = images
        self.labels = labels
        self.config = config
        self.augmentation = Augmentation(config)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        image = self.images[index]
        label = self.labels[index]
        image = self.augmentation._apply_augmentation(image)
        return image, label

def create_data_loader(images: List[np.ndarray], labels: List[int], config: Config) -> DataLoader:
    """Create a data loader for the handwriting dataset."""
    dataset = HandwritingDataset(images, labels, config)
    return DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers)

def main():
    # Example usage
    config = Config()
    images = [np.random.rand(256, 256) for _ in range(100)]
    labels = [0] * 100
    data_loader = create_data_loader(images, labels, config)
    for batch in data_loader:
        images, labels = batch
        print(images.shape, labels.shape)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()