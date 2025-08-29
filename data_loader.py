import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants and configuration
@dataclass
class Config:
    data_dir: str
    batch_size: int
    num_workers: int
    image_size: Tuple[int, int]
    num_classes: int

@dataclass
class DataConfig:
    image_dir: str
    annotation_file: str
    class_labels: List[str]

class DataMode(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3

class ImageDataset(Dataset):
    def __init__(self, config: Config, data_config: DataConfig, mode: DataMode):
        self.config = config
        self.data_config = data_config
        self.mode = mode
        self.image_dir = os.path.join(config.data_dir, data_config.image_dir)
        self.annotation_file = os.path.join(config.data_dir, data_config.annotation_file)
        self.class_labels = data_config.class_labels
        self.image_paths = self.load_image_paths()
        self.annotation_data = self.load_annotation_data()

    def load_image_paths(self):
        image_paths = []
        for root, dirs, files in os.walk(self.image_dir):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def load_annotation_data(self):
        with open(self.annotation_file, "r") as f:
            annotation_data = json.load(f)
        return annotation_data

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = self.load_image(image_path)
        annotation = self.annotation_data[index]
        label = self.get_label(annotation)
        return image, label

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, self.config.image_size)
        image = image / 255.0
        return image

    def get_label(self, annotation):
        label = annotation["label"]
        return self.class_labels.index(label)

class DataLoader:
    def __init__(self, config: Config, data_config: DataConfig, mode: DataMode):
        self.config = config
        self.data_config = data_config
        self.mode = mode
        self.dataset = ImageDataset(config, data_config, mode)
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True if mode == DataMode.TRAIN else False,
        )

    def __iter__(self):
        return iter(self.data_loader)

    def __len__(self):
        return len(self.data_loader)

def create_data_loader(config: Config, data_config: DataConfig, mode: DataMode):
    data_loader = DataLoader(config, data_config, mode)
    return data_loader

def main():
    config = Config(
        data_dir="/path/to/data",
        batch_size=32,
        num_workers=4,
        image_size=(224, 224),
        num_classes=10,
    )

    data_config = DataConfig(
        image_dir="images",
        annotation_file="annotations.json",
        class_labels=["label1", "label2", "label3"],
    )

    data_loader = create_data_loader(config, data_config, DataMode.TRAIN)
    for batch in data_loader:
        images, labels = batch
        logger.info(f"Batch size: {len(images)}")
        logger.info(f"Labels: {labels}")

if __name__ == "__main__":
    main()