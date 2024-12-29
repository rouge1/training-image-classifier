# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

class MNISTDataLoader:
    def __init__(self, data_dir='num_data/mnist', batch_size=128):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        
        # Simpler transform since images are already preprocessed
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Map folder names to correct digit values
        self.class_to_idx = {
            'Eight': 0, 'Five': 1, 'Four': 2, 'Nine': 3, 'One': 4,
            'Seven': 5, 'Six': 6, 'Three': 7, 'Two': 8, 'Zero': 9
        }

    def get_train_loader(self):
        train_dataset = datasets.ImageFolder(
            self.data_dir / 'train',
            transform=self.transform
        )
        # Override the default alphabetical mapping with our digit mapping
        train_dataset.class_to_idx = self.class_to_idx
        
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2
        )

    def get_test_loader(self):
        test_dataset = datasets.ImageFolder(
            self.data_dir / 'test',
            transform=self.transform
        )
        # Override the default alphabetical mapping with our digit mapping
        test_dataset.class_to_idx = self.class_to_idx
        
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2
        )
    
    def get_class_mapping(self):
        return self.class_to_idx