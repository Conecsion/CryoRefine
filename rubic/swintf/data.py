#!/usr/bin/env python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os

class CryoMap(Dataset):
    def __init__(self, map_dir, transform=None, target_transform=None):
        self.map_dir = map_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        num_files = 0
        for root, _, files in os.walk(self.map_dir):
            num_files += len(files)
        return num_files

    def __getitem__(self, idx):
        map_path = os.path.join(self.map_dir)
        image = projection()
