import torch
from torch.utils.data import DataLoader, Dataset
import mrcfile
import os
from operation3D import rotate3D, project


class ProjectionWithAngles():

    def __init__(self, projection: torch.Tensor, angles: torch.Tensor):
        self.projection = projection
        self.angles = angles


class ProjectionDataset(Dataset):

    def __init__(self, volumeDir, transform=None):
        self.volumeDir = volumeDir
        self.volumeFiles = []
        for file in os.listdir(self.volumeDir):
            if file.endswith('.mrc') or file.endswith('.map'):
                self.volumeFiles.append(file)
        self.transfomr = transform

    def __len__(self):
        return len(self.volumeFiles)

    def __getitem__(self, idx):
        pass
