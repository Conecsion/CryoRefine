import torch
from torch.utils.data import DataLoader, IterableDataset
import mrcfile
import os
from operation3D import rotate3D, project


class ProjectionWithAngles():

    def __init__(self, projection: torch.Tensor, angles: torch.Tensor):
        self.projection = projection
        self.angles = angles


class ProjectionDataset(IterableDataset):

    def __init__(self, volumeDir, transform=None):
        super(ProjectionDataset).__init__()
        self.volumeDir = volumeDir
        self.volumeFiles = []
        for file in os.listdir(self.volumeDir):
            if file.endswith('.mrc') or file.endswith('.map'):
                self.volumeFiles.append(file)
        self.transfomr = transform
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        start = 0
        end = 10
        return iter(range(start, end))

