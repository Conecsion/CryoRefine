import os
import cv2
import torch
import mrcfile
import random
import string
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from pytorch3d.transforms import quaternion_to_matrix
from operation3D import project, rotate3D
from config import device, samplesPerVolume, cacheDir, projectionWorldSize

os.makedirs('data/pretrain/cache', exist_ok=True)
volumeDir = 'data/volumes/'


# def quanternionGenerator(batchSize: int, device=device) -> torch.Tensor:
#     '''
#     Generate random quaternions
#     return shape (batchSize, 4)
#     '''
#     # Use abs to make sure the quaternions are unique
#     scalar = torch.abs(torch.randn(batchSize, 1))
#     vec = torch.randn(batchSize, 3)
#     quaternions = torch.cat([scalar, vec], dim=1)
#
#     # Normalize to unit quaternions
#     norm = torch.norm(quaternions, dim=1)[:, None]
#     quaternions = quaternions / norm
#
#     return quaternions.to(device)


def batchProj(volumeName: str, volume: torch.Tensor, batchSize: int):
    '''
    volume shape (D, H, W)
    return shape (batchSize, C, H, W)
    '''
    device = volume.device
    os.makedirs(os.path.join(cacheDir, volumeName), exist_ok=True)
    D, H, W = volume.shape
    quaternions = quanternionGenerator(batchSize).to(
        device)  # shape (batchSize, 4)
    rotMat = quaternion_to_matrix(quaternions)  # shape (batchSize, 3, 3)
    rotVol = rotate3D(volume[None, None, :].expand(batchSize, 1, D, H, W),
                      rotMat)  # shape (batchSize, C, D, H, W)
    projs = project(rotVol)  # shape (batchSize, C, H, W)
    for proj in projs:
        filename = ''.join(
            random.choices(string.ascii_letters + string.digits,
                           k=16)) + '.png'
        cv2.imwrite(os.path.join(cacheDir, volumeName, filename),
                    proj[0].to(device='cpu').numpy())


class DDPProject(nn.Module):

    '''
    projectBatchSize: the number of projections for each GPU
    '''
    def __init__(self, volumeName: str, projectBatchSize: int):
        super(DDPProject, self).__init__()
        self.volumeName = volumeName
        self.projectBatchSize = projectBatchSize
        with mrcfile.open(volumeName) as mrc:
            self.volume = torch.from_numpy(mrc.data).detach().clone() # volume shape (D, H, W)

    def quaternionsGenerator(self):
        scalar = torch.abs(torch.randn(self.projectBatchSize, 1))
        vec = torch.randn(self.projectBatchSize, 3)
        quaternions = torch.cat([scalar, vec], dim=1)

        # Normalize to unit quaternions
        norm = torch.norm(quaternions, dim=1)[:, None]
        return quaternions / norm

    def forward(self):
        matrices = quaternion_to_matrix(self.quaternionsGenerator()) # (N, 3, 3)
        D, H, W = self.volume.shape
        rotatedVolume = rotate3D(self.volume[None,:].expand(self.projectionBatchSize, 1, D, H, W))

def projector(gpu_id, )


# Get all volume files
volumeFiles = []
for file in os.listdir(volumeDir):
    if file.endswith('.mrc') or file.endswith('.map'):
        volumeFiles.append(file)
        with mrcfile.open(os.path.join(volumeDir, file)) as mrc:
            volume = torch.tensor(mrc.data).clone().detach().to(device)
            mpProj(file, volume, samplesPerVolume, samplesPerVolume)
