import os
import cv2
import pandas
import torch
import random
import string
import mrcfile
from operation3D import rotate3D, project
import torch.multiprocessing as mp
from pytorch3d.transforms import quaternion_to_matrix
from config import gpus, projectorBatchsize, cacheDir


def worker(gpu_id, volumeFile: str, batchSize=projectorBatchsize):
    '''
    gpu_id : 0
    volumeFile : 'data/volumes/emd_7770.mrc'
    '''
    device = torch.device(f'cuda:{gpu_id}')
    volumeName = os.path.splitext(
        os.path.basename(volumeFile))[0]  # 'emd_7770'
    with mrcfile.open(volumeFile, mode='r+') as mrc:
        volume = torch.from_numpy(mrc.data).detach().clone().to(device)

    # Generate random and unique quaternions
    scalar = torch.abs(torch.randn(batchSize, 1))
    vec = torch.randn(batchSize, 3)
    quaternions = torch.cat([scalar, vec], dim=1).to(device)
    # Normalize to unit quaternions
    norm = torch.norm(quaternions, dim=1)[:, None]
    quaternions = quaternions / norm

    # Convert quaternions to rotation matrix
    rotMat = quaternion_to_matrix(quaternions)  # shape (batchSize, 3, 3)

    # Rotate the map
    rotVolume = rotate3D(volume.expand(batchSize, 1, *volume.shape), rotMat)
    projections = project(rotVolume).squeeze(1).to(
        'cpu').numpy()  # (batchSize, H, W)
    quaternions = quaternions.to('cpu').numpy()

    # Save projection images and quaternions
    os.makedirs(os.path.join(cacheDir, volumeName), exist_ok=True)
    for (proj, quaternion) in zip(projections, quaternions):
        # Generate random filename
        filename = ''.join(
            random.choices(string.ascii_letters + string.digits,
                           k=16)) + '.png'
        filePath = os.path.join(cacheDir, volumeName, filename)
        data = {'projection': [filePath], 'quaternion': [quaternion]}
        df = pandas.DataFrame(data)
        df.to_csv(os.path.join(cacheDir, volumeName, 'index.csv'), mode='a')
        cv2.imwrite(filePath, proj)


def projector(projectionNumPerVolume: int):
    # Get all .mrc/.map files
    volumeFiles = []
    for file in os.listdir('data/volumes'):
        if file.endswith('.mrc') or file.endswith('.map'):
            volumeFile = os.path.join('data/volumes', file)
            volumeFiles.append(volumeFile)

    for file in volumeFiles:
        batchNum = projectionNumPerVolume // len(gpus) // projectorBatchsize
        for i in range(batchNum):
            processes = []
            for gpu_id in gpus:
                p = mp.Process(target=worker, args=(gpu_id, file))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()


if __name__ == '__main__':
    projector(5000)
