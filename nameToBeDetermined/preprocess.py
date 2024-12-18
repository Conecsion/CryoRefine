import torch
import os
from config import device, projectionNSIDE, samplesPerVolume
import healpy as hp
import mrcfile
from operation3D import rotate3D, project
import concurrent.futures
from IO import cryoWrite

chunksize = 3

os.makedirs('data/pretrain/', exist_ok=True)
volumeDir = 'data/volumes/'
projectionsDir = 'data/pretrain/projections'
volumeFiles = []
for file in os.listdir(volumeDir):
    if file.endswith('.mrc') or file.endswith('.map'):
        volumeFiles.append(file)

def tensor2png(tensor:torch.Tensor, filePath:str):
    pass

for file in volumeFiles:
    with mrcfile.open(os.path.join(volumeDir, file), mode='r') as mrc:
        volume = torch.from_numpy(mrc.data).detach().clone().to(device=device)
    volume = volume.unsqueeze(0).unsqueeze(0).repeat(chunksize, 1, 1, 1, 1)

    outputDir = os.path.join(projectionsDir, file)
    os.makedirs(outputDir, exist_ok=True)

    for i in range(samplesPerVolume // chunksize):
        projectionAngles = (torch.rand(chunksize, 3) * 360).to(device=device)
        # theta ranging from 0 to 180
        projectionAngles[:, 1] /= 2
        # projections shape (chunksize, C, H, W)
        projections = project(rotate3D(volume, projectionAngles))
        projections = projections.squeeze(1).to(device='cpu')
        print(projections.shape)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            id = ''
            outputFilenames = os.path.splitext(file)[0] + id + '.png'
            executor.map(, projections)
