import torch
import torch.nn.functional as F
from pytorch3d.transforms import euler_angles_to_matrix


def degrees2mat(angleDegree: torch.Tensor, convention='ZYZ') -> torch.Tensor:
    '''
    Convert Euler angles to rotation matrices
    angleDegree shape (N, 3);
    The rotation angles are given convention order, which is (phi, theta, psi) in an intrinsic way as default
    return shape (N, 3, 3)
    '''
    device = angleDegree.device
    batchSize = angleDegree.shape[0]
    angleRadian = torch.deg2rad(angleDegree)
    return euler_angles_to_matrix(angleRadian, convention).to(device)

def rotate3D(volume: torch.Tensor,
             rotMat: torch.Tensor,
             interpolation='bilinear') -> torch.Tensor:
    '''
    Rotate the input volume by affineMatrices
    volume shape (N, C, D, H, W), in which (D, H, W) corresponds to (Z, Y, X) axes of a density map
    rotMat shape (N, 3, 3)
    return shape (N, C, D, H, W)
    '''
    device = volume.device
    batchSize = volume.shape[0]

    # affine_grid rotate the grids instead of the object, so the rotation matrices need to be inversed
    rotMat = torch.inverse(rotMat).to(device)

    # Pad translation vectors to construct the affine matrices with shape (N, 3, 4)
    # No translation is applied, got zero vectors
    translationVectors = torch.zeros(batchSize, 3, 1).to(
        device)  # translationVectors shape (N, 3, 1)
    # affineMatrices shape (N, 3, 4)
    affineMatrices = torch.cat((rotMat, translationVectors), 2).to(device)

    # Rotate the volume
    rotatedGrid = F.affine_grid(affineMatrices,
                                volume.shape,
                                align_corners=False)
    volume = F.grid_sample(volume,
                           rotatedGrid,
                           mode=interpolation,
                           align_corners=False).to(device)

    return volume


def project(volume: torch.Tensor, norm=True):
    """
    Generate projections from given volume and angleDegree
    volume shape (N, C, D, H, W)
    angleDegree(N, 3)
    return shape (N, C, H, W)
    """
    import torch.fft as fft
    # proj = volume.sum(dim=2)
    volumeFFT = fft.fftshift(fft.rfftn(volume, dim=(2, 3, 4)))
    centralSlice = volumeFFT[:, :, volumeFFT.shape[2] // 2]
    proj = fft.irfftn(fft.ifftshift(centralSlice, dim=(2, 3)))

    if norm:
        proj = torch.nn.InstanceNorm2d(num_features=1)(proj)
    return proj


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import mrcfile
    fig, axes = plt.subplots(1, 2)
    with mrcfile.new('tests/v.mrc', overwrite=True) as mrc:
        volume = torch.zeros(300, 300, 300)
        for i in range(10):
            volume[150 + i, 145:155, 145:155] = 1.
        for i in range(60):
            volume[145:155, 150 + i, 145:155] = 1.
        for i in range(130):
            volume[145:155, 145:155, 150 + i] = 1.
        mrc.set_data(volume.numpy())

    def unsq(tensor):
        return tensor.unsqueeze(0).unsqueeze(0)

    def sque(tensor):
        return tensor.squeeze(0).squeeze(0)

    def angle(a, b, c):
        return torch.tensor([[a, b, c]])

    def save(volume: torch.Tensor, filename: str):
        with mrcfile.new('tests/' + filename, overwrite=True) as mrc:
            mrc.set_data(sque(volume).numpy())

    import pytorch3d.transforms as p3dt
    import numpy as np

    r1Deg = angle(45, 0, 0)
    r2Deg = angle(45, 50, 0)
    r3Deg = angle(45, 50, 70)
    r1Mat = degrees2mat(r1Deg)
    r2Mat = degrees2mat(r2Deg)
    r3Mat = degrees2mat(r3Deg)
    volume = volume[None, None, :]
    print(volume.shape)
    r1 = rotate3D(volume, r1Mat)
    r2 = rotate3D(volume, r2Mat)
    r3 = rotate3D(volume, r3Mat)
    save(r1, 'r1.mrc')
    save(r2, 'r2.mrc')
    save(r3, 'r3.mrc')
