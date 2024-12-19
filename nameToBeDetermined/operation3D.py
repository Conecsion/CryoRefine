import torch
import torch.nn.functional as F


def zyz2AffineMat(angleDegree: torch.Tensor) -> torch.Tensor:
    '''
    angleDegree shape (N, 3) ; the rotation angles are given by ZYZ convention order, which is (phi, theta, psi) in an intrinsic way
    return shape (N, 3, 4)
    '''
    device = angleDegree.device
    batchSize = angleDegree.shape[0]
    angleRadian = torch.deg2rad(angleDegree)

    # Construct the rotation matrices with shape (N, 3, 3)
    # Rphi rotate around z axis
    Rphi = torch.zeros(batchSize, 3, 3)
    Rphi[:, 0, 0] = torch.cos(angleRadian[:, 0])
    Rphi[:, 0, 1] = -torch.sin(angleRadian[:, 0])
    Rphi[:, 1, 0] = torch.sin(angleRadian[:, 0])
    Rphi[:, 1, 1] = torch.cos(angleRadian[:, 0])
    Rphi[:, 2, 2] = 1
    # Make rotation right-handed
    Rphi = torch.inverse(Rphi)

    # Rtheta rotate around y' axis
    Rtheta = torch.zeros(batchSize, 3, 3)
    Rtheta[:, 0, 0] = torch.cos(angleRadian[:, 1])
    Rtheta[:, 0, 2] = torch.sin(angleRadian[:, 1])
    Rtheta[:, 1, 1] = 1
    Rtheta[:, 2, 0] = -torch.sin(angleRadian[:, 1])
    Rtheta[:, 2, 2] = torch.cos(angleRadian[:, 1])
    Rtheta = torch.inverse(Rtheta)  # Make rotation right-handed

    # Rpsi rotate around z'' axis
    Rpsi = torch.zeros(batchSize, 3, 3)
    Rpsi[:, 0, 0] = torch.cos(angleRadian[:, 2])
    Rpsi[:, 0, 1] = -torch.sin(angleRadian[:, 2])
    Rpsi[:, 1, 0] = torch.sin(angleRadian[:, 2])
    Rpsi[:, 1, 1] = torch.cos(angleRadian[:, 2])
    Rpsi[:, 2, 2] = 1
    Rpsi = torch.inverse(Rpsi)  # Make rotation right-handed

    Rphi = Rphi.to(device)
    Rtheta = Rtheta.to(device)
    Rpsi = Rpsi.to(device)
    '''
    In a intrinsic rotation, the object should first rotate around Z axis for Rphi degree anti-clockwise,
    then Y' axis for Rtheta, then Z'' axis for Rpsi
    I know for an intrinsic rotation I need to write the formula as Rphi @ Rtheta @ Rpsi
    But this strange function affine_grid seems to rotate the grids instead of the object
    So Rpsi @ Rtheta @ Rphi is correct
    '''
    R = Rpsi @ Rtheta @ Rphi  # R shape (N, 3, 3)

    # Construct the affine matrices with shape (N, 3, 4)
    # No translation is applied, got zero vectors
    translationVectors = torch.zeros(batchSize, 3, 1).to(
        device)  # translationVectors shape (N, 3, 1)

    # affineMatrices shape (N, 3, 4)
    affineMatrices = torch.cat((R, translationVectors), 2)

    return affineMatrices


def rotate3D(volume: torch.Tensor,
             affineMatrices: torch.Tensor,
             interpolation='bilinear') -> torch.Tensor:
    '''
    Rotate the input volume by affineMatrices
    volume shape (N, C, D, H, W), in which (D, H, W) corresponds to (Z, Y, X) axes of a density map
    affineMatrices shape (N, 3, 4)
    return shape (N, C, D, H, W)
    '''
    device = volume.device
    affineMatrices.to(device)
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
