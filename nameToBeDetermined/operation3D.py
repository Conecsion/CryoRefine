import torch
from typing import Tuple
from torchvision.transforms.functional import rotate, InterpolationMode


def rotate3D(volume: torch.Tensor,
             angles: torch.Tensor,
             expand=False,
             interpolation=InterpolationMode.BILINEAR,
             fill=None) -> torch.Tensor:
    # volume shape (batch, Z, X, Y)
    # angles shape (batch, 3)
    # return shape (batch, Z, X, Y)
    anglesX = angles[:, 0]
    anglesY = angles[:, 1]
    anglesZ = angles[:, 2]
    if expand:
        L, H, W = volume.shape[-3:]
        volume = torch.nn.functional.pad(
            volume, (W // 2, W // 2, H // 2, H // 2, L // 2, L // 2),
            'constant', fill)
    dtype = volume.dtype if torch.is_floating_point(volume) else torch.float32
    rotatedVolume = volume.clone().detach()
    return rotatedVolume


def rotate3d(
        tensor: torch.Tensor,  # (Z, X, Y)
        rotation_angle,  # rotation angle around (x,y,z) axes counterclockwise in degree
        expand=False,
        fill=None) -> torch.Tensor:
    if expand:
        L, H, W = tensor.shape[-3:]
        tensor = pad(tensor, (W // 2, W // 2, H // 2, H // 2, L // 2, L // 2),
                     'constant', fill)
    rotated_tensor = tensor.clone().detach()
    # rotated_tensor = rotate(rotated_tensor.permute(1, 0, 2),
    #                         rotation_angle[0],
    #                         expand=False).permute(1, 0, 2)
    rotated_tensor = rotate(rotated_tensor.permute(-2, -3, -1),
                            rotation_angle[-3],
                            expand=False).permute(-2, -3, -1)
    rotated_tensor = rotate(rotated_tensor.permute(-1, -3, -2),
                            rotation_angle[-2],
                            expand=False).permute(-3, -2, -1)
    rotated_tensor = rotate(rotated_tensor, rotation_angle[-1], expand=False)
    return rotated_tensor
