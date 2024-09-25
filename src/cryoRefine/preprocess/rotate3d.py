import torch
from scipy.ndimage import rotate
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate
from torch.nn.functional import pad
from typing import Tuple, List


def rotate3d(
        tensor: torch.Tensor,  # (Z, X, Y)
        rotation_angle,  # rotation angle around (x,y,z) axes counterclockwise in degree
        interpolation=InterpolationMode.BILINEAR,
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


if __name__ == '__main__':
    points = torch.ones(3, 3, 3)
    # points = torch.tensor([0, 0, -1])
    print(rotate3d(points, (90, 0, 0)))
    # print(rotate3d(points, (45, 45, 45)))
    # print(rotate3d(points, (45, 45, 45)).shape)
