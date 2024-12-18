import torch
from torchvision.transforms import InterpolationMode, Normalize
from torchvision.transforms.functional import rotate
from torch.nn.functional import pad
from operation_2D import norm
import numpy as np
import os


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
    rotated_tensor = rotate(rotated_tensor.permute(-2, -3, -1),
                            rotation_angle[-3],
                            expand=False).permute(-2, -3, -1)
    rotated_tensor = rotate(rotated_tensor.permute(-1, -3, -2),
                            rotation_angle[-2],
                            expand=False).permute(-3, -2, -1)
    rotated_tensor = rotate(rotated_tensor, rotation_angle[-1], expand=False)
    return rotated_tensor


def projection(tensor: torch.Tensor, norm=Normalize) -> torch.Tensor:
    return (Normalize(mean=1, std=0)(tensor.sum(dim=0).unsqueeze(0)))


def random_proj(tensor: torch.Tensor,
                num: int,
                output_path: str,
                expand=True):
    angles = np.random.randint(0, 361, size=(num, 3)).tolist()
    with open(os.path.join(output_path, 'labels.csv'), 'w') as f:
        writer = csv.writer(f)
        header = ['filename', 'angle1', 'angle2', 'angle3']
        writer.writerow(header)
        for angle in angles:
            print(angle)
            rotated_tensor = rotate3d(tensor, angle, expand=expand)
            proj = projection(rotated_tensor)
            filename = f'{angle[0]}_{angle[1]}_{angle[2]}.tif'
            tensor2tif(proj, os.path.join(output_path, filename))
            row = [filename, angle[0], angle[1], angle[2]]
            print(row)
            writer.writerow(row)
