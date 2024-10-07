import torch
import os
import numpy as np
from mrc import *
from rotate3d import rotate3d
from normalize import norm
import csv


def projection(tensor: torch.Tensor, norm=norm):
    return norm(tensor.sum(dim=0))


def random_proj(tensor: torch.Tensor,
                num: int,
                output_path: str,
                norm=norm,
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
