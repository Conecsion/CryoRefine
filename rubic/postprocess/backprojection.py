import torch
import numpy as np
from typing import Tuple
import torch.fft as fft
import math


def rotation_matrix(angles: torch.Tensor) -> Tuple:
    """
    angles: (batch, 3) in radian, counter-clockwise, around xyz axes in order
    """
    ang_x = angles[:, 0]
    ang_y = angles[:, 1]
    ang_z = angles[:, 2]
    batch = len(angles)
    Rx, Ry, Rz = torch.zeros(3, batch, 3, 3)
    ang_x = angles[:, 0]
    ang_y = angles[:, 1]
    ang_z = angles[:, 2]
    Rx[:, 0, 0] = 1
    Rx[:, 1, 1] = torch.cos(ang_x)
    Rx[:, 1, 2] = -torch.sin(ang_x)
    Rx[:, 2, 1] = torch.sin(ang_x)
    Rx[:, 2, 2] = torch.cos(ang_x)
    Ry[:, 0, 0] = torch.cos(ang_y)
    Ry[:, 0, 2] = torch.sin(ang_y)
    Ry[:, 1, 1] = 1
    Ry[:, 2, 0] = -torch.sin(ang_y)
    Ry[:, 2, 2] = torch.cos(ang_y)
    Rz[:, 0, 0] = torch.cos(ang_z)
    Rz[:, 0, 1] = -torch.sin(ang_z)
    Rz[:, 1, 0] = torch.sin(ang_z)
    Rz[:, 1, 1] = torch.cos(ang_z)
    Rz[:, 2, 2] = 1
    return Rx, Ry, Rz


def backprojection(projections: torch.Tensor,
                   angles: torch.Tensor):  #-> torch.Tensor:
    """
    projections: (batch, C, H, W)
    angles: (batch, 3) in radian, counter-clockwise
    """
    H, W = projections.shape[-2:]
    batch = projections.size(dim=0)

    # Initiate an empty cubic
    slice = torch.zeros(H, H, W)

    # Get the 3D coordinates of an unrotated slice
    # the slice array store the coordinates of a plane in a 3D volume
    # i_index = torch.arange(H // 2 + 1 - H, H // 2 + 1)
    # j_index = torch.arange(W // 2 + 1 - W, W // 2 + 1)
    i_index = torch.arange(-H // 2, H // 2)
    j_index = torch.arange(-W // 2, W // 2)
    slice_i, slice_j = torch.meshgrid(i_index, j_index, indexing='ij')
    slice_k = torch.zeros(slice_i.size())
    slice = torch.dstack((slice_i, slice_j, slice_k))
    slice = slice.unsqueeze(0).repeat(batch, 1, 1, 1)

    # Fourier transform of projections
    fft_proj = fft.fftshift(fft.fft2(projections))
    # proj_vectors = torch.tensor([0, 0,
    #                              -1.]).unsqueeze(0).repeat(batch,
    #                                                        1).unsqueeze(2)

    # Get rotation matrices
    Rx, Ry, Rz = rotation_matrix(-angles)
    # proj_vector = torch.matmul(Rz, proj_vectors)
    # proj_vector = torch.matmul(Ry, proj_vectors)
    # proj_vector = torch.matmul(Rx, proj_vectors)
    # rotated_grid = torch.matmul(Rx, grid)
    # rotated_grid = torch.matmul(Ry, rotated_grid)
    # rotated_grid = torch.matmul(Rz, rotated_grid)

    # Get the coordinates of slices after rotation
    rotated_slice = torch.einsum('ijk, ibck->ibcj', Rx, slice)
    rotated_slice = torch.einsum('ijk, ibck->ibcj', Ry, rotated_slice)
    rotated_slice = torch.einsum('ijk, ibck->ibcj', Rz, rotated_slice)

    # Find coordinates of the 8 nearest neighbor grid points of each rotated slice coordinate
    ceil_x = rotated_slice.ceil()[:, :, :, 0]
    ceil_y = rotated_slice.ceil()[:, :, :, 1]
    ceil_z = rotated_slice.ceil()[:, :, :, 2]
    floor_x = rotated_slice.floor()[:, :, :, 0]
    floor_y = rotated_slice.floor()[:, :, :, 1]
    floor_z = rotated_slice.floor()[:, :, :, 2]
    p1 = torch.stack([floor_x, floor_y, ceil_z], dim=-1)
    p2 = torch.stack([floor_x, ceil_y, ceil_z], dim=-1)
    p3 = torch.stack([ceil_x, ceil_y, ceil_z], dim=-1)
    p4 = torch.stack([ceil_x, floor_y, ceil_z], dim=-1)
    p5 = torch.stack([floor_x, floor_y, floor_z], dim=-1)
    p6 = torch.stack([floor_x, ceil_y, floor_z], dim=-1)
    p7 = torch.stack([ceil_x, ceil_y, floor_z], dim=-1)
    p8 = torch.stack([ceil_x, floor_y, floor_z], dim=-1)

    # Distance(volume of sub-cubic) from the nearest neighbor points
    w1 = (rotated_slice[:, :, :, 0] -
          floor_x) * (rotated_slice[:, :, :, 1] -
                      floor_y) * (ceil_z - rotated_slice[:, :, :, 2])
    w2 = (rotated_slice[:, :, :, 0] - floor_x) * (
        ceil_y - rotated_slice[:, :, :, 1]) * (ceil_z -
                                               rotated_slice[:, :, :, 2])
    w3 = (ceil_x - rotated_slice[:, :, :, 0]) * (
        ceil_y - rotated_slice[:, :, :, 1]) * (ceil_z -
                                               rotated_slice[:, :, :, 2])
    w4 = (ceil_x - rotated_slice[:, :, :, 0]) * (
        rotated_slice[:, :, :, 1] - floor_y) * (ceil_z -
                                                rotated_slice[:, :, :, 2])
    w5 = (rotated_slice[:, :, :, 0] -
          floor_x) * (rotated_slice[:, :, :, 1] -
                      floor_y) * (rotated_slice[:, :, :, 2] - floor_z)
    w6 = (rotated_slice[:, :, :, 0] - floor_x) * (
        ceil_y - rotated_slice[:, :, :, 1]) * (rotated_slice[:, :, :, 2] -
                                               floor_z)
    w7 = (ceil_x - rotated_slice[:, :, :, 0]) * (
        ceil_y - rotated_slice[:, :, :, 1]) * (rotated_slice[:, :, :, 2] -
                                               floor_z)
    w8 = (ceil_x - rotated_slice[:, :, :, 0]) * (
        rotated_slice[:, :, :, 1] - floor_y) * (rotated_slice[:, :, :, 2] -
                                                floor_z)

    # Initialize 3D Fourier volume
    fft_space = torch.zeros(H, H, W)

    # for batch in fft_proj:
    #     for channel in batch:
    #         for i in channel:
    #             for j in fft_proj[batch][channel][i]:
    #                 pass

    print(fft_proj.size())
    print(w1.size())
    print(p1.size())

    B, C, H, W = fft_proj.size()
    print(B, C, H, W)
    for b in range(B):
        for c in range(C):
            for i in range(H):
                for j in range(W):
                    # print(fft_proj[b][c][i][j])
                    print(fft_proj[b][c][i][j])
                    print(rotated_slice[b][i][j])
                    print(p1[b][i][j])
                    print(w1[b][i][j])
                    break
                break
            break
        break

    realpart, imagpart = fft_proj.real, fft_proj.imag


if __name__ == '__main__':
    imgs = torch.rand(4, 1, 338, 338)
    # angles = torch.rand(4, 3)
    angles = torch.tensor([torch.pi / 4, 0, 0]).unsqueeze(0).repeat(4, 1)
    backprojection(imgs, angles)
    # print(angles)
    # tmp = torch.rand(4, 4)
    # torch.floor(tmp)
    # torch.ceil(tmp)
