import torch
import numpy as np
from typing import Tuple
import torch.fft as fft
import math


def rotation_matrix(
        angles: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    angles: (batch, 3) in degrees, counter-clockwise, around xyz axes in order
    """

    # Convert degrees to radians
    angles = angles * math.pi / 180.

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
    projections: (batch, H, W)
    angles: (batch, 3) in degrees, counter-clockwise
    """
    H, W = projections.shape[-2:]
    batch = projections.size(dim=0)

    # Initiate an empty cubic
    slice = torch.zeros(H, H, W)

    # Get the 3D coordinates of an unrotated slice
    # the slice array store the coordinates of a plane in a 3D volume
    i_index = torch.arange(-H // 2, H // 2)
    j_index = torch.arange(-W // 2, W // 2)
    slice_i, slice_j = torch.meshgrid(i_index, j_index, indexing='ij')
    slice_k = torch.zeros(slice_i.size())
    slice = torch.dstack((slice_i, slice_j, slice_k))
    slice = slice.unsqueeze(0).repeat(batch, 1, 1, 1)
    # slice.shape = [batch, H, W, 3]

    # Fourier transform of projections
    # fft_proj.shape = [batch, H, W]
    fft_proj = fft.fftshift(fft.fft2(projections))

    # Get rotation matrices
    # Rx.shape = [batch, 3, 3]
    Rx, Ry, Rz = rotation_matrix(-angles)

    # Get the coordinates of slices after rotation
    rotated_slice = torch.einsum('bij, bhwj->bhwi', Rx, slice)
    rotated_slice = torch.einsum('bij, bhwj->bhwi', Ry, rotated_slice)
    rotated_slice = torch.einsum('bij, bhwj->bhwi', Rz, rotated_slice)

    # Find coordinates of the 8 nearest neighbor grid points of each rotated slice coordinate
    # Nearest neighbor points of a cube in order 111, -111, -1-11, 1-11, 11-1, -11-1, -1-1-1, 1-1-1
    nearest_neighbor_coord = torch.zeros(batch, H, W, 8, 3)
    rotated_slice_ceil = torch.ceil(rotated_slice)
    rotated_slice_floor = torch.floor(rotated_slice)
    nearest_neighbor_coord[:, :, :, 0, :] = rotated_slice_ceil
    nearest_neighbor_coord[:, :, :, 1, :] = rotated_slice_ceil
    nearest_neighbor_coord[:, :, :, 1, 0] = rotated_slice_floor[:, :, :, 0]
    nearest_neighbor_coord[:, :, :, 2, :] = rotated_slice_floor
    nearest_neighbor_coord[:, :, :, 2, 2] = rotated_slice_ceil[:, :, :, 2]
    nearest_neighbor_coord[:, :, :, 3, :] = rotated_slice_ceil
    nearest_neighbor_coord[:, :, :, 3, 1] = rotated_slice_floor[:, :, :, 1]
    nearest_neighbor_coord[:, :, :, 4, :] = rotated_slice_ceil
    nearest_neighbor_coord[:, :, :, 4, 2] = rotated_slice_floor[:, :, :, 2]
    nearest_neighbor_coord[:, :, :, 5, :] = rotated_slice_floor
    nearest_neighbor_coord[:, :, :, 5, 1] = rotated_slice_ceil[:, :, :, 1]
    nearest_neighbor_coord[:, :, :, 6, :] = rotated_slice_floor
    nearest_neighbor_coord[:, :, :, 7, :] = rotated_slice_floor
    nearest_neighbor_coord[:, :, :, 7, 0] = rotated_slice_ceil[:, :, :, 0]

    # Distance(volume of sub-cubic) from the nearest neighbor points
    distances_from_neighbors = torch.zeros(batch, H, W, 8)
    for i in range(8):
        distances_from_neighbors[:, :, :, i] = torch.abs(
            torch.prod(rotated_slice - nearest_neighbor_coord[:, :, :, i, :],
                       dim=3))
    print(rotated_slice[0,0,0])
    print(nearest_neighbor_coord[0,0,0])
    # print(distances_from_neighbors[0,0,0])
    # print(torch.sum(distances_from_neighbors[0,0,0]))

    # print(rotated_slice[0])
    # print(nearest_neighbor_coord[0])
    # print(distances_from_neighbors[0])
    # print(distances_from_neighbors.shape)

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

    B, H, W = fft_proj.size()
    print(B, H, W)
    for b in range(B):
        for i in range(H):
            for j in range(W):
                # print(fft_proj[b][c][i][j])
                print(fft_proj[b][i][j])
                print(rotated_slice[b][i][j])
                print(p1[b][i][j])
                print(w1[b][i][j])
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
