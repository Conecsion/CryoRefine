import numpy as np
import torch
import math


def rotate2D(img: torch.Tensor, angleDegree: float) -> torch.Tensor:
    '''
    Rotate the input img by angleDegree counter-clockwise
    '''
    angleRadian = math.radians(angleDegree)
    rotationMatrix = torch.tensor([[np.cos(angleRadian), -np.sin(angleRadian)],
                                   [np.sin(angleRadian),
                                    np.cos(angleRadian)]])
    center = torch.tensor([img.shape[0] // 2, img.shape[1] // 2])
    rotatedImg = torch.zeros_like(img)

    for i in range(rotatedImg.shape[0]):
        for j in range(rotatedImg.shape[1]):
            originalCoords = torch.tensor([i,j]) - center
            rotatedCoords = torch.mm(rotationMatrix, originalCoords.T)
            rotatedCoords = rotatedCoords + center

            x, y = torch.round(rotatedCoords).int()

            if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]:
                rotatedImg[i,j] = img[x, y]

    return rotatedImg

if __name__ == '__main__':
    import os
    # import cv2
    # import matplotlib.pyplot as plt
    img = torch.zeros(100, 100)
    img[0:50, 50] = 1
    print(img)
