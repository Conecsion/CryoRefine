"""
Tool functions for 2D Tensor operations
"""

import torch
from torchvision import transforms
# import torch.nn.functional as F

def norm(tensor:torch.Tensor, mean=0, std=1) -> torch.Tensor:
    """
    Z-Score normalization for batched grayscale image tensor
    """
    # tensor = (tensor - torch.mean(tensor)) / torch.std(tensor)
    # return tensor * std + mean
    return transforms.Normalize(mean=mean, std=std)(tensor)
