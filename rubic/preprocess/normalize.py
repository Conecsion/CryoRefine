import numpy as np
import torch


def delete_abnorm(tensor, lowerbound=2, upperbound=98):
    tensor = torch.clip(tensor, torch.quantile(tensor, lowerbound),
                        torch.quantile(tensor, upperbound))
    return tensor

def norm(tensor, mean=0, std=1):
    """
    Z-Score normalization
    """
    tensor = (tensor - torch.mean(tensor)) / torch.std(tensor)
    return tensor * std + mean
