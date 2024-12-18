import torch


def percentile_normalize(arr, lowerbound=2, upperbound=98):
    """
    Normalize an array to 0-1
    percentile can be set to clip the array's maximum and minimum to remove the outliers
    """
    arr = torch.clip(arr, torch.quantile(arr, lowerbound),
                  torch.quantile(arr, upperbound))
    return (arr - torch.min(arr)) / (torch.max(arr) - torch.min(arr))


def normalize(arr, mean=0, std=1):
    """
    Z-Score normalization
    """
    arr = (arr - torch.mean(arr)) / torch.std(arr)
    return arr * std + mean
