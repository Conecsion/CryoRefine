import torch
import torch.nn as nn
import torchvision

encoder = torchvision.models.vit_b_16(weights='DEFAULT')
