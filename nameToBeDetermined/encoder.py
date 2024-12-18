import torch
import torch.nn as nn
import torchvision
from configParser import embeddingDim

encoder = torchvision.models.vit_b_16(weights='DEFAULT')
encoder.conv_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
encoder.heads = nn.Sequential(
        nn.Linear(768, embeddingDim),
        nn.GELU(),
        nn.Linear(embeddingDim, embeddingDim),
        )
