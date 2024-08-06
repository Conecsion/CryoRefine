#!/usr/bin/env/ python

import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

class MyViT(nn.Module):
    def __init__(self):
        super(MyViT, self).__init__()

    def forward(self, images):
        pass
