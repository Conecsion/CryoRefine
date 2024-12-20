import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP

from configParser import embeddingDim

dummyDistance = 47.2


def loss_fn(inputEmbeddings, inputAngles, targetEmbeddings,
            targetAngles) -> torch.float32:
    # inputEmbeddings: (batch, embeddingDim)
    # inputAngles: (batch, 3)
    batchedEmbeddingDistances = torch.nn.MSELoss()(
        inputEmbeddings, targetEmbeddings,
        reduction='none').sum(dim=1) / inputEmbeddings.shape[-1]
    # embeddingDistance = (batchedEmbeddingDistances ** 0.5).sum

    sphericalDistance = dummyDistance

    def arcLength(ang1, ang2):
        ang1 = ang1 % (2 * np.pi)
        ang2 = ang2 % (2 * np.pi)
        angDiff = abs(ang1 - ang2)
        acuteAngDiff = min(angDiff, 2 * np.pi - angDiff)

    # return torch.nn.MSELoss()(embeddingDistance, sphericalDistance)


def setup(rank, world_size, backend='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def ddpTrain(rank, world_size, inputs: torch.Tensor, optimizer, loss_fn):
    setup(rank, world_size)
    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = optimizer
    optimizer.zero_grad()
    outputs = ddp_model(inputs)
    loss_fn().backward()
    optimizer.step()

    cleanup()


def run(fn, world_size):
    mp.spawn(fn,
             args=(world_size, inputs, optimizer, loss_fn),
             nprocs=world_size,
             join=True)
