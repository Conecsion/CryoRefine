import torch
import numpy as np
from configParser import embeddingDim

dummyDistance = 47.2


def loss_fn(inputEmbeddings, inputAngles, targetEmbeddings, targetAngles) -> torch.float32:
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
