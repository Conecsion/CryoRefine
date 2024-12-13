import torch
import torch.nn as nn
import torchvision


class ParticleEncoder(nn.Module):

    def __init__(self, embeddingSize):
        super(ParticleEncoder, self).__init__()
        self.vit = torchvision.models.vit_b_16(weights='DEFAULT')
        self.vit.conv_proj = nn.Conv2d(1,
                                       768,
                                       kernel_size=(16, 16),
                                       stride=(16, 16))
        self.vit.heads = nn.Sequential(nn.Linear(768, embeddingSize),
                                       nn.GELU(),
                                       nn.Linear(embeddingSize, embeddingSize))
        self.distanceRatio = torch.exp(torch.randn(1))

    def forward(self, x):
        x = self.vit(x)


def loss_fn(inputEmbeds, targetEmbeds, inputAngles, targetAngles):
    # inputEmbeds shape [batch, embeddingDim]
    # inputAngles shape [batch, 3]

    embedDim = inputEmbeds.shape[-1]


    # embedDistances shape: [batch,]
    embedDistances = torch.sqrt(
        torch.nn.MSELoss(reduction='none')(
            inputEmbeds, targetEmbeds).sum(dim=1) / embedDim)


if __name__ == '__main__':
    model = ParticleEncoder(embeddingSize=1000)
    print(model)
    print(model.distanceRatio)
