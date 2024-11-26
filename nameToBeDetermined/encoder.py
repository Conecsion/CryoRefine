import torch
import torch.nn as nn
import torchvision

encoder = torchvision.models.vit_b_16(weights='DEFAULT')
encoder.conv_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
encoder.heads = nn.Identity()
print(encoder)
x = torch.randn(12,1,224,224)
y = encoder(x)
print(x.shape)
print(y.shape)
