import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

theta = torch.tensor([np.pi/4])
R = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]])
a = torch.tensor([1., 0])
print(R @ a)
grid = F.affine_grid(R.unsqueeze(0).unsqueeze(0), (1,1,10,10))
