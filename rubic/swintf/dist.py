import torch


# mean = [22,57,14]
# std = [15.7, 2.9, 35.7]
mean = 22
std = 5.7

samples = mean + std * torch.randn(1000)
print(samples)
print(samples.mean())
print(samples.std())
