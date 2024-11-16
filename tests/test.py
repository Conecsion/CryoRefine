from context import rubic
# import rubic.preprocess.io as io
# import rubic.postprocess.backprojection as bp
import torch
from torchvision.transforms import Normalize

# vol = torch.from_numpy(io.read_img('../data/emd_7770.mrc'))
# proj = torch.from_numpy(io.read_img('../data/emd7770proj/2_143_6.tif'))
# sim_proj = proj.repeat(7, 1, 1)
#
# sim_angles = torch.tensor([10., 0, 0])
# sim_angles = sim_angles.repeat(7, 1)
# bp.backprojection(sim_proj, sim_angles)

tensor = torch.randn(7, 128, 128)
tensor = tensor.sum(dim=0)
print(tensor.shape)
tensor = tensor.unsqueeze(0)
print(tensor.shape)
y = Normalize(mean=0, std=1)(tensor)
