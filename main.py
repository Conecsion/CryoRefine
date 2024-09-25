import torch
from src.cryoRefine.preprocess.rotate3d import rotate3d
from src.cryoRefine.postprocess.backprojection import backprojection

if __name__ == '__main__':
    imgs = torch.rand(4, 1, 338, 338)
    # angles = torch.rand(4, 3)
    angles = torch.tensor([90,0,0]).unsqueeze(0).repeat(4)
    backprojection(imgs, angles)
    # tmp = torch.rand(4, 4)
    # torch.floor(tmp)
    # torch.ceil(tmp)
