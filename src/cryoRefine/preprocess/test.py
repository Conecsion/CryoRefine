from rotate3d import rotate3d
from normalize import norm
from mrc import *
from projection import projection, random_proj
import torch
import numpy as np
import os
import pandas as pd

dp = '../../../data/'

mrctensor = tiffstack2tensor(dp + 'emd_7770.tif')
# print(mrctensor.shape)
rt = rotate3d(mrctensor, [45, 45, 45])
tensor2tiffstack(rt, dp+'emd7770rotated.tif')
# print(rt.shape)

random_proj(mrctensor, 500, os.path.join(dp, 'train'), expand=False)

# img_labels = pd.read_csv(os.path.join(dp, 'train','labels.csv'))
# label = img_labels.iloc[0,1]
# print(label)
# print(type(label))
# print(img_labels)
